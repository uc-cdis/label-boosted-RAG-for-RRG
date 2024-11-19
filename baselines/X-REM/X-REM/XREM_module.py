import argparse
import random

import models
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import utils
import yaml
from models.model_itm import ALBEF as ALBEF_itm
from models.model_retrieval import ALBEF as ALBEF_retrieval
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize
from tqdm import tqdm


class RETRIEVAL_MODULE:

    def __init__(
        self, mode, config, checkpoint, topk, input_resolution, delimiter, max_token_len
    ):

        self.mode = mode
        assert (
            mode == "cosine-sim" or mode == "image-text-matching"
        ), "mode should be cosine-sim or image-text-matching"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.config = yaml.load(open(config, "r"), Loader=yaml.Loader)
        self.input_resolution = input_resolution
        self.topk = topk
        self.max_token_len = max_token_len
        self.delimiter = delimiter
        self.itm_labels = {"negative": 0, "positive": 2}

        if mode == "cosine-sim":
            self.load_albef_retrieval(checkpoint)
        else:
            self.load_albef_itm(checkpoint)

    # adapted albef codebase
    # For Image-Text Matching, we use ALBEF fine-tuned on visual entailmet to perform binary classification (entail/nonentail)
    def load_albef_itm(self, checkpoint_path):
        model = ALBEF_itm(
            config=self.config,
            text_encoder="bert-base-uncased",
            tokenizer=self.tokenizer,
        ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        model = model.eval()
        self.model = model

    # adapted albef codebase
    def load_albef_retrieval(self, checkpoint_path):
        model = ALBEF_retrieval(
            config=self.config,
            text_encoder="bert-base-uncased",
            tokenizer=self.tokenizer,
        ).to(device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )
        state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped
        for key in list(state_dict.keys()):
            if "bert" in key:
                encoder_key = key.replace("bert.", "")
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        model = model.eval()
        self.model = model

    def predict(self, images_dataset, reports):
        if self.mode == "cosine-sim":
            embeddings = self.generate_embeddings(reports)
            return self.cosine_sim_predict(images_dataset, reports, embeddings)
        else:
            return self.itm_predict(images_dataset, reports)

    # adapted cxr-repair codebase
    def generate_embeddings(self, reports, batch_size=2000):
        # adapted albef codebase
        def _embed_text(report):
            with torch.no_grad():
                text_input = self.tokenizer(
                    report,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_token_len,
                    return_tensors="pt",
                ).to(self.device)
                text_output = self.model.text_encoder(
                    text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                    mode="text",
                )
                text_feat = text_output.last_hidden_state
                text_embed = F.normalize(self.model.text_proj(text_feat[:, 0, :]))
                text_embed /= text_embed.norm(dim=-1, keepdim=True)
            return text_embed

        num_batches = reports.shape[0] // batch_size
        tensors = []
        for i in tqdm(range(num_batches + 1)):
            batch = list(
                reports[batch_size * i : min(batch_size * (i + 1), len(reports))]
            )
            weights = _embed_text(batch)
            tensors.append(weights)
        embeddings = torch.cat(tensors)
        return embeddings

    # adapted cxr-repair codebase
    def select_reports(self, reports, y_pred):
        reports_list = []
        for i, simscores in tqdm(enumerate(y_pred)):
            idxes = np.argsort(np.array(simscores))[-1 * self.topk :]
            idxes = np.flip(idxes)
            report = ""
            for j in idxes:
                if self.mode == "cosine-sim":
                    cand = reports[j]
                else:
                    cand = reports[i][j]
                report += cand + self.delimiter
            reports_list.append(report)
        return reports_list

    # adapted albef codebase
    def itm_predict(self, images_dataset, reports):
        y_preds = []
        bs = 100
        for i in tqdm(range(len(images_dataset))):
            image = images_dataset[i].to(self.device, dtype=torch.float)
            image = torch.unsqueeze(image, axis=0)
            image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            preds = torch.Tensor([]).to(self.device)
            local_reports = reports[i]
            for idx in range(0, len(local_reports), bs):
                try:
                    text = self.tokenizer(
                        local_reports[idx : idx + bs],
                        padding="longest",
                        return_tensors="pt",
                    ).to(self.device)
                    output = self.model.text_encoder(
                        text.input_ids,
                        attention_mask=text.attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    prediction = self.model.cls_head(output.last_hidden_state[:, 0, :])
                    positive_score = prediction[:, self.itm_labels["positive"]]
                except:
                    positive_score = torch.Tensor([0]).cuda()

                preds = torch.cat([preds, positive_score])
            idxes = torch.squeeze(preds).detach().cpu().numpy()
            y_preds.append(idxes)

        return self.select_reports(reports, y_preds)

    # adapted cxr-repair codebase
    def cosine_sim_predict(self, images_dataset, reports, embeddings):
        def softmax(x):
            return np.exp(x) / sum(np.exp(x))

        def embed_img(images):
            images = images.to(self.device, dtype=torch.float)
            image_features = self.model.visual_encoder(images)
            image_features = self.model.vision_proj(image_features[:, 0, :])
            image_features = F.normalize(image_features, dim=-1)
            return image_features

        y_pred = []
        image_loader = torch.utils.data.DataLoader(images_dataset, shuffle=False)
        with torch.no_grad():
            for image in tqdm(image_loader):
                image_features = embed_img(image)
                logits = image_features @ embeddings.T
                logits = np.squeeze(logits.to("cpu").numpy(), axis=0).astype("float64")
                norm_logits = (logits - logits.mean()) / (logits.std())
                probs = softmax(norm_logits)
                y_pred.append(probs)

        y_pred = np.array(np.array(y_pred))
        return self.select_reports(reports, y_pred)
