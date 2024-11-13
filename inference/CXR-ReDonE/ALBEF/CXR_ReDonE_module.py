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
from CXR_ReDonE_dataset import CXRTestDataset, CXRTestDataset_h5
from models.model_retrieval import ALBEF as ALBEF_retrieval
from models.model_ve import ALBEF as ALBEF_ve
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize
from tqdm import tqdm


class RETRIEVAL_MODULE:

    def __init__(
        self,
        impressions,
        mode,
        config,
        checkpoint,
        topk,
        input_resolution,
        img_path,
        delimiter,
        max_token_len,
    ):

        self.mode = mode
        assert (
            mode == "cosine-sim" or mode == "visual-entailment"
        ), "mode should be cosine-sim or visual-entailment"
        self.impressions = impressions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.config = yaml.load(open(config, "r"), Loader=yaml.Loader)
        self.input_resolution = input_resolution
        self.topk = topk
        self.max_token_len = max_token_len
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (input_resolution, input_resolution), interpolation=Image.BICUBIC
                ),
                Normalize(
                    (101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)
                ),
            ]
        )
        self.dset = CXRTestDataset_h5(transform=self.transform, img_path=img_path)
        self.delimiter = delimiter
        self.ve_labels = {"contradiction": 0, "neutral": 1, "entailment": 2}

        if mode == "cosine-sim":
            self.load_albef_retrieval(checkpoint)
        else:
            self.load_albef_ve(checkpoint)

    # adapted albef codebase
    def load_albef_ve(self, checkpoint_path):
        model = ALBEF_ve(
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

    def predict(self):
        if self.mode == "cosine-sim":
            self.generate_embeddings()
            return self.cosine_sim_predict()
        else:
            return self.ve_predict()

    # adapted cxr-repair codebase
    def generate_embeddings(self):
        # adapted albef codebase
        def _embed_text(impression):
            with torch.no_grad():
                text_input = self.tokenizer(
                    impression,
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

        bs = 2000
        num_batches = self.impressions.shape[0] // bs
        tensors = []
        for i in tqdm(range(num_batches + 1)):
            batch = self.impressions[bs * i : bs * i + bs]
            batch = list(
                self.impressions[bs * i : min(bs * i + bs, len(self.impressions))]
            )
            weights = _embed_text(batch)
            tensors.append(weights)
        self.embeddings = torch.cat(tensors)

    # adapted cxr-repair codebase
    def select_reports(self, y_pred):
        reports_list = []
        for i, simscores in tqdm(enumerate(y_pred)):
            idxes = np.argsort(np.array(simscores))[-1 * self.topk :]
            idxes = np.flip(idxes)
            report = ""
            for idx in idxes:
                if self.mode == "cosine-sim":
                    cand = self.impressions[idx]
                else:
                    cand = self.impressions[i][idx]
                report += cand + self.delimiter
            reports_list.append(report)
        return reports_list

    # adapted albef codebase
    def ve_predict(self):
        y_preds = []
        bs = 100
        for i in tqdm(range(len(self.dset))):
            image = self.dset[i].to(self.device, dtype=torch.float)
            image = torch.unsqueeze(image, axis=0)
            image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            preds = torch.Tensor([]).to(self.device)
            local_impressions = self.impressions[i]
            for idx in range(0, len(local_impressions), bs):
                try:
                    text = self.tokenizer(
                        local_impressions[idx : idx + bs],
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
                    entailment_score = prediction[:, self.ve_labels["entailment"]]
                except:
                    entailment_score = torch.Tensor([0]).cuda()

                preds = torch.cat([preds, entailment_score])
            idxes = torch.squeeze(preds).detach().cpu().numpy()
            y_preds.append(idxes)

        reports_list = self.select_reports(y_preds)
        _df = pd.DataFrame(reports_list)
        _df.columns = ["Report Impression"]
        return _df

    # adapted cxr-repair codebase
    def cosine_sim_predict(self):
        def _softmax(x):
            return np.exp(x) / sum(np.exp(x))

        def _embed_img(data):
            images = data.to(self.device, dtype=torch.float)
            image_features = self.model.visual_encoder(images)
            image_features = self.model.vision_proj(image_features[:, 0, :])
            image_features = F.normalize(image_features, dim=-1)
            return image_features

        def _compute_cosine_sim():
            y_pred = []
            loader = torch.utils.data.DataLoader(self.dset, shuffle=False)
            with torch.no_grad():
                for data in tqdm(loader):
                    image_features = _embed_img(data)
                    logits = image_features @ self.embeddings.T
                    logits = np.squeeze(logits.to("cpu").numpy(), axis=0).astype(
                        "float64"
                    )
                    norm_logits = (logits - logits.mean()) / (logits.std())
                    probs = _softmax(norm_logits)
                    y_pred.append(probs)
            return np.array(y_pred)

        y_pred = _compute_cosine_sim()
        reports_list = self.select_reports(y_pred)
        _df = pd.DataFrame(reports_list)
        _df.columns = ["Report Impression"]
        return _df
