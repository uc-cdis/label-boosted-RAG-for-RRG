import argparse
import os

import clip
import pandas as pd
import torch
from tqdm import tqdm
from utils import nonpretrained_params


def encode_texts(imps, model, device):
    trimmed_impressions = imps
    with torch.no_grad():
        imp_toks = clip.tokenize(
            trimmed_impressions, context_length=model.context_length
        ).to(device)
        embeddings = model.encode_text(imp_toks)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (
        args.clip_pretrained
    ):  # clip model is pretrained on chest X-rays, uses different architecture
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else:
        model = clip.CLIP(**nonpretrained_params)
        print("Loaded in clip model.")

    model.load_state_dict(torch.load(args.clip_model_path, map_location=device))
    model = model.to(device)

    impressions = pd.read_csv(args.data_path)["report"]
    # fill null impressions
    impressions = impressions.fillna("None")

    impressions_size = impressions.shape[0]

    bs = args.batch_size
    num_batches = impressions_size // bs
    tensors = []
    for i in tqdm(range(num_batches)):
        batch = impressions[bs * i : bs * i + bs]
        print(type(batch))
        _weights = encode_texts(batch, model, device)
        tensors.append(_weights)
    _weights = encode_texts(impressions[bs * num_batches :], model, device)
    tensors.append(_weights)

    clip_embeddings = torch.cat(tensors)
    print(impressions.shape, clip_embeddings.shape)
    out_data = (impressions, clip_embeddings)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_data, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate clip embeddings for a training corpus (either sentence level or report level"
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        required=True,
        help="name of clip model state dictionary for generating embeddings",
    )
    parser.add_argument(
        "--clip_pretrained",
        action="store_true",
        help="Whether clip model was first pre-trained on natural images",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path of csv file containing training corpus (either sentence level or report level)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="name for saved corpus embeddings (include .pt extension)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Batch size for generating clip embeddings",
    )
    args = parser.parse_args()

    main(args)
