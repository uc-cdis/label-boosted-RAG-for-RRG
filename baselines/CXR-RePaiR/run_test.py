import argparse
import os
from pathlib import Path
from xml.etree.ElementPath import prepare_descendant

import clip
import h5py
import numpy as np
import pandas as pd
import torch
from data import CheXpertImageDataset, MIMICImageDataset
from PIL import Image
from torch.utils import data
from tqdm import tqdm
from utils import nonpretrained_params


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load in model
    if args.clip_pretrained:
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else:
        model = clip.CLIP(**nonpretrained_params)
        print("Loaded in clip model.")

    model.load_state_dict(torch.load(args.clip_model_path, map_location=device))
    model = model.to(device)

    # precalculated corpus clip embeddings
    corpus_embeddings_path = args.corpus_embeddings_name
    raw_impressions, text_embeddings = get_text_embeddings(
        corpus_embeddings_path, args.compression
    )

    if args.dataset == "MIMIC-CXR":
        dset = MIMICImageDataset(
            img_path=args.test_cxr_path, clip_pretrained=args.clip_pretrained
        )
    else:
        dset = CheXpertImageDataset(
            img_path=args.test_cxr_path,
            root_path=args.CheXpert_root_path,
            clip_pretrained=args.clip_pretrained,
        )
    loader = torch.utils.data.DataLoader(
        dset, shuffle=False, batch_size=args.batch_size
    )

    # test the dataset
    # for data in loader:
    #     print(data['img'].shape)
    #     print(data['sid'].shape)
    #     print(data['sid'])
    #     break

    sids_from_h5_dset = []
    for data in loader:
        sids_from_h5_dset.extend(data["sid"].tolist())

    # select top report/sentences
    y_pred, y_sids = predict(loader, text_embeddings, model, device, topk=args.topk)

    # pd.DataFrame.from_dict(
    #     {
    #         "loader_sids" : sids_from_h5_dset,
    #         "preds_sids" : y_sids,
    #     }

    # ).to_csv("/opt/gpudata/rrg-data-2/inference-all/inf-temp/cxr-repair/sids_h5.csv")

    # # save
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_path = args.out_dir + "/generations.csv"
    save_reports(y_pred, y_sids, raw_impressions, out_path)


def get_text_embeddings(corpus_embeddings_path, correlations_path):
    # Get the pre-generated text embeddings and corresponding impressions
    (raw_impressions, text_embeddings) = torch.load(corpus_embeddings_path)

    if correlations_path != "":
        reports_dict = torch.load(correlations_path)
        selected_reports = list(reports_dict.keys())
        raw_impressions = raw_impressions[selected_reports]
        text_embeddings = text_embeddings[selected_reports]
        print(raw_impressions.shape, text_embeddings.shape)
    raw_impressions.index = range(len(raw_impressions))
    return (raw_impressions, text_embeddings)


def predict(loader, text_embeddings, model, device, topk=1):
    # y_pred = []
    predicted_corpus_indices = torch.zeros([len(loader.dataset), topk]).to(device)
    # corpus_sids = torch.zeros(len(loader.dataset))
    corpus_sids = []
    batch_index = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data["img"].to(device)
            sids = data["sid"]

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_embeddings.T
            preds = torch.argsort(logits, dim=-1, descending=True)[
                :, :topk
            ]  # get topk reports
            predicted_corpus_indices[batch_index : batch_index + preds.size(0), :] = (
                preds  # save batch to predictions
            )
            # corpus_sids[batch_index:batch_index+preds.size(0)] = sids
            corpus_sids.extend(sids.numpy().astype(str))

            batch_index += preds.size(0)  # batch size
    return predicted_corpus_indices.to("cpu").numpy(), corpus_sids


def save_reports(outputs, sids, raw_impressions, out_path):
    reports_list = []
    sids_list = []
    for preds, sid in zip(outputs, sids):
        # convert output to a report
        report = ""
        for pred in preds:
            report += raw_impressions[pred] + " "
        reports_list.append(report)
        sids_list.append(str(sid))
    # write reports to a csv
    # _df = pd.DataFrame(reports_list)
    # _df.columns = ["report"]
    # _df.to_csv(out_path, index=False)

    pd.DataFrame(
        {
            "study_id": sids_list,
            "generated_text": reports_list,
        }
    ).to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the top report/sentences based on CXR-RePaiR method"
    )
    parser.add_argument(
        "--corpus_embeddings_name",
        type=str,
        required=True,
        help="name of corpus embeddings file generated by CLIP",
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        required=True,
        help="name of clip model state dictionary",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="directory to save outputted generated reports",
    )
    parser.add_argument(
        "--test_cxr_path",
        type=str,
        required=True,
        help="path of test X-rays, .h5 file for MIMIC and .csv for CheXpert",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        default=1,
        help="number top sentences to retrieve",
    )
    parser.add_argument(
        "--compression",
        type=str,
        required=False,
        default="",
        help="set to file path for compression",
    )
    parser.add_argument(
        "--clip_pretrained",
        action="store_true",
        help="Whether clip model was first pre-trained on natural images, should be same as used for generating corpus embeddings",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MIMIC-CXR",
        choices=["CheXpert", "MIMIC-CXR"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--CheXpert_root_path",
        type=str,
        required=False,
        help="Root to folder where CheXpert test is",
    )
    parser.add_argument("--batch_size", type=int, required=False, default=4)
    args = parser.parse_args()

    main(args)
