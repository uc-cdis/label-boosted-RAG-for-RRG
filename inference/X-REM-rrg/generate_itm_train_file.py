import argparse
import copy
import json
import queue
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def main(args):

    train_impressions = pd.read_json(args.train_path)
    train_impressions = train_impressions.set_index("dicom_id")
    dicom_ids = set(train_impressions.index)
    train_impressions_chexbert = pd.read_csv(args.train_chexbert_path)

    dataset = defaultdict(lambda: queue.Queue())

    for idx, row in train_impressions_chexbert.iterrows():
        dicom_id, study_id, subject_id = (
            row["dicom_id"],
            row["study_id"],
            row["subject_id"],
        )
        if dicom_id not in dicom_ids:
            continue
        label = tuple(row[4:])
        assert len(label) == 14, "length of the chexbert label detected is not 14"
        dataset[label].put(
            {
                "dicom_id": dicom_id,
                "study_id": study_id,
                "sentence": row["Report Impression"],
                "image": train_impressions["image"][dicom_id],
                "label": None,
            }
        )

    total_keys = list(dataset.keys())
    total_keys_freq = np.array(
        [dataset[k].qsize() for k in total_keys], dtype=np.float64
    )
    total_keys_freq /= np.sum(total_keys_freq)

    train_files = []
    edits = np.zeros(14)
    for key in tqdm(dataset.keys()):
        for i in range(dataset[key].qsize()):
            if i % 1000 == 0:
                print(i, dataset[key].qsize())
            el = dataset[key].get()
            dataset[key].put(el)

            positive = copy.deepcopy(el)
            positive["label"] = "positive"
            train_files.append(positive)

            hard_negative = copy.deepcopy(el)
            hard_negative["label"] = "negative"
            hard_negative_keys = []
            hard_negative_freq = []
            for edit_distance in range(1, 14):
                for cand in dataset.keys():
                    if cand == key:
                        continue
                    v = np.array(key) - np.array(cand)
                    if np.sum(np.abs(v)) <= edit_distance:
                        hard_negative_keys.append(cand)
                        hard_negative_freq.append(dataset[cand].qsize())
                if len(hard_negativel_keys) > 0:
                    edits[edit_distance] += 1
                    break

            hard_negative_freq = np.array(hard_negative_freq, dtype=np.float64)
            hard_negative_freq /= np.sum(hard_negative_freq)

            x = np.random.choice(len(hard_negative_keys), 1, p=hard_negative_freq)[0]
            hard_negative_keys = hard_negative_keys[x]
            hard_negative_cand = dataset[hard_negative_keys].get()
            hard_negative["sentence"] = hard_negative_cand["sentence"]
            hard_negative["hard_negative_dicom_id"] = hard_negative_cand["dicom_id"]
            hard_negative["edit_distance"] = edit_distance
            dataset[hard_negative_key].put(hard_negative_cand)
            train_files.append(hard_negative)

            negative = copy.deepcopy(el)
            negative["label"] = "negative"
            while True:
                negative_key = total_keys[
                    np.random.choice(np.arange(len(total_keys)), p=total_keys_freq)
                ]
                if negative_key != key:
                    break

            negative_cand = dataset[negative_key].get()
            dataset[negative_key].put(negative_cand)
            negative["sentence"] = negative_cand["sentence"]
            negative["negative_sentence_dicom_id"] = negative_cand["dicom_id"]
            negative["edit_distance"] = np.sum(
                np.abs(np.array(negative_key) - np.array(key))
            )
            train_files.append(negative)

    with open(args.save_path, "w") as f:
        json.dump(train_files, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        help="path to the train file (in csv format) used to create the trainining set for image-text matching fine-tuning",
    )
    parser.add_argument(
        "--train_chexbert_path",
        help="path to the chexbert labels (in csv format) for the train file",
    )
    parser.add_argument("--save_path", help="path to dump the output")
    args = parser.parse_args()
    main(args)
