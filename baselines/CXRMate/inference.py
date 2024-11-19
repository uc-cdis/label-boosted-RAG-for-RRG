import argparse
import os

import pandas as pd
import torch
import transformers
from datasets import Dataset, Features, Image, Value
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="aehrc/cxrmate-rrg24",
        help="Path to the local cxrmate-rrg4 model or HuggingFace model name",
    )
    parser.add_argument(
        "--findings_csv",
        required=True,
        help="Path to the inference_findings_data.csv file",
    )
    parser.add_argument(
        "--impression_csv",
        required=True,
        help="Path to the inference_impression_data.csv file",
    )
    parser.add_argument(
        "--save_dir", required=True, help="Save path for generations.csv"
    )
    return parser.parse_args()


def generate_reports(
    *,  # enforce kwargs
    model_path: str,
    findings_path: str,
    impression_path: str,
    results_save_dir: str,
):
    findings_df = pd.read_csv(findings_path)
    impression_df = pd.read_csv(impression_path)

    union_df = pd.concat([findings_df, impression_df])
    union_df = union_df.drop_duplicates(subset=["study_id"])

    assert set(findings_df["study_id"]).union(set(impression_df["study_id"])) == set(
        union_df["study_id"]
    )

    data_dict = {
        "study_id": union_df["study_id"].tolist(),
        "dicom_id": union_df["dicom_id"].tolist(),
        "image_path": union_df["dicom_path"].tolist(),
        "image": union_df["dicom_path"].tolist(),
    }

    features = Features(
        {
            "image_path": Value("string"),
            "image": Image(),
            "study_id": Value("string"),
            "dicom_id": Value("string"),
        }
    )

    hf_dataset = Dataset.from_dict(data_dict, features)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        revision="81c38cf",
    )
    model = transformers.AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision="81c38cf",
    )

    device = "cuda" if torch.cuda.is_available else "cpu"

    model.to(device)
    transforms = v2.Compose(
        [
            v2.PILToTensor(),
            v2.Grayscale(num_output_channels=3),
            v2.Resize(size=model.config.encoder.image_size, antialias=True),
            v2.CenterCrop(size=[model.config.encoder.image_size] * 2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=model.config.encoder.image_mean, std=model.config.encoder.image_std
            ),
        ]
    )

    def apply_transforms(examples):
        transformed_images = [transforms(img) for img in examples["image"]]
        examples["image"] = transformed_images
        return examples

    hf_dataset = hf_dataset.map(
        apply_transforms,
        batched=True,
        batch_size=16,
        num_proc=1,
        remove_columns="image_path",
    )

    mbatch_size = 32

    # Custom collate function
    def collate_fn(batch):
        images = [torch.tensor(item["image"], dtype=torch.float32) for item in batch]

        # Stack images into a single tensor
        images = torch.stack(images)
        sids = [item["study_id"] for item in batch]
        dicoms = [item["dicom_id"] for item in batch]
        return {"images": images, "sids": sids, "dicoms": dicoms}

    dataloader = DataLoader(hf_dataset, batch_size=mbatch_size, collate_fn=collate_fn)

    # main loop
    results = {"study_id": [], "dicom_id": [], "findings": [], "impression": []}

    for batch in tqdm(dataloader):
        batch_ids = model.generate(
            pixel_values=batch["images"].unsqueeze(1).to(device),
            max_length=512,
            num_beams=4,
            bad_words_ids=[
                [tokenizer.convert_tokens_to_ids("[NF]")],
                [tokenizer.convert_tokens_to_ids("[NI]")],
            ],
        )

        batch_findings, batch_impression = model.split_and_decode_sections(
            batch_ids, tokenizer
        )
        results["study_id"].extend(batch["sids"])
        results["dicom_id"].extend(batch["dicoms"])
        results["findings"].extend(batch_findings)
        results["impression"].extend(batch_impression)

    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(results_save_dir, "generations.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    generate_reports(
        model_path=args.model,
        findings_path=args.findings_csv,
        impression_path=args.impression_csv,
        results_save_dir=args.save_dir,
    )
