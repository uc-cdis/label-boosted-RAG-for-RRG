# report generation code for chexagent
"""
# RUN

python /path/to/inference_cxrmate.py \
-- model /path/to/cxrmate-rrg24 model \
--findings_csv /path/to/inference_findings_data.csv \
--impression_csv /path/to/inference_impression_data.csv
"""
import argparse
import io
import os

import pandas as pd
import requests
import torch
from datasets import Dataset, Features
from datasets import Image as DImage
from datasets import Value
from PIL import Image
from rich import print
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(
        images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
    ).to(device=device, dtype=dtype)
    output = model.generate(**inputs, generation_config=generation_config)[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the local chexagent model or HuggingFace model name",
    )
    parser.add_argument(
        "--device", required=True, help="cuda device to use for loading model"
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
        "--savepath", required=True, help="Save path for generations.csv"
    )


def generate_reports(
    *,  # enforce kwargs
    model=str,
    device=str,
    findings_path=str,
    impression_path=str,
    results_savepath=str,
):
    # step 1: Setup constant
    # device = "cuda:1"
    device = "cuda" if torch.cuda.is_available else "cpu"
    dtype = torch.float16

    # step 2: Load Processor and Model
    model_path = model
    # model_path = "/opt/gpudata/rrg-data-2/inference-all/inf-models/chexagent/CheXagent-8b"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    # step 3: Fetch the images
    # image_path = "https://upload.wikimedia.org/wikipedia/commons/3/3b/Pleural_effusion-Metastatic_breast_carcinoma_Case_166_%285477628658%29.jpg"
    # image_path = "/opt/gpudata/mimic-cxr/files/p10/p10046166/s53492798/18f0fd6d-f513afc9-e4aa8de2-bc5ac0d6-ea3daaff.jpg"
    # images = Image.open(image_path).convert("RGB")
    # images = [download_image(image_path)]

    # step 3: Fetch the dataset
    # findings_df = pd.read_csv("/opt/gpudata/rrg-data-2/inference-all/inference_findings_data.csv")
    # impression_df = pd.read_csv("/opt/gpudata/rrg-data-2/inference-all/inference_impression_data.csv")

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
            "image": DImage(),
            "study_id": Value("string"),
            "dicom_id": Value("string"),
        }
    )

    hf_dataset = Dataset.from_dict(data_dict, features)

    # step 4: Generate the Findings  and Impression section
    all_studies = []
    all_findings = []
    all_impression = []

    for i in trange(len(hf_dataset)):
        images = hf_dataset[i]["image"]
        findings = []
        for anatomy in anatomies:
            prompt = f'Describe "{anatomy}"'
            response = generate(
                images, prompt, processor, model, device, dtype, generation_config
            )
            findings.append(response)

        prompt = f"Generate impression"
        impression = generate(
            images, prompt, processor, model, device, dtype, generation_config
        )

        all_studies.append(hf_dataset[i]["study_id"])
        all_findings.append(" ".join(findings))
        all_impression.append(impression)

    # pd.DataFrame(
    #     {
    #         "study_id" : all_studies,
    #         "findings" : all_findings,
    #         "impression" : all_impression
    #     }
    # ).to_csv("/opt/gpudata/rrg-data-2/inference-all/inf-results/chexagent/generations.csv")

    pd.DataFrame(
        {
            "study_id": all_studies,
            "findings": all_findings,
            "impression": all_impression,
        }
    ).to_csv(os.path.join(results_savepath, "generations.csv"))


if __name__ == "__main__":
    anatomies = [
        "Airway",
        "Breathing",
        "Cardiac",
        "Diaphragm",
        "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)",
    ]
    args = parse_args()
    generate_reports(
        model=args.model,
        findings_path=args.findings_csv,
        impression_path=args.impression_csv,
        results_savepath=args.savepath,
    )
