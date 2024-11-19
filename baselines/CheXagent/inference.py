import argparse
import os

import pandas as pd
import torch
from datasets import Dataset, Features
from datasets import Image as DImage
from datasets import Value
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="StanfordAIMI/CheXagent-8b",
        help="Path to the local chexagent model or HuggingFace model name",
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
        "--save_dir",
        required=True,
        help="Save path for generations.csv",
    )
    return parser.parse_args()


def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(
        images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
    ).to(device=device, dtype=dtype)
    output = model.generate(**inputs, generation_config=generation_config)[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True)
    return response


def generate_reports(
    *,  # enforce kwargs
    model_path: str,
    findings_path: str,
    impression_path: str,
    results_save_dir: str,
):
    device = "cuda" if torch.cuda.is_available else "cpu"
    dtype = torch.float16

    # step 2: Load Processor and Model
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision="4934e91",
    )
    generation_config = GenerationConfig.from_pretrained(
        model_path,
        revision="4934e91",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        revision="4934e91",
    ).to(device)

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

    out_df = pd.DataFrame(
        {
            "study_id": all_studies,
            "findings": all_findings,
            "impression": all_impression,
        }
    )
    out_df.to_csv(os.path.join(results_save_dir, "generations.csv"), index=False)


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
        model_path=args.model,
        findings_path=args.findings_csv,
        impression_path=args.impression_csv,
        results_save_dir=args.save_dir,
    )
