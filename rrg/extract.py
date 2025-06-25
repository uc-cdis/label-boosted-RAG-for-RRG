import argparse
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args

import h5py
import torch
from _data import DEFAULT_IMG_EMBED_KEY, DEFAULT_IMG_PROJ_KEY
from gloria.models.vision_model import ImageEncoder as GLoRIAImageEncoder
from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import Normalize
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 16
DEFAULT_FILE_EXT = ".jpg"
MODEL_T = Literal["biovil-t", "gloria", "resnet50"]


def extract_image_features(
    *,  # enforce kwargs
    model_type: MODEL_T,
    model_path: str | None = None,
    input_path: str,
    output_h5: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    file_ext: str = DEFAULT_FILE_EXT,
    embed_key: str = DEFAULT_IMG_EMBED_KEY,
    proj_key: str = DEFAULT_IMG_PROJ_KEY,
):
    if os.path.exists(output_h5):
        print()
        print("------------------------------------------------------")
        print(
            "WARNING: output file already exists, is this expected?",
            output_h5,
        )
        print("------------------------------------------------------")
        print()

    paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(file_ext):
                path = os.path.join(root, file)
                paths.append(path)
    paths = sorted(paths)
    ds = ImageDataset(paths=paths, transform_type=model_type)
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    if model_type == "biovil-t":
        model = get_biovil_t_image_encoder()
    elif model_type == "gloria":
        model = get_gloria_image_encoder(model_path)
    elif model_type == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # remove classification layer but still use forward method
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval()
    model = model.to(device)
    for batch in tqdm(dl):
        ims = batch[0].to(device)
        patient_ids, study_ids, dicom_ids = batch[1]
        with torch.inference_mode():
            if model_type == "biovil-t":
                out = model(ims)
                img_embeds = out.img_embedding.cpu().numpy()
                img_projs = out.projected_global_embedding.cpu().numpy()
            elif model_type == "gloria":
                img_embeds = model(ims)
                img_projs = model.global_embedder(img_embeds).cpu().numpy()
            elif model_type == "resnet50":
                img_embeds = [None] * len(ims)
                img_projs = model(ims)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        with h5py.File(output_h5, "a") as h5:
            for img_embed, img_proj, patient_id, study_id, dicom_id in zip(
                img_embeds, img_projs, patient_ids, study_ids, dicom_ids
            ):
                h5[f"{embed_key}/{patient_id}/{study_id}/{dicom_id}"] = img_embed
                h5[f"{proj_key}/{patient_id}/{study_id}/{dicom_id}"] = img_proj


def get_gloria_image_encoder(model_path: str) -> nn.Module:
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            text=SimpleNamespace(embedding_dim=768),
            vision=SimpleNamespace(
                model_name="resnet_50",
                pretrained=True,
                freeze_cnn=False,
            ),
        )
    )
    model = GLoRIAImageEncoder(cfg)

    # load pretrained model weights
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in sd.items():
        k = k.replace("gloria.img_encoder.", "")
        if k.startswith("gloria"):
            continue
        new_state_dict[k] = v
    model.load_state_dict(
        new_state_dict,
        strict=True,
    )

    return model


class ImageDataset(Dataset):
    def __init__(
        self,
        *,  # enforce kwargs
        paths: list[str],
        transform_type: MODEL_T,
    ):
        self.paths = paths
        if transform_type == "biovil-t":
            self.transform = create_chest_xray_transform_for_inference(
                resize=512,
                center_crop_size=448,
            )
        elif transform_type == "gloria":
            self.transform = create_chest_xray_transform_for_inference(
                resize=256,
                center_crop_size=224,
            )
            self.transform.transforms.append(
                Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                )
            )
        elif transform_type == "resnet50":
            # Based on ResNet50 - ImageNet 1K v2
            # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
            self.transform = create_chest_xray_transform_for_inference(
                resize=232,
                center_crop_size=224,
            )
            self.transform.transforms.append(
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def __getitem__(self, index) -> torch.Tensor:
        path = self.paths[index]
        ids = os.path.splitext(path)[0].split(os.path.sep)[-3:]
        try:
            im = load_image(Path(path))
        except Exception as e:
            print(ids)
            raise e
        im = self.transform(im)
        return im, ids  # (patient_id, study_id, dicom_id)

    def __len__(self) -> int:
        return len(self.paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        choices=get_args(MODEL_T),
        help="CXR embedding model type",
    )
    parser.add_argument(
        "--model_path",
        required=False,
        help="Path to model weights, only needed for some model types",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to input directory containing images where the final file hierarchy is patient_id/study_id/image_id",
    )
    parser.add_argument(
        "--output_h5",
        required=True,
        help="Path to output h5",
    )
    parser.add_argument(
        "--batch_size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of samples in a batch",
    )
    parser.add_argument(
        "--num_workers",
        default=DEFAULT_NUM_WORKERS,
        type=int,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--file_ext",
        default=DEFAULT_FILE_EXT,
        help="Image file extension",
    )
    parser.add_argument(
        "--embed_key",
        default=DEFAULT_IMG_EMBED_KEY,
        help="Name of image embedding",
    )
    parser.add_argument(
        "--proj_key",
        default=DEFAULT_IMG_PROJ_KEY,
        help="Name of image projection",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    extract_image_features(
        model_type=args.model_type,
        model_path=args.model_path,
        input_path=args.input_path,
        output_h5=args.output_h5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        file_ext=args.file_ext,
        embed_key=args.embed_key,
        proj_key=args.proj_key,
    )
