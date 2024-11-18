import argparse
import os
from pathlib import Path

import h5py
import torch
from _data import DEFAULT_IMG_EMBED_KEY, DEFAULT_IMG_PROJ_KEY
from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 16
DEFAULT_FILE_EXT = ".jpg"


def extract_image_features(
    *,  # enforce kwargs
    mimic_cxr: str,
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
    for root, _, files in os.walk(mimic_cxr):
        for file in files:
            if file.endswith(file_ext):
                path = os.path.join(root, file)
                paths.append(path)
    paths = sorted(paths)
    ds = ImageDataset(paths)
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_biovil_t_image_encoder()
    model = model.eval()
    model = model.to(device)
    for batch in tqdm(dl):
        ims = batch[0].to(device)
        patient_ids, subject_ids, dicom_ids = batch[1]
        with torch.inference_mode():
            out = model(ims)
        img_embeds = out.img_embedding.cpu().numpy()
        img_projs = out.projected_global_embedding.cpu()
        # Save raw proj, cosine sim normalizes anyways
        # img_projs = F.normalize(img_projs, dim=-1)
        img_projs = img_projs.numpy()

        with h5py.File(output_h5, "a") as h5:
            for img_embed, img_proj, patient_id, subject_id, dicom_id in zip(
                img_embeds, img_projs, patient_ids, subject_ids, dicom_ids
            ):
                h5[f"{embed_key}/{patient_id}/{subject_id}/{dicom_id}"] = img_embed
                h5[f"{proj_key}/{patient_id}/{subject_id}/{dicom_id}"] = img_proj


class ImageDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = create_chest_xray_transform_for_inference(
            resize=512, center_crop_size=448
        )

    def __getitem__(self, index) -> torch.Tensor:
        path = self.paths[index]
        ids = os.path.splitext(path)[0].split(os.path.sep)[-3:]
        im = load_image(Path(path))
        im = self.transform(im)
        return im, ids  # (patient_id, subject_id, dicom_id)

    def __len__(self) -> int:
        return len(self.paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mimic_cxr",
        required=True,
        help="Path to mimic_cxr",
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
        mimic_cxr=args.mimic_cxr,
        output_h5=args.output_h5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        file_ext=args.file_ext,
        embed_key=args.embed_key,
        proj_key=args.proj_key,
    )
