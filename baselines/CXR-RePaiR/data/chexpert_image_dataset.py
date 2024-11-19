from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm


class CheXpertImageDataset(data.Dataset):
    def __init__(self, img_path, root_path, clip_pretrained=True):
        super().__init__()

        if clip_pretrained:
            input_resolution = 224
        else:
            input_resolution = 320

        all_images = []
        imgs_df = pd.read_csv(img_path)
        for _path in tqdm(imgs_df["Path"]):
            if "view1" not in _path:
                continue
            _pth = _path.replace("CheXpert-v1.0", "")
            _np_img = np.asarray(
                Image.open(Path(root_path + _pth)).resize(
                    (input_resolution, input_resolution), resample=Image.BICUBIC
                )
            )  # these images all have diff sizes
            _np_img = _np_img.astype("float32")
            all_images.append(_np_img)
        self.img_dset = np.array(all_images)

        normalize_fn = Normalize(
            (129.4185, 129.4185, 129.4185), (73.3378, 73.3378, 73.3378)
        )
        self.transform = Compose([normalize_fn])

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        sample = {"img": img}
        return sample
