import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


# Adapted cxr-repair
# input: .h5 file containing the images
class CXRTestDataset_h5(data.Dataset):
    def __init__(self, img_path, input_resolution):
        super().__init__()
        self.img_dset = h5py.File(img_path, "r")["cxr"]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (input_resolution, input_resolution), interpolation=Image.BICUBIC
                ),
                transforms.Normalize(
                    (101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)
                ),
            ]
        )

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
        return img


# Adapted cxr-repair
# input: files containing paths to the image files
class CXRTestDataset(data.Dataset):
    def __init__(self, target_files, input_resolution):
        super().__init__()
        self.files = target_files
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (input_resolution, input_resolution), interpolation=Image.BICUBIC
                ),
                transforms.Normalize(
                    (101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)
                ),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fpath = self.files[idx]
        desired_size = 320
        img = Image.open(fpath)
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new("L", (desired_size, desired_size))
        new_img.paste(
            img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )
        img = np.asarray(new_img, np.float64)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img
