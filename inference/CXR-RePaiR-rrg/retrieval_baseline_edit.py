import os
import time
from cgi import test
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_valid_indices(train_path, test_path):
    train_df = pd.read_csv(train_path, index_col=False)
    test_study_ids = pd.read_csv(test_path, index_col=False)["study_id"]
    train_df["study_id"] = train_df["filename"].astype(str).str[1:-4].astype(int)
    train_study_ids = train_df["study_id"]
    contained = train_study_ids.isin(test_study_ids)
    not_present_indices = list(contained[contained == False].index)
    return not_present_indices


class MIMICEncodingsDataset(data.Dataset):
    def __init__(self, encodings_path):
        encodings_file = h5py.File(encodings_path)
        self.encodings = encodings_file.get("encodings")
        self.reports = encodings_file.get("reports")
        self.indices = get_valid_indices(
            "/deep/group/data/med-data/mimic_cxr_impressions.csv",
            "/deep/group/data/med-data/mimic-cxr-jpg-split/bootstrap_test/reports.csv",
        )

    def __len__(self):
        return np.shape(self.encodings)[0]
        # return len(self.indices)

    def __getitem__(self, idx):
        return self.encodings[idx], self.reports[idx]
        # train_index = self.indices[idx]
        # return self.encodings[train_index], self.reports[train_index]


class MIMICDataset(data.Dataset):
    """Represents an abstract HDF5 dataset."""

    def __init__(self, img_path, txt_path, size=None, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, "r")["cxr_unprocessed"]
        self.txt_dset = pd.read_csv(txt_path)["report"]
        self.transform = transform

    def __len__(self):
        return len(self.txt_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]  # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx]  # python str

        img = torch.from_numpy(img)  # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)
        sample = {"img": img, "txt": txt}

        return sample


class MIMICTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, "r")["cxr"]
        self.transform = transform

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        img = self.img_dset[idx]  # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)

        img = torch.from_numpy(img)  # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)
        return img


class CheXpertDataset(data.Dataset):
    def __init__(self, img_path, transform=None):
        super().__init__()

        imgs_df = pd.read_csv(img_path)
        root_path = "/deep/group/CheXpert/CodaLab/"
        self.scale = 320
        self.transform = transform
        self.paths = []
        for _path in imgs_df["Path"]:
            if "view1" not in _path:
                continue  # TODO: check how to aggregate studies, consider them independent??
            _pth = _path.replace("CheXpert-v1.0", "")
            _pth = Path(root_path + _pth)
            self.paths.append(_pth)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # img = Image.open(self.paths[idx]).resize((self.scale, self.scale)).convert('RGB')
        _np_img = np.asarray(
            Image.open(self.paths[idx]).resize(
                (self.scale, self.scale), resample=Image.BICUBIC
            )
        )  # these images all have diff sizes!
        _np_img = _np_img.astype("float32")
        img = np.expand_dims(_np_img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)  # goes from H x W x C to C x H x W
        return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SET VARIABLES
save_train_encodings = False
retrieve_most_similar_train = True
batch_size = 256
chexpert_filepath = "/deep/group/CheXpert/CodaLab/test_image_paths.csv"
cxr_filepath = "/deep/group/data/med-data/cxr.h5"
txt_filepath = "/deep/group/data/med-data/mimic_cxr_impressions.csv"
encodings_type = "chexpert"
out_path = "/deep/u/markendo/CXR-RePaiR/results/CheXpert/Retrieval-Baseline/generated_reports.csv"
if encodings_type == "chexpert":
    mimic_h5py_path = "/deep/group/report-clip/resnet_mimic_encodings/chexpert.h5"
    # out_path = '/deep/u/markendo/R2Gen/out/retrieval_baseline_mimic_generated_reports.csv'
    state_dict_path = "/deep/u/markendo/aihc-winter19-robustness/chexpert-model/classification_model_checkpoints/resnet18/1z6xfh2n/epoch=1-chexpert_competition_AUROC=0.88_v0.ckpt"
elif encodings_type == "moco-cxr":
    mimic_h5py_path = "/deep/group/report-clip/resnet_mimic_encodings/moco-cxr.h5"
    # out_path = '/deep/group/report-clip/aihc-win21-clip/notebooks/eval/out/moco_normalized/moco_normalized_generated_reports.csv'
    state_dict_path = "/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments/ntruongv/r8w1n416s_20201202_resnet18-chexpert-0.0001-pretrained_20201202-093038_SLURM1931462/checkpoint_0000.pth.tar"

time_before = time.time()

# load resnet 18 model
model = models.resnet18(pretrained=False)
model.fc = Identity()
state_dict = torch.load(state_dict_path)
state_dict = state_dict["state_dict"]
if encodings_type == "chexpert":
    for key in list(state_dict.keys()):
        new_key = key.replace("model.model.", "")
        if new_key != "fc.bias" and new_key != "fc.weight":
            state_dict[new_key] = state_dict[key]
        del state_dict[key]
elif encodings_type == "moco-cxr":
    for key in list(state_dict.keys()):
        if "encoder_q" in key:  # discard encoder_k, only use query not key
            new_key = key.replace("module.encoder_q.", "")
            state_dict[new_key] = state_dict[key]
        del state_dict[key]
    del state_dict["fc.0.weight"]
    del state_dict["fc.0.bias"]
    del state_dict["fc.2.weight"]
    del state_dict["fc.2.bias"]
model.load_state_dict(state_dict)
model = torch.nn.DataParallel(model)
model.to(device)

if save_train_encodings:
    mimic_transform = transforms.Compose(
        [
            # means computed from sample in `cxr_stats` notebook
            transforms.Normalize(
                (101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)
            ),
        ]
    )
    mimic_dset = MIMICDataset(
        img_path=cxr_filepath, txt_path=txt_filepath, transform=mimic_transform
    )
    mimic_loader = data.DataLoader(mimic_dset, batch_size=batch_size)
    # get the images and impressions

    if not os.path.exists(Path(mimic_h5py_path).parent.absolute()):
        os.makedirs(Path(mimic_h5py_path).parent.absolute())

    dset_size = len(mimic_loader.dataset)
    with h5py.File(mimic_h5py_path, "w") as f:
        with torch.no_grad():
            encodings_dset = f.create_dataset("encodings", shape=(dset_size, 512))
            reports_dset = f.create_dataset(
                "reports", shape=(dset_size,), dtype=h5py.string_dtype()
            )
            for i, pack in enumerate(tqdm(mimic_loader)):
                imgs = pack["img"]
                resnet_encodings = model(imgs)
                encodings_arr = resnet_encodings.data.cpu().numpy()
                reports = pack["txt"]

                start_index = i * batch_size
                if i == len(mimic_loader) - 1:
                    encodings_dset[start_index:] = encodings_arr
                    reports_dset[start_index:] = reports
                else:
                    encodings_dset[start_index : start_index + batch_size] = (
                        encodings_arr
                    )
                    reports_dset[start_index : start_index + batch_size] = reports
    saved_file = h5py.File(mimic_h5py_path)
    print(saved_file.get("encodings")[0])
    print(saved_file.get("reports")[0])


# retrieve train examples that are most similar to train set encodings
if retrieve_most_similar_train:
    # MIMIC-CXR test set
    # CXR_FILEPATH = '/deep/group/data/med-data/mimic_test_cxr.h5'
    # CXR_FILEPATH = '/deep/group/data/med-data/mimic-cxr-jpg-split/bootstrap_test/cxr.h5'
    # mimic_transform = transforms.Compose([
    #     # means computed from sample in `cxr_stats` notebook
    #     transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    # ])
    # # mimic_transform = transforms.Compose([
    # #             transforms.Resize((224, 224)),
    # #             transforms.ToTensor(),
    # #             transforms.Normalize((0.485, 0.456, 0.406),
    # #                                 (0.229, 0.224, 0.225))])
    # mimic_test_dset = MIMICTestDataset(CXR_FILEPATH, transform=mimic_transform)
    # loader = data.DataLoader(mimic_test_dset, shuffle=False, batch_size=batch_size)

    # CheXpert test set
    chexpert_transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            # transforms.Normalize((.5020, .5020, .5020),(.085585, .085585, .085585)),
            transforms.Normalize(
                (129.4185, 129.4185, 129.4185), (73.3378, 73.3378, 73.3378)
            )
        ]
    )
    torch_chexpert_dset = CheXpertDataset(
        img_path=chexpert_filepath, transform=chexpert_transform
    )
    chexpert_loader = data.DataLoader(torch_chexpert_dset, batch_size=batch_size)

    # data_mean = next(iter(chexpert_loader))
    # print(data_mean[0].mean(), data_mean[0].std())

    train_encodings_dset = MIMICEncodingsDataset(mimic_h5py_path)
    train_encodings_loader = data.DataLoader(train_encodings_dset, batch_size=256)

    output_reports = []

    with torch.no_grad():
        for i, images in enumerate(chexpert_loader):
            images = images.to(device)
            test_encodings_batch = model(images)
            batch_size = len(test_encodings_batch)
            highest_similarities = np.array([-1.0] * batch_size)
            best_reports = [""] * batch_size
            for train_encodings in tqdm(train_encodings_loader):
                train_encodings_batch = train_encodings[0]

                # Using torch operations (for gpu)
                train_encodings_batch = train_encodings_batch.to(device)
                test_encodings_batch_norm = (
                    test_encodings_batch / test_encodings_batch.norm(dim=1)[:, None]
                )
                train_encodings_batch_norm = (
                    train_encodings_batch / train_encodings_batch.norm(dim=1)[:, None]
                )
                sim = torch.mm(
                    test_encodings_batch_norm,
                    train_encodings_batch_norm.transpose(0, 1),
                )
                maxes = torch.max(sim, dim=1)

                # Using np operations (for cpu)
                # sim = cosine_similarity(test_encodings_batch.cpu(), train_encodings_batch)
                # maxes = np.amax(sim, axis=1)

                for minibatch_index in range(batch_size):
                    if (
                        maxes.values[minibatch_index]
                        > highest_similarities[minibatch_index]
                    ):
                        highest_similarities[minibatch_index] = maxes.values[
                            minibatch_index
                        ]
                        best_reports[minibatch_index] = train_encodings[1][
                            maxes.indices[minibatch_index]
                        ].decode("utf-8")
                        # best_reports[minibatch_index] = train_encodings[1][np.argmax(sim[minibatch_index])].decode("utf-8")
            output_reports.extend(best_reports)
    print(time.time() - time_before)
    _df = pd.DataFrame(output_reports)
    _df.columns = ["report"]
    _df.to_csv(out_path, index=False)
