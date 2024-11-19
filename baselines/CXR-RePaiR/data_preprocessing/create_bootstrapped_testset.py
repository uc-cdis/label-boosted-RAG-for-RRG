import argparse
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def filter_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv, index_col=[0])
    filtered = df[df["report"].notnull()]
    filtered.to_csv(output_csv, index=False)


def create_cxr_h5(reports_path, cxr_files_dir, cxr_outpath):
    cxr_paths, study_ids = get_cxr_paths(reports_path, cxr_files_dir)
    img_to_hdf5(cxr_paths, study_ids, cxr_outpath)


def get_cxr_paths(reports_path, cxr_files_dir):
    df = pd.read_csv(reports_path)
    df["Path"] = (
        cxr_files_dir
        + "p"
        + df["subject_id"].astype(str).str[:2]
        + "/p"
        + df["subject_id"].astype(str)
        + "/s"
        + df["study_id"].astype(str)
        + "/"
        + df["dicom_id"]
        + ".jpg"
    )
    cxr_paths = df["Path"]
    study_ids = df["study_id"].astype(str)
    return cxr_paths, study_ids


def img_to_hdf5(cxr_paths, study_ids, out_filepath, resolution=320):
    dset_size = len(cxr_paths)
    with h5py.File(out_filepath, "w") as h5f:
        img_dset = h5f.create_dataset("cxr", shape=(dset_size, resolution, resolution))

        str_dtype = h5py.string_dtype(encoding="utf-8")
        sid_dset = h5f.create_dataset("sid", shape=(dset_size,), dtype=int)
        did_dset = h5f.create_dataset("did", shape=(dset_size,), dtype="S44")
        for idx, (path, sid) in enumerate(tqdm(zip(cxr_paths, study_ids))):
            dicom_id = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path)
            img = preprocess(img, resolution)
            img_dset[idx] = img
            sid_dset[idx] = sid
            did_dset[idx] = dicom_id


def preprocess(img, desired_size):
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)

    new_img = Image.new("L", (desired_size, desired_size))
    new_img.paste(
        img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )
    return new_img


def create_bootstrap_indices(bootstrap_dir, n=10):
    test_h5_path = os.path.join(bootstrap_dir, "cxr.h5")
    h5 = h5py.File(test_h5_path, "r")["cxr"]
    dset_size = len(h5)

    for i in range(n):
        subdir = os.path.join(bootstrap_dir, f"{str(i)}/")
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        indices = np.random.choice(dset_size, size=dset_size, replace=True)
        np.savetxt(subdir + "indices.txt", indices, fmt="%d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the bootstrapped test-set, including the report impressions, images, and bootstrap indices."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory where test report impressions is stored",
    )
    parser.add_argument(
        "--bootstrap_dir",
        type=str,
        required=False,
        default="bootstrap_test",
        help="directory where bootstrap test set will be stored",
    )
    parser.add_argument(
        "--cxr_files_dir",
        type=str,
        required=True,
        help="mimic-cxr-jpg files directory containing chest X-rays",
    )
    args = parser.parse_args()

    unfiltered_test_impressions_path = os.path.join(
        args.dir, "mimic_test_impressions.csv"
    )

    if not os.path.exists(args.bootstrap_dir):
        os.makedirs(args.bootstrap_dir)

    # filter out impressions that are nan
    filter_csv(
        unfiltered_test_impressions_path,
        os.path.join(args.bootstrap_dir, "reports.csv"),
    )

    # crop and save cxrs as an h5 file
    cxr_outpath = os.path.join(args.bootstrap_dir, "cxr.h5")
    create_cxr_h5(
        os.path.join(args.bootstrap_dir, "reports.csv"), args.cxr_files_dir, cxr_outpath
    )

    # create files of indices to sample for testing
    create_bootstrap_indices(args.bootstrap_dir)
