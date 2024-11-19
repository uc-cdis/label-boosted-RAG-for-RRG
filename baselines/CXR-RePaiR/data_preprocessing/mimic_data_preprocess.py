import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

train_csv_filepath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_impressions.csv"
)
test_csv_filepath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_test_impressions.csv"
)
filtered_train_csv_filepath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_impressions_filtered.csv"
)
filtered_test_csv_filepath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_test_impressions_filtered.csv"
)

final_train_csv_filepath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_impressions_final.csv"
)

train_cxr_outpath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_cxr_impressions.h5"
)
test_cxr_outpath = (
    "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_test_cxr_impressions.h5"
)

old_csv_filepath = "/deep/group/data/med-data/train_unprocessed.csv"
old_h5_filepath = "/deep/group/data/med-data/cxr.h5"


def clean_train_csv(input_csv):
    non_functioning_indices = [127893]  # cannot read jpeg
    df = pd.read_csv(input_csv)
    final_test_csv = df.drop(non_functioning_indices)
    final_test_csv.to_csv(final_train_csv_filepath, index=False)


def filter_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv, index_col=[0])
    filtered = df[df["report"].notnull()]
    filtered.to_csv(output_csv, index=False)


def get_cxr_paths(filepath):
    mimic_img_dir = "/deep/group/data/mimic-cxr/mimic-cxr-jpg/2.0.0/files/"
    df = pd.read_csv(filepath)
    df["Path"] = (
        mimic_img_dir
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
    return cxr_paths


def match_and_save_h5(cxr_paths, out_filepath):
    old_df = pd.read_csv(old_csv_filepath)
    old_cxr_paths = old_df["Path"]
    contained = cxr_paths.isin(old_cxr_paths)
    not_present_indices = list(contained[contained == False].index)
    old_cxr_paths_index = pd.Index(old_cxr_paths)
    old_h5 = h5py.File(old_h5_filepath, "r")["cxr_unprocessed"]
    dset_size = len(cxr_paths)
    resolution = 320
    with h5py.File(out_filepath, "w") as h5f:
        img_dset = h5f.create_dataset(
            "cxr_unprocessed", shape=(dset_size, resolution, resolution)
        )
        for index, path in tqdm(cxr_paths.items(), total=cxr_paths.size):
            if index in not_present_indices:  # have to process
                img = Image.open(path)
                img = preprocess(img, resolution)
                img_dset[index] = img
            else:  # already in old h5
                matching_index = old_cxr_paths_index.get_loc(path)
                img_dset[index] = old_h5[matching_index]


#             img = Image.open(path)
# #             img = preprocess(img, resolution)
# #             img_dset[idx] = img

# print(matching_index)
# break
# matching_index = old_cxr_paths[old_cxr_paths == path].index
# if len(matching_index) == 0:
#     print(index)
# else:
#     pass


# print(len(matching_index))
# print(matching_index)
# print(index, path)
# break
# print(cxr_paths)
# print(old_cxr_paths)

# filtered_data = pd.read_csv(filtered_train_csv_filepath)
# removed_non_present = filtered_data.drop(not_present_indices)
# print(filtered_data)
# print(removed_non_present)
# filtered_train_csv_filepath
# print(not_present_indices)
# print(contained)


# print(contained.value_counts())
# print(cxr_paths)
# print(old_df)


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


# def img_to_hdf5(cxr_paths, out_filepath, resolution=320):
#     dset_size = len(cxr_paths)
#     with h5py.File(out_filepath,'w') as h5f:
#         img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution))

#         for idx, path in enumerate(tqdm(cxr_paths)):
#             if idx < 127893:
#                 continue
#             elif idx < 127895:
#                 print(path)
#             img = Image.open(path)
#             img = preprocess(img, resolution)
#             img_dset[idx] = img


def main():
    # filter_csv(train_csv_filepath, filtered_train_csv_filepath)
    # remove_non_present_examples(filtered_train_csv_filepath, out)
    # clean_train_csv(filtered_train_csv_filepath)
    cxr_paths = get_cxr_paths(final_train_csv_filepath)

    # match_and_save_h5(cxr_paths, train_cxr_outpath)
    # img_to_hdf5(cxr_paths, train_cxr_outpath)


if __name__ == "__main__":
    main()
