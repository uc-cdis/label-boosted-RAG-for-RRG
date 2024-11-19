import argparse
import os

import h5py
import pandas as pd
from XREM import XREM
from XREM_config import XREMConfig
from XREM_dataset import CXRTestDataset_h5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        required=True,
        help="Path to the image dataset cxr.h5",
    )
    parser.add_argument(
        "--rep_path",
        required=True,
        help="Path to the report impression mimic_train_impressions.csv",
    )
    parser.add_argument(
        "--albef_path", required=True, help="Path to albef pretrained models "
    )
    parser.add_argument(
        "--save_dir", required=True, help="Path to save the generated results"
    )
    return parser.parse_args()


def generate_reports(
    *,  # enforce kwargs
    image_datapath: str,
    report_datapath: str,
    albef_ckpt_path: str,
    save_dir: str,
):
    dset_cosine = CXRTestDataset_h5(image_datapath, 256)
    dset_itm = CXRTestDataset_h5(image_datapath, 384)
    train_data = pd.read_csv(report_datapath)
    reports = train_data["report"].drop_duplicates().dropna().reset_index(drop=True)

    cwd = os.getcwd()
    config = XREMConfig(
        albef_retrieval_config=os.path.join(cwd, "configs/Cosine-Retrieval.yaml"),
        albef_retrieval_ckpt=os.path.join(
            albef_ckpt_path, "sample/pretrain/checkpoint_59.pth"
        ),
        albef_itm_config=os.path.join(cwd, "configs/ITM.yaml"),
        albef_itm_ckpt=os.path.join(albef_ckpt_path, "sample/ve/checkpoint_7.pth"),
    )

    xrem = XREM(config)
    itm_output = xrem(reports, dset_cosine, dset_itm)

    with h5py.File(image_datapath, "r") as h5:
        sids = h5["sid"][:]
    itm_df = pd.DataFrame({"study_id": sids, "Report Impression": itm_output})
    itm_df.to_csv(os.path.join(save_dir, "itm_results_temp.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    generate_reports(
        image_datapath=args.img_path,
        report_datapath=args.rep_path,
        albef_ckpt_path=args.albef_path,
        save_dir=args.save_dir,
    )
