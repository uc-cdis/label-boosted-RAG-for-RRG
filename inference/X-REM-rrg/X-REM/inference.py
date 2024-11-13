import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
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


def generate_reports(
    *,  # enforce kwargs
    image_datapath: str,
    report_datapath: str,
    albef_ckpt_path: str,
    save_dir: str,
):
    cwd = os.getcwd()
    # cxr_h5_path = "/opt/gpudata/rrg-data-2/inference-all/inf-temp/cxr-repair/bootstrapped_test_set/cxr.h5"
    cxr_h5_path = image_datapath
    dset_cosine = CXRTestDataset_h5(cxr_h5_path, 256)
    dset_itm = CXRTestDataset_h5(cxr_h5_path, 384)
    # train_data = pd.read_csv("/opt/gpudata/rrg-data-2/inference-all/inf-temp/cxr-repair/mimic_train_impressions.csv")
    train_data = pd.read_csv(report_datapath)
    # print(train_data.columns)
    reports = train_data["report"].drop_duplicates().dropna().reset_index(drop=True)

    from XREM_config import XREMConfig

    # config = XREMConfig(
    #         albef_retrieval_config = "/opt/gpudata/anirudh/git-repos/X-REM-anirudh/X-REM/configs/Cosine-Retrieval.yaml",
    #         albef_retrieval_ckpt = "/opt/gpudata/rrg-data-2/inference-all/inf-models/x-rem/albef_checkpoint/sample/pretrain/checkpoint_59.pth",
    #         albef_itm_config = "/opt/gpudata/anirudh/git-repos/X-REM-anirudh/X-REM/configs/ITM.yaml",
    #         albef_itm_ckpt = "/opt/gpudata/rrg-data-2/inference-all/inf-models/x-rem/albef_checkpoint/sample/ve/checkpoint_7.pth",
    # )

    config = XREMConfig(
        albef_retrieval_config=f"{cwd}/X-REM/configs/Cosing-Retrieval.yaml",
        albef_retrieval_ckpt=f"{albef_ckpt_path}/albef_checkpoint/sample/pretrain/checkpoint_59.pth",
        albef_itm_config=f"{cwd}/X-REM/configs/ITM.yaml",
        albef_itm_ckpt=f"{albef_ckpt_path}/albef_checkpoint/sample/ve/checkpoint_7.pth",
    )

    # test dataloading functionality and sids order
    image_loader = torch.utils.data.DataLoader(dset_cosine, shuffle=False)

    all_sids = []
    with torch.no_grad():
        for batch in tqdm(image_loader):
            all_sids.extend(batch[1].numpy().astype(str))

    # pd.DataFrame(
    #     {
    #         "study_id" : all_sids,
    #     }
    # ).to_csv("/opt/gpudata/rrg-data-2/inference-all/inf-temp/x-rem/xrem_sids.csv", index=False)

    pd.DataFrame(
        {
            "study_id": all_sids,
        }
    ).to_csv(save_dir + "xrem_sids.csv", index=False)

    from XREM import XREM

    xrem = XREM(config)
    itm_output = xrem(reports, dset_cosine, dset_itm)

    itm_df = pd.DataFrame(itm_output, columns=["Report Impression"])
    # itm_df.to_csv("/opt/gpudata/rrg-data-2/inference-all/inf-temp/x-rem/itm_results_temp.csv", index = False)
    itm_df.to_csv(save_dir + "itm_results_temp.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    generate_reports(
        image_datapath=args.image_path,
        report_datapath=args.rep_path,
        albef_ckpt_path=args.albef_path,
        # save_dir = "/opt/gpudata/rrg-data-2/inference-all/inf-temp/x-rem/"
        save_dir=args.save_dir,
    )
