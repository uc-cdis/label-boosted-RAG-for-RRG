import argparse

import h5py
import pandas as pd
from CXR_ReDonE_module import RETRIEVAL_MODULE


def main(args):
    df = pd.read_csv(args.impressions_path)
    impressions = df["report"].drop_duplicates().dropna().reset_index(drop=True)
    cosine_sim_module = RETRIEVAL_MODULE(
        impressions=impressions,
        mode="cosine-sim",
        config=args.albef_retrieval_config,
        checkpoint=args.albef_retrieval_ckpt,
        topk=args.albef_retrieval_top_k,
        input_resolution=256,
        img_path=args.img_path,
        delimiter="|",
        max_token_len=25,
    )
    cosine_sim_output = cosine_sim_module.predict()

    if args.use_ve:
        new_impressions = [
            el.split("|") for el in cosine_sim_output["Report Impression"]
        ]

        ve_module = RETRIEVAL_MODULE(
            impressions=new_impressions,
            mode="visual-entailment",
            config=args.albef_ve_config,
            checkpoint=args.albef_ve_ckpt,
            topk=args.albef_ve_top_k,
            input_resolution=384,
            img_path=args.img_path,
            delimiter="[SEP]",
            max_token_len=30,
        )
        ve_output = ve_module.predict()
        generated = ve_output["Report Impression"]
    else:
        generated = cosine_sim_output["Report Impression"]

    with h5py.File(args.img_path, "r") as h5:
        sids = h5["sid"][:]
    out_df = pd.DataFrame({"study_id": sids, "Report Impression": generated})
    out_df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impressions_path",
        default="../data/mimic_train_impressions.csv",
        help="path to the mimic train corpus",
    )
    parser.add_argument(
        "--img_path", default="../data/cxr.h5", help="path to the test file"
    )
    parser.add_argument("--save_path", default="example.csv")
    parser.add_argument("--use_ve", type=bool, default=False)
    parser.add_argument(
        "--albef_retrieval_config", default="configs/Retrieval_flickr.yaml"
    )
    parser.add_argument(
        "--albef_retrieval_ckpt", default="output/pretrain/checkpoint_59.pth"
    )
    parser.add_argument("--albef_retrieval_top_k", type=int, default=50)
    parser.add_argument("--albef_ve_config", default="configs/VE.yaml")
    parser.add_argument("--albef_ve_ckpt", default="output/ve/checkpoint_7.pth")
    parser.add_argument(
        "--albef_ve_top_k",
        default=10,
        type=int,
        help="url used to set up distributed training",
    )
    args = parser.parse_args()

    if not args.use_ve:
        args.albef_retrieval_top_k = 1

    main(args)
