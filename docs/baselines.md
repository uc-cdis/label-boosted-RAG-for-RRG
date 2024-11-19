# Baseline Results

We compare against 3 retrieval based methods and 3 supervised fine-tuned methods from the literature as our baseline results. While we provide static clones of each of the methods, we made modifications as needed to get things to run. These changes are detailed within each method's specific section below. Our linting tools also formatted all code.

* Retrieval methods:
  * [CXR-RePaiR](#cxr-repair): Generates impression as the retrieval of the most similar report(s) via crossmodal embedding retrieval by a model trained with CLIP. [GitHub](https://github.com/rajpurkarlab/CXR-RePaiR).
  * [CXR-ReDonE](#cxr-redone): Similar to CXR-RePaiR except the joint embedding model was trained via ALBEF and the retrieval data are preprocessed by a langauge model to remove references to prior studies. [GitHub](https://github.com/rajpurkarlab/CXR-ReDonE).
  * [X-REM](#x-rem): Similar to as CXR-ReDonE except for a new custom similarity score used in ALBEF training and no data preprocessing is done, instead rather doing postprocessing of intermediate retrieved reports to filter to the most relevant reports. [GitHub](https://github.com/rajpurkarlab/X-REM).
* Fine-tuned methods:
  * [CheXagent](#chexagent): Generates findings and impression. Findings are generated as a concatenation of generated localized findings, i.e. findings describing specific anatomical regions, by following prompted instructions. Uses a decoder-only language model with image features prepended to the tokens. [GitHub](https://github.com/Stanford-AIMI/CheXagent).
  * [CXRMate](#cxrmate): Generates findings and impression. A full encoder-decoder transformer that jointly generates findings and impression and postprocesses the output into separate findings and impression sections. [HuggingFace](https://huggingface.co/aehrc/cxrmate-rrg24).
  * [RGRG](#rgrg): Generates findings. Trained to first do object detection of anatomical regions with pathologies or abnormalities, then generates region specific findings. Final findings are a concatenation of localized findings. Uses a decoder-only language model with image features prepended to the tokens. [GitHub](https://github.com/ttanida/rgrg).

## Prepare baseline inference data

Some methods require special data preparation; these instructions are detailed under the specific section describing steps to run the method. Otherwise, extract data to run baseline model inference.

```bash
conda activate labrag
python rrg/prepare_inference_data.py \
--data_dir /path/to/mimic-cxr/files/ \
--split_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv \
--metadata_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-metadata.csv \
--report_csv /path/to/mimic-cxr/mimic_cxr_sectioned.csv \
--true_label_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv \
--output_dir /path/to/data/dir/
```

## CXR-RePaiR

We provide a static clone of the CXR-RePaiR repo at commit hash `c11fa85`. We added `environment.yml`, modified `data_preprocessing/create_bootstrapped_testset.py` to save the `dicom_id`, and modified `gen_corpus_embeddings.py` and `run_test.py` to remove hardcoded data paths.

1. Create and activate the `cxrrepair` conda environment.
    ```bash
    cd baselines/CXR-RePaiR
    conda env create -f environment.yml
    conda activate cxrrepair
    ```
1. Run CXR-RePaiR data preprocessing.
    ```bash
    python data_preprocessing/split_mimic.py \
    --report_files_dir /path/to/mimic-cxr/notes/ \
    --split_path /path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz \
    --out_dir /path/to/cxrrepair/data/dir/

    python data_preprocessing/extract_impressions.py \
    --dir /path/to/cxrrepair/data/dir/

    python data_preprocessing/create_bootstrapped_testset.py \
    --dir /path/to/cxrrepair/data/dir/ \
    --bootstrap_dir /path/to/cxrrepair/data/dir/test/ \
    --cxr_files_dir /path/to/mimic-cxr/files/
    ```
1. Manually download the model weights `clip-imp-pretrained_128_6_after_4.pt` from [this Box folder](https://stanfordmedicine.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml).
1. Generate whole-report embeddings of the retrieval corpus.
    ```bash
    python gen_corpus_embeddings.py \
    --clip_model_path /path/to/clip-imp-pretrained_128_6_after_4.pt \
    --clip_pretrained \
    --data_path /path/to/cxrrepair/data/dir/mimic_train_impressions.csv \
    --out /path/to/cxrrepair/data/dir/mimic_train_impressions_embeddings.pt
    ```
1. Generate the final reports using top-1 retrieval of whole reports.
    ```bash
    python run_test.py \
    --corpus_embeddings_name /path/to/cxrrepair/data/dir/mimic_train_impressions_embeddings.pt \
    --clip_model_path /path/to/clip-imp-pretrained_128_6_after_4.pt \
    --clip_pretrained \
    --out_dir /path/to/cxrrepair/outputs/ \
    --test_cxr_path /path/to/cxrrepair/data/dir/test/cxr.h5 \
    --topk 1
    ```

## CXR-ReDonE

We provide a static clone of the CXR-ReDonE repo at commit hash `b385988`. We modified `environment.yml` to change the environment name, bumped the version of pycocotools `2.0.4 → 2.0.5` to fix some installation error, and specified torch and torchvision with CUDA 11.3 to run on newer hardware. We also modified `remove_prior_refs.py` to prevent overwriting input data and to join multi-token words (i.e. `" ##" → ""`). Finally, we modified `CXR_ReDonE_pipeline.py` to save the study ids of each generated reports.

1. [Follow steps 1 and 2 of the CXR-RePaiR](#cxr-repair) section to prepare initial data if you have not already.
1. Create and activate the `cxrredone` conda environment.
    ```bash
    cd baselines/CXR-ReDonE
    conda env create -f environment.yml
    conda activate cxrredone
    ```
1. Remove prior references from impressions extracted by CXR-RePaiR. This step can take a long time.
    ```bash
    python remove_prior_refs.py \
    --dir /path/to/cxrrepair/data/dir/ \
    --out /path/to/cxrredone/data/dir/
    ```
1. Manually download the model weights `checkpoint_59.pth` from [this dropbox link](https://www.dropbox.com/s/b4tkf2z4v6wa4zj/checkpoint_59.pth?dl=0). We specifically use the model trained data with prior references omitted.
1. Generate the final reports using top-1 retrieval of whole reports with prior references removed.
    ```bash
    cd ALBEF

    python CXR_ReDonE_pipeline.py \
    --impressions_path /path/to/cxrredone/data/dir/mimic_train_impressions.csv \
    --img_path /path/to/cxrrepair/data/dir/test/cxr.h5 \
    --save_path /path/to/cxrredone/outputs/generations.csv \
    --albef_retrieval_ckpt /path/to/checkpoint_59.pth \
    --albef_retrieval_top_k 1
    ```

## X-REM

We provide a static clone of the X-REM repo at commit hash `c9c3571` with ifcc subrepo at commit hash `0c5c24c`. Per the author instructions, we move `m2trans_nli_filter.py` and `M2TransNLI.py` to the ifcc directory. We modified `X-REM/environment.yml` to change the environment name and specified torch and torchvision with CUDA 11.1 to run on newer hardware. We additionally modified `X-REM/ifcc/environment.yml` to change the environment name and specified the oldest possible torch and torchvision with CUDA 11.0, the minimum CUDA required to run on our hardware. We removed unused data files provided by the repo authors. Lastly, we add `X-REM/inference.py`, a short script to generate intermediate output, inspired by the authors' inference guidelines.

1. [Follow steps 1 and 2 of the CXR-RePaiR](#cxr-repair) section to prepare initial data if you have not already.
1. Create and activate the `xrem` conda environment. 
    ```bash
    cd baselines/X-REM
    conda env create -f environment.yml
    conda activate xrem
    ```
1. Manually download and unzip the `albef_checkpoint` directory from [this google drive link](https://drive.google.com/file/d/11UorBbh5cOcDfIzy_lCgMdn0zThvpDbp/view?usp=sharing).
1. Generate intermediate reports with top-10 scored reports.
    ```bash
    cd X-REM

    python inference.py \
    --img_path /path/to/cxrrepair/data/dir/test/cxr.h5 \
    --rep_path /path/to/cxrrepair/data/dir/mimic_train_impressions.csv \
    --albef_path /path/to/unzipped/albef_checkpoint/ \
    --save_dir /path/to/xrem/outputs/
    ```
1. Download additional models `model_medrad_19k` using provided tooling.
    ```bash
    cd ../ifcc
    ./resources/download.sh
1. Create and activate the `ifcc` conda environment.
    ```bash
    conda env create -f environment.yml
    conda activate ifcc
    ```
1. Postprocess the intermediate reports using the NLI filter.
    ```bash
    python m2trans_nli_filter.py \
    --m2trans_nli_model_path /path/to/model_medrad_19k/ \
    --input_path /path/to/xrem/outputs/itm_results_temp.csv \
    --save_path /path/to/xrem/outputs/generations.csv \
    --topk 2
    ```

## CXRMate

We specifically use CXRMate-RRG24, the authors' submisson to the RRG task of the ACL 2024 BioNLP workshop. We pin the model from its huggingface repository at commit hash `81c38cf`. We add `inference.py` based on the authors' inference guidelines and add `environment.yml`.

1. [Follow steps](#prepare-baseline-inference-data) to generate `findings.csv` and `impression.csv` if you have not already.
1. Create and cctivate the `cxrmate` conda environment.
    ```bash
    cd baselines/CXRMate
    conda env create -f environment.yml
    conda activate cxrmate
    ```
1. Generate findings and impression.
    ```bash
    python inference.py \
    --findings_csv /path/to/data/dir/findings.csv \
    --impression_csv /path/to/data/dir/impression.csv \
    --save_dir /path/to/cxrmate/outputs/
    ```

## CheXagent

We pin the model from its huggingface repository at commit hash `4934e91`. We add `inference.py` based on the authors' inference guidelines.

1. [Follow steps](#prepare-baseline-inference-data) to generate `findings.csv` and `impression.csv` if you have not already.
1. Activate the `cxrmate` environment (reusing the same environment).
    ```bash
    cd baselines/CheXagent
    conda activate cxrmate
    ```
1. Generates findings and impression. This takes a while.
    ```bash
    python inference.py \
    --findings_csv /path/to/data/dir/findings.csv \
    --impression_csv /path/to/data/dir/impression.csv \
    --save_dir /path/to/chexagent/outputs/
    ```

## RGRG

We provide a static clone of the RGRG repo at commit hash `9520b6d`. We modified `environment.yml` to change the environment name and specified torch and torchvision with CUDA 11.6 to run on newer hardware. We also modified `src/full_model/generate_reports_for_images.py` to save associated information for each generation.

1. [Follow steps](#prepare-baseline-inference-data) to generate `findings.csv` and `impression.csv` if you have not already.
1. Create and activate the `rgrg` conda environment.
    ```bash
    cd baselines/RGRG
    conda env create -f environment.yml 
    conda activate rgrg
    ```
1. Manually download the model weights `full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt` from [this google drive link](https://drive.google.com/file/d/1rDxqzOhjqydsOrITJrX0Rj1PAdMeP7Wy/view?usp=sharing).
1. Generate findings.
    ```bash
    python src/full_model/generate_reports_for_images.py \
    --model_path /path/to/full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt \
    --input_csv /path/to/data/dir/findings.csv \
    --output_csv /path/to/rgrg/outputs/generations.csv
    ```


## Postprocessing and evaluation metrics

Our evaluation tools expect model generations to be in a CSV file with columns `study_id`, `actual_text`, and `generated_text`. Run all cells in the notebook [`baselines/collate_baseline_results.ipynb`](../baselines/collate_baseline_results.ipynb) to postprocess results into the expected format.

<details>
<summary>Then run evaluation (expand for template).</summary>

```bash
conda activate labrag

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/cxrrepair_impression.csv \
--output_csv /path/to/cxrrepair_impression_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/cxrredone_impression.csv \
--output_csv /path/to/cxrredone_impression_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/xrem_impression.csv \
--output_csv /path/to/xrem_impression_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/cxrmate_impression.csv \
--output_csv /path/to/cxrmate_impression_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/chexagent_impression.csv \
--output_csv /path/to/chexagent_impression_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/cxrmate_findings.csv \
--output_csv /path/to/cxrmate_findings_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/chexagent_findings.csv \
--output_csv /path/to/chexagent_findings_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/rgrg_findings.csv \
--output_csv /path/to/rgrg_findings_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/cxrmate_both.csv \
--output_csv /path/to/cxrmate_both_METRICS.csv

python /path/to/labrag-repo/rrg/eval.py \
--report_csv /path/to/chexagent_both.csv \
--output_csv /path/to/chexagent_both_METRICS.csv
```

</details>
<br>


Though we use paired t-tests to compare results, we reorder the results by `study_id` when we plot comparisons so there is no need to enforce result order here.
