# Baseline Results

We compare against 3 retrieval based methods and 3 supervised fine-tuned methods from the literature as our baseline results.

* Retrieval methods:
  * [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR)
  * [CXR-ReDonE](https://github.com/rajpurkarlab/CXR-ReDonE)
  * [X-REM](https://github.com/rajpurkarlab/X-REM)
* Fine-tuned methods:
  * [RGRG](https://github.com/ttanida/rgrg)
  * [CheXagent](https://github.com/Stanford-AIMI/CheXagent)
  * [CXRMate](https://huggingface.co/aehrc/cxrmate-rrg24)

## Prepare baseline inference data

Some methods require special data preparation; these instructions are detailed under the specific section describing steps to run the method. Otherwise, extract data to run baseline model inference.

```bash
conda activate labrag
python rrg/prepare_inference_data.py # specify args as needed
```

## CXR-RePaiR

We provide a static clone of the CXR-RePaiR repo at commit hash `c11fa85`. We added `environment.yml` and modified `gen_corpus_embeddings.py` and `run_test.py` to remove hardcoded data paths.

1. Create and activate the conda environment. 
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
1. Generate whole-report embeddings of the retrieval corpus. This step runs on the GPU so prefix with `CUDA_VISIBLE_DEVICES` as needed.
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

1. Follow steps 1 and 2 of the CXR-RePaiR section to prepare initial data if you have not already.
1. Create and activate the conda environment. 
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

We provide a static clone of the X-REM repo at commit hash `c9c3571` with ifcc subrepo at commit hash `0c5c24c`. Per the author instructions, we move `m2trans_nli_filter.py` and `M2TransNLI.py` to the ifcc directory. We modified `X-REM/environment.yml` to change the environment name and specified torch and torchvision with CUDA 11.1 to run on newer hardware. We additionally modified `X-REM/ifcc/environment.yml` to change the environment name and specified the oldest possible torch and torchvision with CUDA 11.0, the minimum CUDA required to run on our hardware. We also remove unused data files provided by the repo authors.

1. Follow steps 1 and 2 of the CXR-RePaiR section to prepare initial data if you have not already.
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
1. Post-process the intermediate reports using the NLI filter.
    ```bash
    python m2trans_nli_filter.py \
    --m2trans_nli_model_path /path/to/model_medrad_19k/ \
    --input_path /path/to/xrem/outputs/itm_results_temp.csv \
    --save_path /path/to/xrem/outputs/generations.csv \
    --topk 2
    ```

# THINGS BELOW THIS LINE HAVE NOT BEEN REVIEWED

## 4. CXRMate-RRG24 

First setup the environment by running;
```bash
conda create env -f env.yml -n cxrmate_env
```
The CXRMate-RRG24 model generates both findings and impression separately, we choose to then concatenate the reports to form full reports including both the avaialable sections of findings and impression. 

* Activate the conda environment : 
    ```bash
    conda activate cxrmate_env
    ```
1. `inference_cxrmate.py` generates findings and impression.
    ```bash
    python /path/to/rrg-repo/inference/cxrmate/inference_cxrmate.py \
    --model /path/to/cxrmate-rrg24 model \
    --findings_csv /path/to/inference_findings_data.csv \
    --impression_csv /path/to/inference_impression_data.csv \
    --savepath /path/to/save/directory
    ```
2. Run `cxrmate` section in `inference-results.ipynb` to split the generations into the 3 splits, findings only, impression only and findings & impression

## 5. CheXagent

We will use the same environment as we did for CXRMATE-RRG24. CheXagent generates both findings and impression similar to CXRMATE, so we will concatenate them to have full reports.

* Activate the environment
    ```bash
    conda activate cxrmate_env
    ```

1. `inference_chexagent.py` generates findings and impression.
    ```bash
    python /path/to/rrg-repo/inference/chexagent/inference_chexagent.py \
    --model /path/to/chexagent model \
    --findings_csv /path/to/inference_findings_data.csv \
    --impression_csv /path/to/inference_impression_data.csv \
    --savepath /path/to/save/directory
    ```
2. Run `chexagent` section in `inference-results.ipynb` to split the generations into the 3 splits, findings only, impression only and findings & impression

### 6. RGRG
The model from the RGRG paper generates only findings so we will use the findings subset of the data. Download the full model checkpoint from the link in the [repo](https://github.com/ttanida/rgrg).

* Create and activate the conda environment 
    ```bash
    cd rgrg-rrg
    conda env create -f environment.yml 
    conda activate rgrg_env
    ```

1. `generate_reports_for_images.py` generates findings for a lit of image paths
    ```bash
        python ./src/full_model/generate_reports_for_images.py \
        --model_path /path/to/rgrg_full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt \
        --input_csv path/to/inference_findings_data.csv \
        --output_csv path/to/results/generations_findings.csv
    ```

1. Finally we will switch back into the rrg environment and run the eval script to generate the METRICS files for the 6 comparison models. Use the following run commands on each of the generated files to run evaluation metrics for radiology report generations
    ```bash
        python /path/to/rrg-repo/rrg/eval.py \
        --report_csv /path/to/generations_<split>.csv file \
        --output_csv /path/to/results/generations_<split>_METRICS.csv 
    ```
1. The resulting file structure should look like the following:
    ```
    ./path/to/output/directory
    ├── chexagent
    │   ├── generations.csv
    │   ├── generations_findings.csv
    │   ├── generations_findings_METRICS.csv
    │   ├── generations_full_reports.csv
    │   ├── generations_full_reports_METRICS.csv
    │   ├── generations_impression.csv
    │   └── generations_impression_METRICS.csv
    ├── cxr-mate
    │   ├── generations.csv
    │   ├── generations_findings.csv
    │   ├── generations_findings_METRICS.csv
    │   ├── generations_full_reports.csv
    │   ├── generations_full_reports_METRICS.csv
    │   ├── generations_impression.csv
    │   └── generations_impression_METRICS.csv
    ├── cxr-redone
    │   ├── generations.csv
    │   ├── generations_impression.csv
    │   └── generations_impression_METRICS.csv
    ├── cxr-repair
    │   ├── generations.csv
    │   ├── generations_impression.csv
    │   └── generations_impression_METRICS.csv
    ├── rgrg
    │   ├── generations_findings.csv
    │   └── generations_findings_METRICS.csv
    └── x-rem
        ├── final_results_filtered.csv
        ├── generations_impression.csv
        ├── generations_impression_METRICS.csv
        ├── itm_results_temp.csv
        └── xrem_sids.csv
    ```