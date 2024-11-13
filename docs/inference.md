# Inference and Replication of Inference Results

## Setup
We will use the same data setup as the experiments section. We present a comparison our results to 6 other models on our data split and steps to replicate these results. The other methods are as follows:

1. CXR-RePaiR (https://github.com/rajpurkarlab/CXR-RePaiR)
2. CXR-ReDonE (https://github.com/rajpurkarlab/CXR-ReDonE)
3. X-REM (https://github.com/rajpurkarlab/X-REM)
4. CheXagent (https://github.com/Stanford-AIMI/CheXagent)
5. CXRMate-RRG24 (https://huggingface.co/aehrc/cxrmate-rrg24)
6. RGRG (https://github.com/ttanida/rgrg)

### Steps
1. Run the `create dataset` section of `inference-results.ipynb` to generate the dataset files `inference_findings_data.csv` and `inference_impression_data.csv`. These will serve as the split files for our splitwise experiments
1. Change into the inference directory 
    ```bash
    cd /inference
    ```
1. The following are the detailed steps to replicate generations for the 6 comparison models on our data split:
    ### 1. CXR-RePaiR

    * Create and activate the conda environment. 
        ```bash
        cd CXR-RePaiR-rrg
        conda env create -f environment.yml
        conda activate cxr-repair_env
        ```
    We follow the same steps as in CXR-RePaiR
    1. `split_mimic.py` creates the data split.
        ```bash
        python ./data_preprocessing/split_mimic.py \
        --report_files_dir path/to/mimic-cxr/notes/ \
        --split_path path/to/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz \
        --out_dir /path/to/save directory
        ```
    1. `extract_impressions.py` extracts the impression sections
        ```bash
        python ./data_preprocessing/extract_impressions.py \
        --dir /path/to/save directory
        ```
    1. `create_bootstrapped_testset.py` creates the test split dataset of reports and cxr pairs
        ```bash
        python data_preprocessing/create_bootstrapped_testset.py \
        --dir  /path/to/save directory \
        --bootstrap_dir /path/to/save directory/bootstrapped_test_set/ \
        --cxr_files_dir path/to/mimic-cxr/files/
        ```
    The CLIP model checkpoint trained on MIMIC-CXR train set is available for download [here](https://stanfordmedicine.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml).
    1. `gen_corpus_embeddings.py` generates the embeddings for the corpus
        ```bash
        python gen_corpus_embeddings.py \
        --clip_model_path path/to/clip-imp-pretrained_128_6_after_4.pt \
        --clip_pretrained \
        --data_path path/to/mimic_train_impressions.csv \
        --out path/to/corpus_embeddings/clip_pretrained_mimic_train_sentence_embeddings.pt
        ```
    1. `run_test.py` generates the final reports
        ```bash
        python run_test.py \
        --corpus_embeddings_name path/to/corpus_embeddings/clip_pretrained_mimic_train_sentence_embeddings.pt \
        --clip_model_path  path/to/clip-imp-pretrained_128_6_after_4.pt \
        --clip_pretrained \
        --out_dir /path/to/save directory \
        --test_cxr_path path/to/bootstrapped_test_set/cxr.h5 \
        --topk 2
        ```


    ### 2. X-REM-rrg

    1. Change into the X-REM-rrg directory and create the environment:
        ```bash
        cd X-REM-rrg
        conda env create -f environment.yml -n X-REM_env
        conda activate X-REM_env 
        ```

    2. Move into the X-REM directory 
        ```bash
        cd X-REM
        ``` 

    3. `inference.py` generates the impressions on the cxr-repair test split of the data.
        ```bash
        python inference.py \
        --img_path path/to/cxr.h5 \
        --rep_path path/to/mimic_train_impressions.csv \
        --albef_path path/to/albef_checkpoint
        --save_dir path/to/save directory
        ```

    4. Download the required model and Post-process the results using a NLI filter:
        ```bash
        cd ../ifcc
        cd /resources
        ./download.sh
        cd ..
        python m2trans_nli_filter.py \
        --m2trans_nli_model_path path/to/X-REM-anirudh repo/ifcc/resources/model_medrad_19k \
        --input_path path/to/itm_results_temp.csv
        --save_path path/to/final_results_filtered.csv
        ```


    ### 3. CXR-ReDonE
    * Switch to the CXR-ReDonE directory and activate the environment
        ```bash
        cd cxr-redone
        conda activate X-REM_env
        ```
    Download the ALBEF retrieval checkpoint without removing references from the link provided in the repo docs. Also download the ALBEF ve model and place it in the ALBEF directory. 
    1. `CXR_ReDonE_pipeline.py` generates the reports over the test split as defined by the data pre-processing steps of CXR-RePaiR
        ```bash
        python CXR_ReDonE_pipeline.py \
        --impression_path /path/to/mimic_train_impressions.csv \
        --img_path /path/to/cxr.h5 \
        --save_path /path/to/save directory \
        --albef_retrieval_ckpt /path/to/checkpoint_59_w_priors.pth \
        --albef_ve_ckpt /path/to/checkpoint_7.pth
        ```
    2. Run the `cxr-redone` section of `inference-results.ipynb` to create the appropriate generations file by split    


    ### 4. CXRMate-RRG24 

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