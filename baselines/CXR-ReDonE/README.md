# CXR-ReDonE

![](https://i.imgur.com/6X0P1HT.jpg)

Current deep learning models trained to generate radiology reports from chest radiographs are capable of producing clinically accurate, clear, and actionable text that can advance patient care. However, such systems all succumb to the same problem: making hallucinated references to non-existent prior reports. Such hallucinations occur because these models are trained on datasets of real-world patient reports that inherently refer to priors. To this end, we propose two methods to directly remove references to priors in radiology reports: (1) a GPT-3-based few-shot approach to rewrite medical reports without references to priors; and (2) a BioBERT-based token classification approach to directly remove tokens referring to priors. We use the aforementioned approaches to modify MIMIC-CXR, a publicly available dataset of chest X-rays and their associated free-text radiology reports; we then retrain CXR-RePaiR, a radiology report generation system, on the adapted MIMIC-CXR dataset. We find that our re-trained model--which we call CXR-ReDonE--outperforms previous report generation methods on clinical metrics, and expect it to be broadly valuable in enabling current radiology report generation systems to be more directly integrated into clinical pipelines.

## Setup

CXR-ReDonE makes use of multiple GitHub repos. To set up the complete CXR-ReDonE directory, clone the below repos, then perform the following commands inside `CXR-ReDonE/`:

- [ifcc](https://github.com/ysmiura/ifcc)

```bash
cd ifcc
sh resources/download.sh
cd ..
mv ifcc_CXR_ReDonE_code/* ifcc/
rm -rf ifcc_CXR_ReDonE_code
```

- [CheXbert](https://github.com/stanfordmlgroup/CheXbert)

```bash
cd CheXbert
mkdir models
cd models
```

> Once inside the `models` directory, download the [pretrained weights](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) for CheXbert.

- [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR)

- [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)

```bash
mv CXR-Report-Metric_CXR_ReDonE_code/* CXR-Report-Metric/
rm -rf CXR-Report-Metric_CXR_ReDonE_code
```

- [ALBEF](https://github.com/salesforce/ALBEF)

> As we made multiple edits to the ALBEF directory, please refer to the ALBEF directory uploaded here instead of cloning a new one. Make sure to download [ALBEF_4M.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth) (`wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth`) and place it in `ALBEF/`.

## Data Preprocessing

Here, we make use of CXR-RePaiR's data preprocessing steps:

> #### Environment Setup

```bash
cd CXR-RePaiR
conda env create -f cxr-repair-env.yml
conda activate cxr-repair-env
```

> #### Data Access

> First, you must get approval for the use of [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). With approval, you will have access to the train/test reports and the JPG images.

> #### Create Data Split

```bash
python data_preprocessing/split_mimic.py \
  --report_files_dir=<directory containing all reports> \
  --split_path=<path to split file in mimic-cxr-jpg> \
  --out_dir=mimic_data
```

> #### Extract Impressions Section

```bash
python data_preprocessing/extract_impressions.py \
  --dir=mimic_data
```

> #### Create Test Set of Report/CXR Pairs

```bash
python data_preprocessing/create_bootstrapped_testset_copy.py \
  --dir=mimic_data \
  --bootstrap_dir=bootstrap_test \
  --cxr_files_dir=<mimic-cxr-jpg directory containing chest X-rays>
```

<br>

The above commands produce the following files: `cxr.h5`, `mimic_train_impressions.csv`, and `mimic_test_impressions.csv`. The expert-annotated test set must still be download from [here](https://drive.google.com/file/d/1kSY-GLpuDLwTcuosEb8GpEVz5CaTjqVV/view?usp=sharing).

To remove references to priors from the above data files, move the above three files to a new folder `CXR-ReDonE/data`, then run:

```bash
python remove_prior_refs.py
```

<br>

To skip these steps and instead obtain the final data, navigate to the `CXR-PRO` directory and follow the steps there.

## Training

To pretrain ALBEF, run:

```bash
cd ALBEF
sh pretrain_script.sh
```

Linked here are the ALBEF model checkpoints [with](https://www.dropbox.com/s/b4tkf2z4v6wa4zj/checkpoint_59.pth?dl=0) and [without](https://drive.google.com/file/d/183TClsB_fzCOHa6ESWfefV6EoN0EfmMI/view?usp=sharing) removing references to priors from the MIMIC-CXR reports corpus.

## Inference

To run inference, run the following:

```bash
cd ALBEF
python CXR_ReDonE_pipeline.py --albef_retrieval_ckpt <path-to-checkpoint>
```

For sentence-level report retrieval (where k = # of sentences in outputted report), run:

```bash
cd ALBEF
python CXR_ReDonE_pipeline.py --impressions_path ../data/mimic_train_sentence_impressions.csv --albef_retrieval_ckpt <path-to-checkpoint> --albef_retrieval_top_k <k value>
```

## Evaluation

For evaluating the generated reports, we make use of CXR-Report-Metric:

### Setup

```bash
cd ../CXR-Report-Metric
conda create -n "cxr-report-metric" python=3.7.0 ipython
conda activate cxr-report-metric
pip install -r requirements.txt
```

Next, download the RadGraph model checkpoint from PhysioNet [here](https://physionet.org/content/radgraph/1.0.0/). The checkpoint file can be found under the "Files" section at path `models/model_checkpoint/`. Set `RADGRAPH_PATH` in `config.py` to the path to the downloaded checkpoint.

### Evaluation

1. Use `prepare_df.py` to select the inferences for the corresponding 2,192 samples from our generation.

```
python prepare_df.py --fpath <input path> --opath <output path>
```

2. In `config.py`, set `GT_REPORTS` to `../data/cxr2_generated_appa.csv` and `PREDICTED_REPORTS` to `<output path>`. Set `OUT_FILE` to the desired path for the output metric scores. Set `CHEXBERT_PATH` to the path to the downloaded checkpoint (`CheXbert/models/chexbert.pth`).

3. Use `test_metric.py` to generate the scores.

```
python test_metric.py
```

4. Finally, use `compute_avg_score.py` to output the average scores.

```
python3 compute_avg_score.py --fpath <input path>
```

<br>

## Download pre-trained BioBERT models

The above steps to run CXR-ReDonE rely on two pretrained models:

### FilBERT: <u>Fil</u>tering Sentence-Level References to Priors with Bio<u>BERT</u>

The model can be called programmatically as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_name="rajpurkarlab/filbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def run_bert_classifier(sentence, tokenizer, model):
    pipe = pipeline("sentiment-analysis", model=model.to("cpu"), tokenizer=tokenizer)
    return int(pipe(sentence)[0]['label'][-1])

tokenizer, model = load_model()
run_bert_classifier("SINGLE SENTENCE FROM A REPORT", tokenizer, model)
```

### GILBERT: <u>G</u>enerating <u>I</u>n-text <u>L</u>abels of References to Priors with Bio<u>BERT</u>

The model can be called programmatically as follows:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

def get_pipe():
    model_name = "rajpurkarlab/gilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline(task="token-classification", model=model.to("cpu"), tokenizer=tokenizer, aggregation_strategy="simple")
    return pipe

def remove_priors(pipe, report):
    ret = ""
    for sentence in report.split("."):
        if sentence and not sentence.isspace():
            p = pipe(sentence)
            string = ""
            for item in p:
                if item['entity_group'] == 'KEEP':
                    string += item['word'] + " "
            ret += string.strip().replace("redemonstrate", "demonstrate").capitalize() + ". "
    return ret.strip()

modified_report = remove_priors(get_pipe(), "YOUR REPORT")
```
