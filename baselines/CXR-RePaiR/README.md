# CXR-RePaiR: Retrieval-Based Chest X-Ray Report Generation Using a Pre-trained Contrastive Language-Image Model

CXR-RePaiR (Contrastive X-ray-Report Pair Retrieval) is a retrieval-based radiology report generation approach that uses a contrastive language-image model. See our paper [here](https://proceedings.mlr.press/v158/endo21a/endo21a.pdf)!

![CXR-RePaiR](_assets/cxr-repair.png)

## Running CXR-RePaiR

### Installation
#### Using conda

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies. On a CUDA GPU machine, the following will do the trick:

```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm pandas h5py sklearn
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

### Data Preprocessing
In order to run our method, we must run a series of steps to process the MIMIC-CXR-JPG dataset.

#### Data Access

First, you must get approval for the use of [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). With approval, you will have access to the train/test reports and the jpg images.

#### Create Data Split
```
python data_preprocessing/split_mimic.py \
  --report_files_dir=<directory containing all reports> \
  --split_path=<path to split file in mimic-cxr-jpg> \
  --out_dir=mimic_data
```

#### Extract Impressions Section
```
python data_preprocessing/extract_impressions.py \
  --dir=mimic_data
```

#### Create Test Set of Report/CXR Pairs
```
python data_preprocessing/create_bootstrapped_testset.py \
  --dir=mimic_data \
  --bootstrap_dir=bootstrap_test \
  --cxr_files_dir=<mimic-cxr-jpg directory containing chest X-rays>
```

#### Get groundtruth labels for test reports
Either retrieve chexpert embeddings of the mimic test reports provided in the mimic-cxr-2.0.0-chexpert.csv.gz file, or run CheXbert on the reports.csv file to get labels. Title the file labels.csv, and put the file under the bootstrap_test directory.


### Pre-trained CLIP Model
The CLIP model checkpoint trained on MIMIC-CXR train set is available for download [here](https://stanfordmedicine.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml).

### Generating embeddings for the corpus
```
python gen_corpus_embeddings.py \
  --clip_model_path=<name of clip model state dictionary for generating embeddings> \
  --clip_pretrained \
  --data_path=<path of csv file containing training corpus (either sentence level or report level)> \
  --out=clip_pretrained_mimic_train_sentence_embeddings.pt
```

*Note: if you are using a clip model that was not first pre-trained on  natural language-image pairs, then you shouldn't set the `--clip_pretrained` flag.*

### Creating reports
```
python run_test.py \
  --corpus_embeddings_name=clip_pretrained_mimic_train_sentence_embeddings.pt \
  --clip_model_path=<name of clip model state dictionary> \
  --clip_pretrained \
  --out_dir=CXR-RePaiR-2_mimic_results \
  --test_cxr_path=bootstrap_test/cxr.h5 \
  --topk=2
```

###  Generating labels of predicted reports
In order to generate per-pathology predictions from the outputted reports, use [CheXbert](https://github.com/stanfordmlgroup/CheXbert).

### Testing performance
```
python test_acc_batch.py \
 --dir=CXR-RePaiR-2_mimic_results \
 --bootstrap_dir=bootstrap_test/
```

## License

This repository is made publicly available under the MIT License.


## Citing

If you are using this repo, please cite this paper:

```
@InProceedings{pmlr-v158-endo21a,
  title = 	 {Retrieval-Based Chest X-Ray Report Generation Using a Pre-trained Contrastive Language-Image Model},
  author =       {Endo, Mark and Krishnan, Rayan and Krishna, Viswesh and Ng, Andrew Y. and Rajpurkar, Pranav},
  booktitle = 	 {Proceedings of Machine Learning for Health},
  pages = 	 {209--219},
  year = 	 {2021},
  volume = 	 {158},
  series = 	 {Proceedings of Machine Learning Research}
}
```
