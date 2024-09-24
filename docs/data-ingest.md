# Data Ingest

## To prepare the dataset
1. Download MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
    * This will take a few days if using `wget`.
1. Cleanup files:
    1. Remove unnecessary folder depth. Should look like below at this point:
        ```
        mimic-cxr
        ├── files
        │   └── [...]
        ├── IMAGE_FILENAMES
        ├── LICENSE.txt
        ├── README
        ├── SHA256SUMS.txt
        ├── mimic-cxr-2.0.0-chexpert.csv.gz
        ├── mimic-cxr-2.0.0-metadata.csv.gz
        ├── mimic-cxr-2.0.0-negbio.csv.gz
        ├── mimic-cxr-2.0.0-split.csv.gz
        └── mimic-cxr-2.1.0-test-set-labeled.csv
        ```
    1. Unzip compressed files:
        ```bash
        gzip -d *.csv.gz
        ```
    1. The folder should look like this:
        ```
        mimic-cxr
        ├── files
        │   └── [...]
        ├── IMAGE_FILENAMES
        ├── LICENSE.txt
        ├── README
        ├── SHA256SUMS.txt
        ├── mimic-cxr-2.0.0-chexpert.csv
        ├── mimic-cxr-2.0.0-metadata.csv
        ├── mimic-cxr-2.0.0-negbio.csv
        ├── mimic-cxr-2.0.0-split.csv
        └── mimic-cxr-2.1.0-test-set-labeled.csv
        ```
1. Download the notes from MIMIC-CXR. Rather than download all of MIMIC-CXR which has all of the original `.dcm` files, we can get just the notes:
    ```bash
    wget -r -N -c -np --user <USER> --ask-password https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip
    ```
1. Move `mimic-cxr-reports.zip` to the `mimic-cxr` directory. This will result in the notes being unzipped alongside the jpgs.
    1. The folder **before** unzipping should look like this:
        ```
        mimic-cxr
        ├── files
        │   └── [...]
        ├── IMAGE_FILENAMES
        ├── LICENSE.txt
        ├── README
        ├── SHA256SUMS.txt
        ├── mimic-cxr-2.0.0-chexpert.csv
        ├── mimic-cxr-2.0.0-metadata.csv
        ├── mimic-cxr-2.0.0-negbio.csv
        ├── mimic-cxr-2.0.0-split.csv
        ├── mimic-cxr-2.1.0-test-set-labeled.csv
        └── mimic-cxr-reports.zip
        ```
    1. Unzip the reports:
        ```bash
        unzip mimic-cxr-reports.zip
        ```
    1. The folder **after** unzipping should look like this:
        ```
        mimic-cxr
        ├── files
        │   └── [...]
        ├── IMAGE_FILENAMES
        ├── LICENSE.txt
        ├── README
        ├── SHA256SUMS.txt
        ├── mimic-cxr-2.0.0-chexpert.csv
        ├── mimic-cxr-2.0.0-metadata.csv
        ├── mimic-cxr-2.0.0-negbio.csv
        ├── mimic-cxr-2.0.0-split.csv
        ├── mimic-cxr-2.1.0-test-set-labeled.csv
        ├── mimic-cxr-reports.zip
        └── notes
            └── [...]
        ```
1. Extract note sections:
    1. Activate the conda environment:
        ```bash
        conda activate rrg
        ```
    1. Run the following [1]:
        ```bash
        python /path/to/rrg-repo/data-ingest/create_section_files.py \
        --reports_path /path/to/mimic-cxr/notes \
        --output_path /path/to/mimic-cxr
        ```
    1. At the end of it all, your folder structure should look like this:
        ```
        mimic-cxr
        ├── files
        │   └── [...]
        ├── IMAGE_FILENAMES
        ├── LICENSE.txt
        ├── README
        ├── SHA256SUMS.txt
        ├── mimic-cxr-2.0.0-chexpert.csv
        ├── mimic-cxr-2.0.0-metadata.csv
        ├── mimic-cxr-2.0.0-negbio.csv
        ├── mimic-cxr-2.0.0-split.csv
        ├── mimic-cxr-2.1.0-test-set-labeled.csv
        ├── mimic-cxr-reports.zip
        ├── mimic_cxr_sectioned.csv
        ├── mimic_cxr_selected_section.csv
        └── notes
            └── [...]
        ```

## References
1. `create_section_files.py` and `section_parser.py` modified from https://doi.org/10.5281/zenodo.2591653
