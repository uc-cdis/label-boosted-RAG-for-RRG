# Data sources and provenance

The pipeline reads **only files released by the dataset authors**, under `/opt/gpudata/cxr`.
Everything derived from them (sections, labels, splits) is computed in this repo. Older local
derivations live in `<dataset>/derived/` and are never read. Audit of kg35-nvl01: 2026-11-10;
the other nodes are checked against it with [tools/data-audit/](../tools/data-audit/README.md).
kg35-nvl03 and kg35-nvl04 are byte-identical to it (2026-11-10). kg35-nvl02 has no copy of the data.
The PhysioNet `SHA256SUMS.txt` files were deleted after the audit. The hashes of the verified
files remain in the kg35-nvl01 reference manifests.

## Inputs and verification

| Dataset | Files the pipeline reads | Verified against |
|---|---|---|
| MIMIC-CXR-JPG 2.1.0 | `files/` (377,110 JPGs), `mimic-cxr-2.0.0-{split,metadata,chexpert,negbio}.csv.gz`, `mimic-cxr-2.1.0-test-set-labeled.csv` | PhysioNet `SHA256SUMS.txt`: every listed file is present and matches |
| MIMIC-CXR 2.1.0 (reports) | `notes/` (227,835 reports, the unzipped `mimic-cxr-reports.zip`) | PhysioNet `SHA256SUMS.txt` of the DICOM release, which hashes each report: all match |
| MIMIC-CXR sections | computed from `notes/` with MIT-LCP's `section_parser.py` and custom rules ([external/mimic-cxr](../external/mimic-cxr)) | no author file exists |
| CheXpert Plus | `df_chexpert_plus_240401.csv`, `PNG/` (223,461 PNGs) | no author checksums. The PNG set equals the csv's image list. The extracted `*_fixed.json` and `section_*.json` equal the members of the author zips |
| ReXGradient-160K | `metadata/{train,valid,test}_metadata*.{csv,json}`, `deid_png/` (273,004 PNGs) | Hugging Face etags: all 17 metadata/README/LICENSE/example files match. The images can't be checked (the download tars are gone); their counts and paths match the metadata |
| PadChest | `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`, `images/` (160,861 PNGs) | no author checksums. The image set equals the csv |
| PadChest-GR | `grounded_reports_20240819.json`, `master_table.csv`, `PadChest_GR/` (4,555), `PadChest_GR_progression_prior_studies/` (1,446) | no author checksums. The image sets equal the JSON, and every image is byte-identical to its PadChest copy |
| BRAX | not used | author `SHA256SUMS.txt`: every file matches |
| VinDr-CXR | not used | author `SHA256SUMS.txt`: every file matches |
| OpenI (IU X-ray) | not used | NLM gives no checksums. `pngs/` and `reports/` equal the members of `NLMCXR_png.tgz` (sha256 `baf3abfe19ba5d58efe69002aed1e71aa2e6d5efb3238db9adcac210ad44bdf2`, except a `Thumbs.db`) and `NLMCXR_reports.tgz` (`8fb6de7eec73d8c3665067ad4bb003ccd57f971ae316d2642e1627ac7268667a`), which were then deleted. `NLMCXR_dcm.tgz` is kept and passes `gzip -t` |

**Archives:** a single-file `.gz` stays compressed (pandas reads it directly), and its
uncompressed copy goes to `derived/`. Multi-file archives are kept extracted: the extracted
files are the source. CheXpert Plus keeps both its zips and the extracted JSONs.

## Known issues (handled in the adapters)

- **MIMIC test-set-labeled:** the author file has 9 studies from the validate split and 2 study
  IDs that aren't in MIMIC-CXR. Filter it to the official test split.
- **CheXpert Plus:** `PNG/train/patient32368/study1/view1_frontal.png` is corrupt and was
  renamed `.png.BAK`. Skip it. Reports differ per image within a study (Stanford-AIMI/chexpert-plus#13), so take
  one canonical text per study.
- **PadChest:** 33 PNGs don't decode (none is a PadChest-GR image). Drop images that fail to
  decode.
- **PadChest-GR split:** use `master_table.csv` column `split` (study level). The
  `study_is_benchmark` and `study_is_validation` columns are constant.

## Local derivations (moved to `<dataset>/derived/`, never read)

- `derived/` (top level, from 2025-12): old section and label csvs for CheXpert Plus, MIMIC and
  OpenI. Split by name prefix into each dataset's `derived/`.
- MIMIC-CXR: `mimic_cxr_sectioned.csv`, `mimic_cxr_selected_section.csv` (old modified
  section script), `chexbert_labels.csv`, and `mimic-cxr-2.0.0-{chexpert,metadata,negbio,split}.csv`
  (uncompressed copies, identical to the author `.gz`).
- CheXpert Plus: `split.csv` (an old carve-out: "validate" is part of official train, "test"
  is official valid), `metadata.csv`, `report.csv`.
- ReXGradient: `metadata/*_chexbert_labels.csv`, `by-site/`.
- OpenI: `metadata.csv`, `report.csv`, `split.csv`, and `IU-xray-projections.csv` (the Kaggle
  re-release's projections table, not an NLM file).
