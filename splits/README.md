# Frozen split files

Committed once and never redrawn. `labrag splits --dataset <name>` rewrites a dataset's
files in place from the code in this repo (git diff shows any change), and
`tests/test_splits.py` asserts the redraw is byte-identical to the committed files.

## `chexpertplus_pilot_split.csv`

400 CheXpert Plus patients for the pilot (docs/revision-plan.md §2, "Leakage controls"):
200 `pilot` + 200 `prompt_tuning`. Columns: `subject_id`, `study_id`, `set`.

- **Source:** `chexpertplus/df_chexpert_plus_240401.csv` (author release) only.
- **Eligible:** official-train patients with exactly one study whose Findings section
  has text (17,972 of 26,634). See `has_text` in `src/labrag/data/chexpertplus.py`.
- **Draw:** `numpy.random.default_rng(20261109).choice(sorted(eligible), 400, replace=False)`;
  the first 200 drawn are `pilot`.
- **Code:** `src/labrag/data/chexpertplus.py::draw_pilot_split`.
- **History:** drawn 2026-11-09 by an earlier script that read locally derived files
  (`chexpertplus/split.csv`, `derived/chexpertplus-findings-ids.csv`). The current
  code reads only the source CSV and reproduces the same file byte for byte.

## `chexpertplus_classifier_val_split.csv`

6,412 CheXpert Plus patients whose studies set the CheXpert Plus classifier thresholds
(the classifiers are fit on the other train patients). Column: `subject_id`.

- **Why:** the official valid split has only 61 studies with Findings, with one
  positive or none for some labels, too few to set per-label thresholds.
- **Source:** `chexpertplus/df_chexpert_plus_240401.csv` and the pilot split above.
- **Eligible:** official-train patients not in the pilot split (64,125). The draw
  ignores report sections, so the same patients are the validation set for every
  section's classifiers (4,530 of their studies have Findings).
- **Draw:** `numpy.random.default_rng(20261111).choice(sorted(eligible), 6412, replace=False)`
  (10%, rounded), written sorted.
- **Code:** `src/labrag/data/chexpertplus.py::draw_classifier_val_split`.

## `padchest_classifier_val_split.csv`

6,590 PadChest patients whose studies set the PadChest classifier thresholds (the
classifiers are fit on the other patients). Column: `subject_id`.

- **Why:** PadChest has no official validation split, and PadChest-GR validation (455
  studies) leaves 22 of the 60 PadChest-60 labels with fewer than 10 positives.
- **Source:** `padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz` and
  `padchest-gr/master_table.csv` (author releases).
- **Eligible:** patients of the studies the adapter keeps (one usable frontal image),
  minus every patient in PadChest-GR validation or test (65,904). PadChest-GR train
  patients are eligible.
- **Draw:** `numpy.random.default_rng(20261117).choice(sorted(eligible), 6590, replace=False)`
  (10%, rounded), written sorted.
- **Code:** `src/labrag/data/padchest.py::draw_classifier_val_split`.
