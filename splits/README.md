# Frozen split files

Committed once and never redrawn. Each file is regenerated and checked by code in this
repo, and `tests/` asserts the redraw is byte-identical to the committed file.

## `chexpertplus_pilot_split.csv`

400 CheXpert Plus patients for the pilot (docs/revision-plan.md §2, "Leakage controls"):
200 `pilot` + 200 `prompt_tuning`. Columns: `subject_id`, `study_id`, `set`.

- **Source:** `chexpertplus/df_chexpert_plus_240401.csv` (author release) only.
- **Eligible:** official-train patients with exactly one study whose Findings section
  has text (17,972 of 26,634). See `has_text` in `src/labrag/data/chexpertplus.py`.
- **Draw:** `numpy.random.default_rng(20261109).choice(sorted(eligible), 400, replace=False)`;
  the first 200 drawn are `pilot`.
- **Code:** `src/labrag/data/chexpertplus.py::draw_pilot_split`.
  `labrag splits pilot` redraws and checks against this file (it never overwrites
  without `--overwrite`).
- **History:** drawn 2026-11-09 by an earlier script that read locally derived files
  (`chexpertplus/split.csv`, `derived/chexpertplus-findings-ids.csv`). The current
  code reads only the source CSV and reproduces the same file byte for byte.
