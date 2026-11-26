# LaB-RAG: Results on Test (methods + results draft, 2026-11-26)

This is a copy of [preliminary-val-results.md](preliminary-val-results.md) with every
number replaced by the **official test split** results. The val draft stays unchanged as
the development record. Sections 2 (methods) and 3 (results) are drafted as paper
prose. Section 4 lists what I noticed while writing them, for you to decide.

Sources (read from disk 2026-11-26):
- MIMIC-CXR: `comparisons/test/mimic/findings+impression/qwen3.5-27b.core-no-shuffled/test`
  (all cells except shuffled labels; that run failed and its rerun is deferred).
- ReXGradient: `comparisons/test/rexgradient/findings+impression/qwen3.5-27b.core/test`.
- Experiment: `experiments/core-test.sh` (commit a9b72a2), after
  `experiments/rebuild-labels.sh`.

---

## 1. Overview

LaB-RAG generates a chest X-ray report with a frozen, general-purpose vision-language
model. The model gets the study's frontal image, five example reports retrieved from a
training corpus, the indication and the patient's prior report. LaB-RAG adds labels to
this retrieval-augmented setup in two places:

1. **Retrieval.** Corpus studies are ranked by how well their reference labels overlap
   the target's predicted labels, not by image similarity alone.
2. **Prompt.** The target's predicted labels are shown, and each example is shown with
   its reference labels.

The labels come from one small image classifier: RAD-DINO fine-tuned with a per-label
attention head, ≈ 90M parameters against the generator's 27B. The baseline,
retrieval-only (standard RAG), gets everything else unchanged: the same image,
indication, prior report, corpus, instructions and generator, with examples ranked by
image similarity and no labels.

## 2. Methods

### 2.1 Data

**Datasets.** MIMIC-CXR (v2.0.0 JPG metadata and official split, free-text reports) and
ReXGradient-160K (official patient-disjoint train/valid/test split) are the evaluation
datasets. CheXpert Plus is used only to train the classifier and is never an evaluation
or retrieval dataset. We read only the files released by the dataset authors. Sections,
labels and splits are derived in-repo. MIMIC-CXR reader IDs (2.1.0 provider list) are
used only for the same-reader cells (§3.7).

**Studies and images.** One frontal image per study, picked by view preference (PA,
then AP; ties broken by path). Studies without a labelled frontal view are dropped.
The same image is used for retrieval, the classifier and the generator prompt. Lateral
views are not used.

**Report sections.** MIMIC-CXR sections come from the authors' own section parser
(MIT-LCP `mimic-cxr`, unmodified), keeping the last section of each name. ReXGradient
gives Findings and Impression as fields. The generation target is
**Findings+Impression**: the study must have both sections, and the reference text is
the two sections joined by a space.

**Indication.** MIMIC-CXR: the parser's *history* then *indication* sections, joined.
ReXGradient: the release's Indication field. Shown as "Not provided" if missing (test
targets: MIMIC 2.0%, ReXGradient 0.1%).

**Prior report.** The prior is the patient's latest study strictly before the target.
It must have a frontal image and at least one report section. Studies are ordered by
StudyDate+StudyTime on MIMIC and by StudyDate only on ReXGradient; ties among earlier
studies are broken by study ID. The prior is shown as "Findings: … Impression: …" with
the sections it has, or "None" if there is no prior. Coverage of test targets: **87.1%
MIMIC** (val: 60.7%), 27.8% ReXGradient (val: 26.2%).

**Roles.**

| | MIMIC-CXR | ReXGradient | CheXpert Plus |
|---|---|---|---|
| Classifier training (train) | 204,016 | 108,127 | 167,749 |
| Classifier epoch selection (val) | 1,663 | 7,745 | 18,707 (drawn 10% patient split) |
| Retrieval corpus (train, both sections) | 121,273 | 108,127 | – |
| Generation targets (test, both sections) | 1,549 of 2,759 | 7,755 of 7,755 | – |

No val generation feeds any test number: the classifier's epoch is chosen on val
classifier metrics only, and the corpus is train only. Retrieval never returns a study
of the target's patient. The splits are patient-disjoint, and an explicit mask enforces
this as well.

### 2.2 Reference labels

Each study gets 40 labels in three groups. Each label takes one of four values:
positive, negative, uncertain or unmentioned.

- **CheXbert-14** (14 labels, including No Finding and Support Devices): the authors'
  CheXbert checkpoint and code, run on each section separately. It reproduces
  upstream's output.
- **Device types** (15 labels) and **incidental findings** (11 labels): rule labelers
  with a shared clause, negation and hedge scope, described in
  [labeler-rules-draft.md](labeler-rules-draft.md). Device types: PICC, pulmonary artery
  catheter, chest port, central venous catheter, endotracheal tube, tracheostomy tube,
  enteric tube, chest tube, cardiac valve device, pacemaker/defibrillator, circulatory
  support device, sternotomy wires, surgical clips, stent, monitoring leads. Incidental
  findings: degenerative spine changes, scoliosis/kyphosis, aortic calcification,
  tortuous or dilated aorta, hiatal hernia, emphysema/hyperinflation, calcified
  granuloma, low lung volumes, elevated hemidiaphragm, chronic interstitial
  changes/scarring, osteopenia.

The test labels use the rules as fixed after val (line-wrap fix, negation scope, the
pattern-bug fixes from the labeler audit, and "chronic" required for interstitial and
reticular wording). Each rule-label artifact records a hash of its rule sources, so
labels from older rules are refused. On a held-out sample of 100 random reports rated
blind by an LLM, the rules score devices P 1.000 / R .980 and incidentals P .957 /
R .917 (measured before the interstitial/reticular change; §4 note 6).

**Combining sections.** A study's Findings+Impression value for a label is the first of
positive, uncertain, negative, unmentioned that either section gives. **Consistency:**
No Finding is cleared wherever a pathology label is positive. Support Devices, the
device types and the incidental findings are not pathologies, so they do not clear No
Finding.

For retrieval and display, a label counts as present only if it is positive.
Uncertain counts as absent (U-Zeros).

### 2.3 Label classifier

**Architecture.** RAD-DINO ViT-B/14 (public `microsoft/rad-dino`, pinned revision).
Images are min-max scaled to 8-bit, resized so the short side is 518 px, center-cropped
to 518² and normalized. That gives 1 CLS + 37×37 patch tokens.

A per-label attention head reads the tokens:

- One learned query per label (40) cross-attends over all 1,370 tokens with 8 heads.
- A residual feed-forward block follows (LayerNorm, FFN 3072 with GELU, dropout 0.1).
- The queries do not attend to each other. Labels share the projection and FFN
  weights, but each label pools its own image regions.
- Each label's pooled feature has two readouts (Findings, Impression) × four classes
  (positive, negative, uncertain, unmentioned).

**Training.** One model is trained on the train roles of all three datasets pooled, with
all 40 labels. The label-set ladder (§3.2) uses this same model and only changes which
labels are shown and ranked on; there is no second fine-tune.

- **Loss:** softmax cross-entropy over the four classes for each (label, section)
  entry, masked to the sections the study has.
- **LP-FT:** 2 epochs head only (lr 1e-4), then 10 epochs end to end (head lr 1e-4,
  backbone lr 2e-5, layer-wise decay 0.75, linear warmup over 5% of steps, cosine to 0).
- **Optimizer:** AdamW, weight decay 0.05, none on backbone biases, norms or
  embeddings.
- **Batch and precision:** global batch 128 over 4 GPUs (DDP), bfloat16 autocast.
- **Augmentation:** random resized crop (scale 0.5–1, aspect 3/4–4/3), rotation ±10°.
  No flips.
- **Epoch selection:** best mean over datasets of each dataset's mean val AUROC for the
  positive class (one vs rest). Selected epoch: 9 of 12. Mean val AUROC 0.920
  (CheXpert Plus 0.904, MIMIC 0.922, ReXGradient 0.933), over 40 labels × 2 readouts.
  Epochs 7–12 are within 0.001.

**Prediction.** For each label and readout, the predicted value is the most probable
class (argmax), so there are no thresholds. No Finding is cleared wherever a pathology
is predicted positive. The two readouts are combined with the reference rule (§2.2).
The target's predicted label set is its positives.

### 2.4 Retrieval

The corpus is the dataset's own train studies that have both sections. Image similarity
is the cosine between **frozen** RAD-DINO CLS embeddings (the public model, not the
fine-tuned classifier).

- **Retrieval-only:** top 5 by cosine similarity.
- **LaB-RAG:** ranked first by the Jaccard overlap between the target's predicted
  positive label set and each corpus study's reference positive label set, over the
  labels in use. No Finding counts as a label, and two empty sets have Jaccard 1. Ties
  are broken by cosine similarity, then by corpus order. Top 5 are kept.
- **Same-reader** (MIMIC only): the corpus is restricted to train studies read by the
  target's radiologist; ranking otherwise unchanged.

The target's patient is always excluded.

### 2.5 Prompt

There is one user message, with versions named in `configs/prompts.yaml` and never
edited after use.

- **Role and task:** "You are an expert radiological assistant…"
- **Instruction 1:** base the report on the image, indication, prior report, context
  and (label conditions) the current label.
- **Instruction 2:** mimic the style of the examples (conditions with examples).
- **Instruction 3:** do not generate blank reports.
- **Instruction 4:** use the prior report for the patient's earlier findings and
  devices. Compare only for a specific finding or device, and only where the current
  image supports it. No general "unchanged/stable" statement. If the prior is "None",
  make no reference to prior exams. No dates, times or names.
- **Instruction 5:** write "Findings:" then "Impression:", each with its header.
- **Instruction 6:** state pertinent negatives.
- **Instruction 7:** use the indication to judge pertinence, but do not report a
  finding just because the indication names it.
- **CONTEXT** (conditions with examples): five examples, best first. Each is
  `Example: i`, then (labels shown) `Label: <reference positives>`, then
  "Findings: … Impression: …".
- **CURRENT CASE:** indication, prior report, image.
- **Closing line (labels shown):** "…using its label below…", then
  `Label: <predicted positives>`, comma-separated in a fixed label order, or "Other" if
  there are none.

Conditions without examples (no-context, labels-only) drop the CONTEXT block and the
instructions about examples; everything else is word for word the same.

### 2.6 Generation

Qwen3.5-27B, served with its vision encoder by vLLM 0.32.0 (image pinned by digest,
weights checked by sha256).

- **Image input:** min-max scaled to 8-bit, short side 518 px (bicubic, no crop), sent
  inline as a PNG.
- **Thinking:** on. Model card's general settings: temperature 1.0, top-p 0.95, top-k
  20, min-p 0, presence penalty 1.5, cap 32,768 tokens. Only the final content (after
  vLLM's qwen3 reasoning parser) is scored.
- **Seeds:** every generated cell is run three times, with sampling seeds 20261112,
  20261115 and 20261119.

### 2.7 Evaluation

**Scoring the output.** A strict parser needs exactly one "Findings:" at the start and
one "Impression:" after it, both non-empty; markdown bold is allowed. The two parts are
joined and scored as one text against the joined reference, with no headers. Outputs
that fail the parser are still scored with headers removed (≤ 0.02% of outputs in any
condition). There were no empty outputs.

**Metrics.** Each metric runs its official code in its own container with its
upstream's pins: CXR-Report-Metric (BLEU-2, BERTScore, SembScore, RadGraph F1,
RadCliQ-v1; original Python 3.7 environment, CPU), RaTEScore, GREEN (GREEN-RadLlama2-7b,
served by vLLM in batch-invariant, deterministic-compile mode) and CRIMSON (upstream
`crimson-score` with its MedGemma-4B judge, greedy, served by vLLM with prefix caching
and chunked prefill off; `clean_report_text`, no guidelines, no patient context). We
report 1/RadCliQ-v1 = 1 / mean RadCliQ-v1.

**CRIMSON failures.** Upstream leaves unscorable studies out of the mean, and so do we.
Rates on test: 0.0–0.7% per generated condition. Each condition's mean is over the
studies it scored; a difference is over the studies both scored.

**Statistics.** Each study's score is averaged over the three sampling runs. Conditions
are compared on the same studies with a paired bootstrap over studies (10,000
resamples, fixed seed, 95% percentile interval). No correction for multiple
comparisons. The bootstrap covers which studies are drawn, not generation sampling.

### 2.8 Conditions

| Short name | Examples ranked by | Labels shown | Label source |
|---|---|---|---|
| Template | – (no LLM) | rendered as sentences | classifier |
| Template-oracle | – (no LLM) | rendered as sentences | reference |
| No-context | – (no examples) | – | – |
| RO (retrieval-only) | image cosine | – | – |
| Filter-only | label Jaccard, then cosine | – | classifier |
| Labels-only | – (no examples) | target only | classifier |
| L-C | label Jaccard, then cosine | target + examples, CheXbert-14 only | classifier |
| L-CD | label Jaccard, then cosine | target + examples, CheXbert-14 + device types | classifier |
| **LaB-RAG (L-CDI)** | **label Jaccard, then cosine** | **target + examples, all 40 labels** | **classifier** |
| Shuffled (ReXG only) | as LaB-RAG | as LaB-RAG | another patient's predicted labels |
| Oracle | as LaB-RAG | as LaB-RAG | the target's reference labels (ceiling, never a method) |
| RO-SR, LaB-RAG-SR (MIMIC only) | as RO / LaB-RAG | as RO / LaB-RAG | same-reader corpus |

All generated conditions share the image, indication, prior report, instructions,
corpus, generator and sampling. L-C, L-CD and LaB-RAG use the same classifier and the
same predictions; only the label subset changes. Filter-only and labels-only use
LaB-RAG's labels. **Shuffled** gives each target the predicted labels of a different
target from another patient (a fixed random derangement, seed 20261126), used for both
ranking and display. **Template** writes one Findings sentence per shown label
("Pleural effusion.", or "No acute cardiopulmonary abnormality." for No Finding or no
label) and an Impression of the shown CheXbert pathologies; it is the floor an LLM has
to beat.

---

## 3. Results

All results are on the test split, for Findings+Impression, with Qwen3.5-27B given the
image. Each study's scores are averaged over three sampling seeds. Brackets give 95%
paired-bootstrap confidence intervals; * = interval excludes 0.

**Table 1. Mean scores, MIMIC-CXR test (n = 1,549).** For GREEN false and GREEN missing
(clinically significant errors per study), lower is better. CRIMSON is in (−1, 1];
higher is better.

| | BLEU-2 | BERTScore | SembScore | RadGraph F1 | 1/RadCliQ-v1 | RaTEScore | GREEN | GREEN false | GREEN missing | CRIMSON |
|---|---|---|---|---|---|---|---|---|---|---|
| Template | .009 | .154 | .332 | .039 | .551 | .445 | .180 | .503 | 4.979 | .260 |
| Template-oracle | .014 | .183 | .489 | .055 | .628 | .514 | .244 | .278 | 4.435 | .519 |
| No-context | .217 | .366 | .429 | .233 | .885 | .615 | .383 | 1.864 | 2.002 | .133 |
| RO | .232 | .394 | .447 | .245 | .954 | .619 | .397 | 1.738 | 2.058 | .158 |
| Filter-only | .236 | .401 | .461 | .252 | .985 | .623 | .406 | 1.652 | 2.039 | .202 |
| Labels-only | .224 | .377 | .473 | .238 | .943 | .625 | .385 | 1.853 | 1.955 | .171 |
| L-C | .230 | .402 | .477 | .245 | .993 | .622 | .397 | 1.702 | 2.076 | .175 |
| L-CD | .233 | .402 | .474 | .249 | .995 | .623 | .399 | 1.682 | 2.061 | .190 |
| **LaB-RAG** | .233 | .402 | .474 | .250 | .997 | .623 | **.405** | **1.592** | 2.071 | **.208** |
| Oracle | .259 | .426 | .591 | .285 | 1.242 | .661 | .459 | 1.525 | 1.639 | .420 |
| RO-SR | .257 | .418 | .455 | .270 | 1.031 | .630 | .414 | 1.684 | 2.008 | .172 |
| LaB-RAG-SR | .257 | .425 | .489 | .275 | 1.088 | .637 | .413 | 1.564 | 2.034 | .205 |

**Table 2. Mean scores, ReXGradient test (n = 7,755).**

| | BLEU-2 | BERTScore | SembScore | RadGraph F1 | 1/RadCliQ-v1 | RaTEScore | GREEN | GREEN false | GREEN missing | CRIMSON |
|---|---|---|---|---|---|---|---|---|---|---|
| Template | .023 | .271 | .406 | .052 | .652 | .396 | .134 | .224 | 4.473 | .465 |
| Template-oracle | .031 | .288 | .506 | .063 | .718 | .452 | .189 | .202 | 4.160 | .649 |
| No-context | .201 | .391 | .454 | .199 | .905 | .605 | .555 | 1.225 | 1.372 | .244 |
| RO | .257 | .447 | .511 | .253 | 1.124 | .623 | .543 | 1.167 | 1.484 | .312 |
| Filter-only | .266 | .456 | .528 | .261 | 1.181 | .629 | .549 | 1.077 | 1.473 | .362 |
| Labels-only | .213 | .408 | .498 | .208 | .978 | .617 | .566 | 1.134 | 1.319 | .320 |
| L-C | .263 | .454 | .534 | .259 | 1.181 | .629 | .547 | 1.100 | 1.485 | .351 |
| L-CD | .265 | .456 | .536 | .262 | 1.195 | .630 | .553 | 1.070 | 1.455 | .371 |
| **LaB-RAG** | .266 | .458 | .538 | .263 | 1.203 | .632 | **.553** | **1.036** | 1.455 | **.384** |
| Shuffled | .204 | .389 | .398 | .195 | .855 | .568 | .437 | 1.553 | 1.822 | .006 |
| Oracle | .289 | .482 | .625 | .293 | 1.497 | .668 | .594 | 1.048 | 1.131 | .562 |

**Table 3. Paired differences.** GREEN false / missing and CRIMSON false / missing are
errors per study (negative = fewer errors).

| Contrast | Dataset | GREEN | CRIMSON | RaTEScore | RadGraph F1 | GREEN false | GREEN missing | CRIMSON false | CRIMSON missing |
|---|---|---|---|---|---|---|---|---|---|
| LaB-RAG − RO (main result) | MIMIC | +.009 [+.003, +.015] | +.051 [+.033, +.068] | +.004 [+.001, +.007] | +.005 [+.001, +.008] | −.145 [−.192, −.099] | +.013 [−.031, +.058] | −.286 [−.362, −.208] | −.123 [−.191, −.060] |
| | ReXG | +.010 [+.007, +.013] | +.071 [+.063, +.080] | +.009 [+.008, +.011] | +.011 [+.009, +.013] | −.131 [−.150, −.113] | −.029 [−.046, −.013] | −.271 [−.299, −.244] | −.102 [−.122, −.083] |
| L-C − RO (CheXbert-14) | MIMIC | +.001 [−.005, +.006] | +.017 [+.000, +.033] | +.003 [+.000, +.006] | −.000 [−.004, +.003] | −.036 [−.079, +.008] | +.018 [−.025, +.062] | −.130 [−.201, −.058] | −.077 [−.141, −.016] |
| | ReXG | +.004 [+.001, +.007] | +.039 [+.031, +.047] | +.006 [+.005, +.007] | +.006 [+.004, +.008] | −.067 [−.084, −.050] | +.001 [−.013, +.016] | −.125 [−.152, −.100] | −.054 [−.071, −.036] |
| L-CD − L-C (+ device types) | MIMIC | +.002 [−.003, +.006] | +.016 [+.006, +.026] | +.000 [−.002, +.002] | +.004 [+.002, +.007] | −.020 [−.056, +.013] | −.015 [−.051, +.020] | −.070 [−.116, −.023] | −.026 [−.064, +.011] |
| | ReXG | +.005 [+.003, +.007] | +.019 [+.014, +.024] | +.001 [+.000, +.002] | +.003 [+.002, +.005] | −.030 [−.043, −.016] | −.030 [−.042, −.018] | −.062 [−.081, −.043] | −.046 [−.058, −.034] |
| LaB-RAG − L-CD (+ incidentals) | MIMIC | +.006 [+.002, +.011] | +.018 [+.007, +.029] | +.001 [−.001, +.003] | +.001 [−.001, +.004] | −.090 [−.125, −.053] | +.011 [−.025, +.046] | −.086 [−.136, −.035] | −.020 [−.060, +.019] |
| | ReXG | +.001 [−.002, +.003] | +.013 [+.008, +.018] | +.002 [+.001, +.003] | +.001 [−.000, +.002] | −.035 [−.049, −.020] | −.001 [−.013, +.012] | −.083 [−.103, −.064] | −.002 [−.015, +.010] |
| LaB-RAG − filter-only (labels shown) | MIMIC | −.001 [−.006, +.004] | +.007 [−.008, +.021] | −.000 [−.002, +.002] | −.002 [−.004, +.000] | −.060 [−.098, −.021] | +.033 [−.007, +.072] | −.025 [−.086, +.036] | −.066 [−.112, −.019] |
| | ReXG | +.004 [+.002, +.006] | +.021 [+.016, +.027] | +.003 [+.002, +.004] | +.002 [+.001, +.003] | −.041 [−.056, −.027] | −.018 [−.031, −.006] | −.062 [−.080, −.044] | −.046 [−.061, −.031] |
| LaB-RAG − labels-only (examples) | MIMIC | +.020 [+.015, +.025] | +.037 [+.025, +.049] | −.001 [−.003, +.001] | +.012 [+.009, +.015] | −.261 [−.298, −.224] | +.116 [+.080, +.151] | −.351 [−.404, −.298] | +.144 [+.109, +.181] |
| | ReXG | −.013 [−.016, −.010] | +.064 [+.057, +.071] | +.015 [+.013, +.017] | +.055 [+.052, +.058] | −.099 [−.116, −.082] | +.136 [+.121, +.151] | −.233 [−.253, −.212] | +.039 [+.025, +.052] |
| RO − no-context (examples, no labels) | MIMIC | +.013 [+.009, +.018] | +.025 [+.013, +.037] | +.004 [+.002, +.006] | +.013 [+.010, +.016] | −.126 [−.165, −.087] | +.057 [+.019, +.094] | −.136 [−.197, −.079] | +.013 [−.038, +.074] |
| | ReXG | −.012 [−.015, −.009] | +.068 [+.061, +.075] | +.018 [+.017, +.020] | +.053 [+.050, +.056] | −.058 [−.076, −.040] | +.111 [+.096, +.127] | −.209 [−.235, −.184] | −.010 [−.025, +.005] |
| Shuffled − LaB-RAG | ReXG | −.116 [−.121, −.111] | −.378 [−.391, −.364] | −.065 [−.067, −.062] | −.068 [−.072, −.065] | +.518 [+.489, +.547] | +.368 [+.345, +.391] | +1.049 [+.994, +1.102] | +.689 [+.655, +.724] |
| Oracle − LaB-RAG (remaining gap) | MIMIC | +.053 [+.047, +.060] | +.211 [+.191, +.233] | +.038 [+.035, +.041] | +.035 [+.031, +.039] | −.068 [−.116, −.019] | −.432 [−.477, −.387] | −.464 [−.553, −.376] | −.939 [−1.005, −.872] |
| | ReXG | +.040 [+.037, +.044] | +.179 [+.169, +.188] | +.035 [+.034, +.037] | +.029 [+.027, +.032] | +.012 [−.007, +.032] | −.324 [−.342, −.305] | −.159 [−.186, −.132] | −.712 [−.739, −.684] |
| Template − LaB-RAG | MIMIC | −.225 [−.235, −.216] | +.052 [+.036, +.067] | −.178 [−.185, −.172] | −.211 [−.216, −.207] | −1.089 | +2.908 | −.707 | +.531 |
| | ReXG | −.419 [−.426, −.413] | +.082 [+.074, +.090] | −.237 [−.240, −.233] | −.211 [−.215, −.207] | −.811 | +3.018 | −.458 | +.244 |

### 3.1 LaB-RAG vs retrieval-only

LaB-RAG improves on retrieval-only on both datasets for GREEN, CRIMSON, RaTEScore,
RadGraph F1, BERTScore, SembScore and 1/RadCliQ-v1, and every one of these intervals
excludes 0 (Tables 1–3). BLEU-2 rises on ReXGradient (+.009*) and is unchanged on
MIMIC-CXR (+.000 [−.003, +.003]).

- GREEN +.009 on MIMIC-CXR and +.010 on ReXGradient.
- CRIMSON +.051 / +.071.
- RaTEScore +.004 / +.009.
- RadGraph F1 +.005 / +.011.
- 1/RadCliQ-v1 +.044 / +.079.

The gain holds in each sampling seed (difference of run means). GREEN: +.007, +.007,
+.012 on MIMIC-CXR; +.009, +.010, +.011 on ReXGradient. CRIMSON: +.051, +.060, +.043 on
MIMIC-CXR; +.075, +.070, +.068 on ReXGradient.

The gain comes mainly from fewer false findings. GREEN counts −.15 and −.13 clinically
significant false findings per study (8% and 11% of retrieval-only's rate); missed
findings are unchanged on MIMIC-CXR and slightly lower on ReXGradient (−.03). CRIMSON
counts −.29 and −.27 false findings per study (14% and 23%) and fewer missed findings
(−.12 / −.10, 6% / 7%).

The baseline is not handicapped. Retrieval-only gets the same image, indication, prior
report, corpus and instructions.

### 3.2 The label ladder

One classifier predicts all 40 labels; the ladder changes only which of its labels are
shown and ranked on.

- **CheXbert-14 only (L-C).** GREEN unchanged on MIMIC-CXR (+.001) and +.004 on
  ReXGradient. CRIMSON +.017 / +.039.
- **+ Device types (L-CD − L-C).** GREEN +.002 (n.s.) / +.005. CRIMSON +.016 / +.019.
- **+ Incidental findings (LaB-RAG − L-CD).** GREEN +.006 / +.001 (n.s.). CRIMSON
  +.018 / +.013.

Every step raises CRIMSON on both datasets. On GREEN the extended label groups carry
the gain, with the dataset deciding which one: incidental findings on MIMIC-CXR, device
types on ReXGradient. Over the whole ladder, LaB-RAG − L-C is GREEN +.008* / +.006* and
CRIMSON +.034* / +.032*: with CheXbert-14 alone, LaB-RAG would not beat retrieval-only
on GREEN on MIMIC-CXR. Each added label group mainly removes false findings (GREEN
−.09 for incidentals on MIMIC-CXR; CRIMSON −.06 to −.09 per step).

### 3.3 Where the labels act: retrieval vs prompt

**Retrieval (filter-only − RO).** Ranking examples by predicted labels, without showing
any label, already gives most of the gain: GREEN +.009 / +.006, CRIMSON +.044 / +.050.

**Showing the labels on top (LaB-RAG − filter-only).** On ReXGradient it adds to every
metric: GREEN +.004, CRIMSON +.021, fewer false and missed findings. On MIMIC-CXR GREEN
and CRIMSON are unchanged (−.001 / +.007, n.s.), although GREEN false findings fall
(−.06*) and CRIMSON missed findings fall (−.07*).

**Labels without examples (labels-only).** The examples matter. Against LaB-RAG,
labels-only makes many more false findings (GREEN +.26 / +.10, CRIMSON +.35 / +.23 per
study) but misses fewer (GREEN −.12 / −.14), and loses on CRIMSON (−.037 / −.064), BLEU,
BERTScore and RadGraph on both datasets. GREEN disagrees in direction across datasets:
LaB-RAG − labels-only is +.020* on MIMIC-CXR and −.013* on ReXGradient, where
labels-only has the highest GREEN of any non-oracle condition (.566).

**Examples without labels (RO − no-context).** Retrieval helps on MIMIC-CXR for every
metric (GREEN +.013). On ReXGradient it raises CRIMSON (+.068), RaTEScore and all text
metrics but lowers GREEN (−.012*): with examples, the generator misses more findings by
GREEN's count (+.11 per study).

### 3.4 The generator follows the labels (ReXGradient)

With another patient's predicted labels (shuffled), every metric falls far below
retrieval-only: GREEN .437 vs .543, CRIMSON .006 vs .312. Against LaB-RAG, false
findings rise by +.52 (GREEN) / +1.05 (CRIMSON) per study and missed findings by +.37 /
+.69. The labels steer what the model writes, so the gains above come from label
content, not from the format of a labelled prompt. (MIMIC-CXR shuffled is pending; §4.)

### 3.5 Headroom

With the target's reference labels (oracle), GREEN reaches .459 on MIMIC-CXR and .594 on
ReXGradient, and CRIMSON .420 and .562. LaB-RAG closes 14% / 20% of the gap from
retrieval-only to the oracle on GREEN and 19% / 29% on CRIMSON.

The remaining gap is mostly missed findings: oracle − LaB-RAG is −.43 / −.32 GREEN
missed findings per study, against false findings −.07 / +.01. CRIMSON puts it in the
same place (−.94 / −.71 missed) and also gives the oracle fewer false findings (−.46 /
−.16). This matches the earlier finding that the gap to the oracle comes from the
classifier's false negatives.

### 3.6 The template floor and what CRIMSON rewards

Rendering the classifier's labels as sentences, with no LLM, scores far below every
generated condition on GREEN (.180 / .134), RaTEScore and the text metrics, as
expected. On CRIMSON it scores **higher** than every generated non-oracle condition:
.260 vs LaB-RAG's .208 on MIMIC-CXR, .465 vs .384 on ReXGradient (both CIs exclude 0),
and the template-oracle (.519 / .649) beats the generated oracle (.420 / .562). GREEN
counts the template's missed findings at ≈ 3 more per study than LaB-RAG; CRIMSON
counts +.53 / +.24.

CRIMSON scores abnormal findings and gives expected-benign findings weight 0, so a
short list of the right abnormal labels does well on it even when it says nothing else.
CRIMSON gains from labels should therefore be read alongside GREEN and RaTEScore, not
on their own.

### 3.7 Same-reader retrieval (MIMIC-CXR, current configuration)

**Table 4. Same-reader retrieval, MIMIC-CXR test (n = 1,549), paired differences.**

| Contrast | GREEN | CRIMSON | RaTEScore | RadGraph F1 |
|---|---|---|---|---|
| RO-SR − RO | +.017 [+.012, +.022] | +.014 [+.003, +.026] | +.011 [+.009, +.014] | +.025 [+.021, +.029] |
| LaB-RAG-SR − LaB-RAG | +.008 [+.003, +.012] | −.004 [−.014, +.007] | +.013 [+.011, +.016] | +.025 [+.021, +.029] |
| LaB-RAG-SR − RO-SR | −.001 [−.007, +.005] | +.033 [+.013, +.051] | +.006 [+.003, +.009] | +.005 [+.001, +.009] |
| LaB-RAG − RO (all readers) | +.009 [+.003, +.015] | +.051 [+.033, +.068] | +.004 [+.001, +.007] | +.005 [+.001, +.008] |

The val pattern mostly holds in the current configuration: same-reader examples raise
GREEN, RaTEScore and RadGraph in both conditions, but CRIMSON much less (RO, +.014) or
not at all (LaB-RAG). Labels raise CRIMSON with or without same-reader examples. With
same-reader examples, labels no longer add GREEN (−.001); the two effects overlap on
GREEN but not on CRIMSON. LaB-RAG-SR has the best non-oracle RaTEScore (.637),
RadGraph (.275) and 1/RadCliQ-v1 (1.088) on MIMIC-CXR.

---

## 4. Notes for your review (not paper text)

1. **Gains are a little smaller than on val, as expected.** LaB-RAG − RO, val → test:
   MIMIC GREEN +.014 → +.009, CRIMSON +.062 → +.051; ReXG GREEN +.011 → +.010, CRIMSON
   +.079 → +.071. Every val estimate (GREEN, CRIMSON, RaTEScore, RadCliQ-v1, both
   datasets) lies inside the test CI; signs and significance all hold. Reasons, not separable: winner's curse (the cell
   was picked as the best of many on val), MIMIC val was 960 studies × 2 seeds (±.01
   noise), and the labels/classifier were rebuilt after val.
2. **MIMIC test is a different population from val.** RO GREEN .397 (val .514), CRIMSON
   .158 (val .316); 87% of test targets have a prior vs 61% on val. Relative to the
   lower baseline, the MIMIC CRIMSON gain is larger on test (+32% vs +20%). Worth one
   sentence so readers don't read the lower absolute scores as a regression.
3. **MIMIC filter-only ≈ LaB-RAG on GREEN/CRIMSON.** Showing the labels adds nothing
   measurable there on the headline judges (it does on ReXG). The paper's story has to
   lean on "labels for retrieval" being part of the method, or on the error-count
   shifts (fewer GREEN false findings, fewer CRIMSON misses).
4. **ReXG no-context and labels-only beat RO on GREEN** (+.012*, +.023*), but not on
   MIMIC (−.013*, −.011*), and both lose on every other metric on ReXG. On ReXG GREEN
   counts more missed findings whenever examples are present. One possible reading is
   that ReXG examples are terse and the generator copies their brevity, which GREEN
   counts as omissions. Not checked.
5. **CRIMSON label-content caveat (§3.6).** The no-LLM template beats LaB-RAG on CRIMSON
   on both datasets. This weakens "labels are much larger on CRIMSON" as a headline and
   bears on the open paper-message question: §3.7 still separates style (GREEN) from
   label content (CRIMSON), but CRIMSON's insensitivity to everything except abnormal
   findings now cuts both ways.
6. **Labeler validation is slightly stale.** The held-out P/R in §2.2 predate the
   "chronic" change for interstitial/reticular wording (81c99c5), which moves that one
   incidental label (1,703 MIMIC / 769 ReXG sections were positive only through it).
   The deferred audit needs a third held-out sample anyway.
7. **MIMIC shuffled cell missing.** `donors()` (cyclic shift) can't avoid same-patient
   pairs on MIMIC test (277 patients, up to 43 studies each). You deferred the fix and
   rerun; ReXG shuffled should be rerun with the same new algorithm so both datasets
   use one. The full MIMIC compare then publishes at `qwen3.5-27b.core`.
8. **SembScore/RadCliQ overlap with the labeler** (as on val): SembScore uses CheXbert
   and RadCliQ includes it. Rest the claim on GREEN, CRIMSON, RaTEScore and RadGraph.
9. **CRIMSON failures** 0.0–0.7%. On ReXG, the oracle fails slightly more than LaB-RAG
   (+.15 pp*), and no-context and labels-only fail more than the conditions with
   examples (+.1 to +.2 pp*). Not inspected.
10. **Not in this draft:** per-section (Findings vs Impression) scores from the strict
    parser's split, classifier test-set metrics, and test run statistics (reasoning
    length, cost). All can be computed from existing artifacts.
11. **Oracle ladder not run.** Only the 40-label oracle exists on test, so §3.2 has no
    per-step ceilings (val had O-D vs O-DI).
