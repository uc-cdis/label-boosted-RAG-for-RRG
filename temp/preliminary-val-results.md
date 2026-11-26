# LaB-RAG: Preliminary Results on Val (methods + results draft, updated 2026-11-25)

Every number here is from the **val split**, which was also the development set. The
test split has not been run. Test will use 3 sampling seeds per main cell and labels
rebuilt with the fixed rule labelers ([labeler-rules-draft.md](labeler-rules-draft.md)).

Describes the configuration that beat retrieval-only on GREEN on both datasets
(`experiments/incidental-readouts-val.sh`, commit 3c9d7b4; cell
`mc-incidentals.positives.negatives.indication.prior-changes.devices.incidentals`
vs `retrieval-only.negatives.indication.prior-changes`). CRIMSON scores were added on
2026-11-25 (`experiments/crimson-val.sh`). Sections 2 (methods) and 3 (results) are
drafted as paper prose. Section 4 lists what I noticed while writing them, for you to
decide.

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

**Datasets.** MIMIC-CXR (v2.0.0 JPG metadata and official split, free-text reports;
reader IDs from the 2.1.0 provider list are not used here) and ReXGradient-160K (official
patient-disjoint train/valid/test split, 67 sites) are the evaluation datasets. CheXpert
Plus is used only to train the classifier and is never an evaluation or retrieval
dataset. We read only the files released by the dataset authors. Sections, labels and
splits are derived in-repo.

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
ReXGradient: the release's Indication field. Shown as "Not provided" if missing (MIMIC:
2% of targets).

**Prior report.** The prior is the patient's latest study strictly before the target.
It must have a frontal image and at least one report section. Studies are ordered by
StudyDate+StudyTime on MIMIC and by StudyDate only on ReXGradient; ties among earlier
studies are broken by study ID. The prior is shown as "Findings: … Impression: …" with
the sections it has, or "None" if there is no prior. Coverage of val targets: 60.7%
MIMIC, 26.2% ReXGradient.

**Roles.**

| | MIMIC-CXR | ReXGradient | CheXpert Plus |
|---|---|---|---|
| Classifier training (train) | 204,016 | 108,127 | 167,749 |
| Classifier epoch selection (val) | 1,663 | 7,745 | 18,707 (drawn 10% patient split) |
| Retrieval corpus (train, both sections) | 121,273 | 108,127 | – |
| Generation targets (val, both sections) | 960 | 7,745 | – |

Test splits (MIMIC 2,759 / ReXGradient 7,755 studies) have not been touched.
Retrieval never returns a study of the target's patient. The splits are
patient-disjoint, and an explicit mask enforces this as well.

### 2.2 Reference labels

Each study gets 40 labels in three groups. Each label takes one of four values:
positive, negative, uncertain or unmentioned.

- **CheXbert-14** (14 labels, including No Finding and Support Devices): the authors'
  CheXbert checkpoint and code, run on each section separately. It reproduces
  upstream's output.
- **Device types** (15 labels): a rule labeler. Reports are split into clauses. Each
  device pattern is matched in a fixed order, most specific first, and every match is
  cut out before the next pattern runs, so "PICC" is not also counted as a central
  line. A device is positive if any clause names it without removal or negation
  language, negative if every clause that names it says it is removed or absent, and
  unmentioned otherwise. Types: PICC, pulmonary artery catheter, chest port, central
  venous catheter, endotracheal tube, tracheostomy tube, enteric tube, chest tube,
  cardiac valve device, pacemaker/defibrillator, circulatory support device, sternotomy
  wires, surgical clips, stent, monitoring leads.
- **Incidental findings** (11 labels): a rule labeler with the same clause logic, using
  negation cues ("no", "without", "resolved", "negative for", "free of"). Categories:
  degenerative spine changes, scoliosis/kyphosis, aortic calcification, tortuous or
  dilated aorta, hiatal hernia, emphysema/hyperinflation, calcified granuloma, low lung
  volumes, elevated hemidiaphragm, chronic interstitial changes/scarring, osteopenia.

**Combining sections.** A study's Findings+Impression value for a label is the first of
positive, uncertain, negative, unmentioned that either section gives. Running CheXbert
on the joined text instead drops positives that both sections state, and then agrees
with this rule on only ≈ 81% of studies. **Consistency:** No Finding is cleared
wherever a pathology label is positive. Support Devices, the device types and the
incidental findings are not pathologies, so they do not clear No Finding.

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
  (positive, negative, uncertain, unmentioned). The sections share where to look and
  differ only in the last linear layer.

**Training.** One model is trained on the train roles of all three datasets pooled.

- **Loss:** softmax cross-entropy over the four classes for each (label, section)
  entry. Each entry is masked to the sections the study actually has.
- **LP-FT:** 2 epochs training the head only (lr 1e-4), then 10 epochs end to end. In
  the end-to-end phase the head lr is 1e-4 and the backbone lr is 2e-5 with layer-wise
  decay of 0.75, linear warmup over 5% of steps, then cosine decay to 0.
- **Optimizer:** AdamW, weight decay 0.05, none on backbone biases, norms or
  embeddings.
- **Batch and precision:** global batch 128 over 4 GPUs (DDP), bfloat16 autocast.
- **Augmentation:** random resized crop (scale 0.5–1, aspect ratio 3/4–4/3) and
  rotation of ±10°. No flips, because laterality matters.
- **Determinism:** deterministic kernels, and augmentation drawn per (seed, epoch,
  row).
- **Epoch selection:** the kept epoch has the best mean over datasets of each dataset's
  mean val AUROC for the positive class (one vs rest). Selected epoch: 10 of 12. Mean
  val AUROC 0.915 (CheXpert Plus 0.898, MIMIC 0.915, ReXGradient 0.931), over the 40
  labels × 2 readouts.

**Prediction.** For each label and readout, the predicted value is the most probable
class (argmax), so there are no thresholds. No Finding is cleared wherever a pathology
is predicted positive. The two readouts are then combined with the same rule as the
reference labels (§2.2), so predicted, example and oracle labels share one definition.
The target's predicted label set is its positives.

### 2.4 Retrieval

The corpus is the dataset's own train studies that have both sections. Image similarity
is the cosine between **frozen** RAD-DINO CLS embeddings (the public model, not the
fine-tuned classifier).

- **Retrieval-only:** top 5 by cosine similarity.
- **LaB-RAG:** ranked first by the Jaccard overlap between the target's predicted
  positive label set and each corpus study's reference positive label set, over all 40
  labels. No Finding counts as a label, and two empty sets have Jaccard 1. Ties are
  broken by cosine similarity, then by corpus order. Top 5 are kept.

The target's patient is always excluded. In practice the label ranking almost always
finds exact label matches: the top example has Jaccard 1 for 96.4% of MIMIC and 98.4% of
ReXGradient targets, and the mean Jaccard of the five examples is 0.986 / 0.993. Within
a matched label set, the order is by image similarity.

### 2.5 Prompt

There is one user message, with versions named in `configs/prompts.yaml` and never
edited after use. The two conditions share the same instructions, with two differences:
LaB-RAG's text mentions labels, and its examples and target carry label lines.

- **Role and task:** "You are an expert radiological assistant…"
- **Instruction 1:** base the report on the image, indication, prior report, context
  and (LaB-RAG) the current label.
- **Instruction 2:** mimic the style of the examples.
- **Instruction 3:** do not generate blank reports.
- **Instruction 4:** use the prior report for the patient's earlier findings and
  devices. Compare only for a specific finding or device, and only where the current
  image supports it. No general "unchanged/stable" statement. If the prior is "None",
  make no reference to prior exams. No dates, times or names.
- **Instruction 5:** write "Findings:" then "Impression:", each with its header.
- **Instruction 6:** state pertinent negatives, as the examples do.
- **Instruction 7:** use the indication to judge pertinence, but do not report a
  finding just because the indication names it.
- **CONTEXT:** five examples, best first. Each is `Example: i`, then (LaB-RAG)
  `Label: <reference positives>`, then "Findings: … Impression: …". Whitespace is
  collapsed and the text is otherwise unedited.
- **CURRENT CASE:** indication, prior report, image.
- **Closing line (LaB-RAG):** "…using its label below…", then
  `Label: <predicted positives>`, comma-separated in a fixed label order, or "Other" if
  there are none.

Only positive labels are shown. The classifier's negative, uncertain and unmentioned
predictions are used only to decide which labels are positive. "Negatives" in the cell
name refers to instruction 6, not to a list of absent labels.

### 2.6 Generation

Qwen3.5-27B, served with its vision encoder by vLLM 0.32.0 (image pinned by digest,
weights checked by sha256).

- **Image input:** the frontal image is min-max scaled to 8-bit, resized so the short
  side is 518 px (bicubic, no crop) and sent inline as a PNG. On the CheXpert Plus pilot,
  native resolution scored the same at ≈ 13× the image tokens.
- **Thinking:** on. Sampling follows the model card's general settings: temperature
  1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, cap 32,768 tokens.
- **Scored text:** the reasoning block is split off by vLLM's qwen3 parser, and only the
  final content is scored.
- **Seeds:** each cell is generated twice, with sampling seeds 20261112 and 20261115.
- **Run statistics** (MIMIC, LaB-RAG): no truncation. Reasoning averages 2.3k tokens
  (p95 4.2k). Prompts average 1.3k tokens. Cost is ≈ 1 H200-second per report.

### 2.7 Evaluation

**Scoring the output.** A strict parser needs exactly one "Findings:" at the start and
one "Impression:" after it, with both sections non-empty; markdown bold is allowed. The
two parts are joined and scored as one text against the joined reference, with no
headers. Outputs that fail the parser are still scored, with all headers removed, and
counted per condition (0% in this cell). Empty outputs would be scored as "EMPTY"
(there were none).

**Metrics.** Each metric runs its official code in its own container with its
upstream's pins:

- **CXR-Report-Metric:** BLEU-2, BERTScore, SembScore, RadGraph F1, RadCliQ-v1. Original
  Python 3.7 / torch 1.6 environment, on CPU.
- **RaTEScore.**
- **GREEN:** GREEN-RadLlama2-7b judge, served by vLLM in batch-invariant and
  deterministic-compile mode. Set means are within 0.0012 of upstream's HF generate.
- **CRIMSON:** the upstream `crimson-score` code with its MedGemma-4B judge, run
  greedy. The judge is served by vLLM with prefix caching and chunked prefill off,
  which makes its output identical across batches and jobs. Reports go through
  upstream's `clean_report_text`. Guidelines and patient context are not passed. On
  MIMIC val retrieval-only, the served set mean is within +.005 [−.001, +.010] of
  upstream's HF generate. Per study, the two differ by .02 on average.

We report 1/RadCliQ-v1 = 1 / mean RadCliQ-v1. Input conventions that differ from
ReXrank are listed in the revision plan §7.

**CRIMSON failures.** When CRIMSON's judge output cannot be scored, upstream leaves
the study out of the mean, and so do we. This happened for 0.1–0.6% of studies per
condition, mostly judge repetition loops that hit the 8,192-token cap on ordinary
reports. Each condition's mean is over the studies it scored. A difference is over
the studies both conditions scored. The failure rate is reported per condition.

**Statistics.** Each study's score is averaged over the two sampling runs. Conditions
are compared on the same studies with a paired bootstrap over studies (10,000
resamples, fixed seed, 95% percentile interval). There is no correction for multiple
comparisons. The bootstrap covers which studies are drawn, not variation in generation
sampling.

### 2.8 Conditions

| Condition | Examples ranked by | Labels in prompt | Label source |
|---|---|---|---|
| Retrieval-only | image cosine | – | – |
| LaB-RAG | label Jaccard, then cosine | target (predicted) + examples (reference) | classifier |
| LaB-RAG oracle | label Jaccard, then cosine | target (reference) + examples (reference) | the target's own report: the ceiling for any classifier, never a method |

All conditions share the image, indication, prior report, instructions 1–7, corpus,
generator and sampling.

---

## 3. Results

All results are on the val split, for Findings+Impression, with Qwen3.5-27B given the
image. Each study's scores are averaged over two sampling seeds. Brackets give 95%
paired-bootstrap confidence intervals. The test split has not been run.

**Conditions.** Every condition has the base prompt from §2.5 (image, indication, prior
report, instructions 1–7).

| Short name | Condition |
|---|---|
| RO | retrieval-only |
| L-D | LaB-RAG; classifier trained on CheXbert-14 + device types; those labels shown and ranked on |
| L-D′ | LaB-RAG; classifier trained on CheXbert-14 + device types + incidental findings; only CheXbert-14 + device types shown and ranked on (the control for retraining) |
| **L-DI** | **LaB-RAG; the same classifier as L-D′; all 40 labels shown and ranked on (the full method)** |
| O-D, O-DI | oracle: the target's reference labels, for the L-D and L-DI label sets |

**Table 1. Mean scores, MIMIC-CXR val (n = 960).** For GREEN false and GREEN missing
(clinically significant errors per study), lower is better. CRIMSON is in (−1, 1];
higher is better.

| | BLEU-2 | BERTScore | SembScore | RadGraph F1 | 1/RadCliQ-v1 | RaTEScore | GREEN | GREEN false | GREEN missing | CRIMSON |
|---|---|---|---|---|---|---|---|---|---|---|
| RO | .261 | .451 | .541 | .281 | 1.222 | .652 | .514 | 1.270 | 1.629 | .316 |
| L-D | .268 | .460 | .565 | .286 | 1.294 | .656 | .514 | 1.239 | 1.618 | .346 |
| L-D′ | .269 | .461 | .568 | .289 | 1.309 | .656 | .523 | 1.177 | 1.609 | .359 |
| **L-DI** | .269 | .463 | .565 | .290 | 1.309 | .657 | **.528** | **1.096** | 1.614 | **.378** |
| O-D | .281 | .470 | .655 | .305 | 1.558 | .673 | .543 | 1.171 | 1.434 | .465 |
| O-DI | .296 | .486 | .663 | .322 | 1.701 | .690 | .569 | 1.140 | 1.206 | .534 |

**Table 2. Mean scores, ReXGradient val (n = 7,745).**

| | BLEU-2 | BERTScore | SembScore | RadGraph F1 | 1/RadCliQ-v1 | RaTEScore | GREEN | GREEN false | GREEN missing | CRIMSON |
|---|---|---|---|---|---|---|---|---|---|---|
| RO | .265 | .454 | .516 | .263 | 1.164 | .628 | .553 | 1.138 | 1.445 | .320 |
| L-D | .272 | .462 | .540 | .272 | 1.235 | .635 | .562 | 1.042 | 1.410 | .378 |
| L-D′ | .272 | .464 | .539 | .274 | 1.240 | .636 | .561 | 1.029 | 1.429 | .383 |
| **L-DI** | .275 | .466 | .542 | .275 | 1.252 | .637 | **.564** | **.980** | 1.434 | **.399** |
| O-D | .288 | .478 | .616 | .289 | 1.452 | .654 | .584 | 1.032 | 1.248 | .492 |
| O-DI | .298 | .491 | .630 | .304 | 1.576 | .673 | .600 | 1.050 | 1.102 | .580 |

**Table 3. Paired differences.**

| Contrast | Dataset | GREEN | CRIMSON | RaTEScore | RadGraph F1 | GREEN false | GREEN missing |
|---|---|---|---|---|---|---|---|
| L-DI − RO (main result) | MIMIC | +.014 [+.005, +.023] | +.062 [+.037, +.086] | +.006 [+.002, +.010] | +.009 [+.003, +.014] | −.174 [−.232, −.116] | −.015 [−.072, +.042] |
| | ReXG | +.011 [+.008, +.014] | +.079 [+.070, +.088] | +.009 [+.008, +.011] | +.012 [+.010, +.014] | −.158 [−.178, −.138] | −.011 [−.028, +.007] |
| L-D − RO (device labels) | MIMIC | −.000 [−.009, +.009] | +.030 [+.006, +.054] | +.005 [+.001, +.009] | +.005 [+.000, +.011] | −.031 [−.089, +.028] | −.010 [−.067, +.045] |
| | ReXG | +.008 [+.005, +.012] | +.058 [+.049, +.067] | +.007 [+.006, +.009] | +.009 [+.007, +.012] | −.096 [−.115, −.076] | −.034 [−.051, −.017] |
| L-D′ − L-D (retraining only) | MIMIC | +.010 [+.003, +.016] | +.013 [−.003, +.028] | +.000 [−.003, +.003] | +.003 [−.001, +.006] | −.062 [−.110, −.016] | −.009 [−.055, +.037] |
| | ReXG | −.000 [−.003, +.002] | +.006 [−.001, +.012] | +.001 [+.000, +.002] | +.001 [−.000, +.003] | −.013 [−.029, +.003] | +.019 [+.005, +.032] |
| L-DI − L-D′ (incidentals shown) | MIMIC | +.005 [−.003, +.012] | +.019 [+.005, +.033] | +.001 [−.002, +.004] | +.001 [−.003, +.005] | −.080 [−.127, −.034] | +.005 [−.042, +.053] |
| | ReXG | +.003 [+.000, +.006] | +.016 [+.009, +.022] | +.001 [−.000, +.002] | +.001 [−.000, +.003] | −.049 [−.065, −.033] | +.005 [−.009, +.019] |
| O-DI − L-DI (remaining gap) | MIMIC | +.041 [+.031, +.051] | +.156 [+.129, +.184] | +.033 [+.028, +.038] | +.032 [+.026, +.038] | +.043 [−.018, +.104] | −.408 [−.472, −.346] |
| | ReXG | +.036 [+.032, +.039] | +.181 [+.171, +.191] | +.037 [+.035, +.038] | +.029 [+.027, +.032] | +.070 [+.050, +.090] | −.332 [−.352, −.312] |

### 3.1 LaB-RAG vs retrieval-only

LaB-RAG improves on retrieval-only on all eight metrics on both datasets, and every
confidence interval excludes 0 (Tables 1–3):

- GREEN +.014 on MIMIC-CXR and +.011 on ReXGradient.
- CRIMSON +.062 / +.079.
- RaTEScore +.006 / +.009.
- RadGraph F1 +.009 / +.012.

The gain holds in each sampling seed separately. For GREEN it is +.018 and +.011 on
MIMIC-CXR, +.008 and +.014 on ReXGradient. For CRIMSON it is +.067 and +.054 on
MIMIC-CXR, +.079 and +.080 on ReXGradient.

The gain comes mainly from fewer false findings. GREEN counts −.17 and −.16 clinically
significant false findings per study, about 14% of retrieval-only's rate, and no change
in missed findings. CRIMSON counts −.28 false findings per study on both datasets
(21% and 26% of retrieval-only's rate) and also a small drop in missed findings (−.09
on both, 6%).

The baseline is not handicapped. Retrieval-only gets the same image, indication, prior
report, corpus and instructions, and it gained from each of them in earlier experiments.

### 3.2 What each label group contributes

**Device types alone (L-D).** L-D improves on retrieval-only on all text metrics on both
datasets. GREEN rises on ReXGradient (+.008) but not on MIMIC-CXR (+.000). CRIMSON rises
on both (+.030 / +.058).

**Incidental findings.** Adding them raises GREEN by a further +.014 on MIMIC-CXR and
+.003 on ReXGradient (L-DI − L-D), and CRIMSON by +.032 / +.021. We split this into two
parts:

- **Retraining the classifier** with incidental labels, without showing them
  (L-D′ − L-D): +.010 on MIMIC-CXR, which is within the ±.01 GREEN spread we measured
  between runs of identical prompts there, and .000 on ReXGradient. CRIMSON: +.013 /
  +.006, neither significant. We do not claim a retraining effect.
- **Showing and ranking on the predicted incidental findings** (L-DI − L-D′): GREEN
  +.005 (not significant) on MIMIC-CXR and +.003 on ReXGradient. CRIMSON +.019 / +.016,
  both significant. On both datasets, both seeds with incidental findings shown score
  higher on CRIMSON than both seeds without them. On both metrics the gain comes from
  fewer false findings: GREEN −.08 / −.05 per study, CRIMSON −.14 / −.08.

**How the incidental labels act.** These figures are GREEN only, from a per-study analysis whose
error categories were assigned by keyword rules and not audited. The labels work
through two routes that partly cancel:

- **Studies with at least one predicted incidental finding** (17% of MIMIC-CXR, 11% of
  ReXGradient): the generator writes every shown incidental finding. Missed findings
  fall (−.24 / −.20 per study) and GREEN rises (+.019 / +.010; CIs touch 0).
- **All other studies:** label ranking now prefers examples without incidental
  findings. Incidental mentions across the five examples fall from 2.06 to 0.14
  (MIMIC-CXR) and from 1.65 to 0.01 (ReXGradient). The generator then copies fewer of
  them: false findings −.08 / −.06, but missed findings +.06 / +.03, and GREEN is about
  unchanged.

### 3.3 Headroom

With reference labels of the same 40 types (O-DI), GREEN reaches .569 on MIMIC-CXR and
.600 on ReXGradient, and CRIMSON .534 and .580. LaB-RAG therefore closes about a quarter
of the gap from retrieval-only to the oracle: 26% / 24% on GREEN, 28% / 30% on CRIMSON.

The remaining gap is almost all missed findings: −.41 and −.33 per study for O-DI
against L-DI. False findings are about the same on MIMIC-CXR (+.04, not significant)
and lower for LaB-RAG than for the oracle on ReXGradient (+.07 for the oracle). CRIMSON
puts the gap in the same place: −.74 and −.72 missed findings per study. Unlike GREEN, it
also gives the oracle fewer false findings (−.15 / −.10). This matches
our earlier analysis that the gap to the oracle comes from the classifier's false
negatives, not from how the generator uses the labels. The incidental findings
themselves have a larger ceiling than their predicted gain: O-DI − O-D is +.026 on
MIMIC-CXR and +.016 on ReXGradient (CRIMSON +.069 / +.087).

### 3.4 GREEN and CRIMSON disagree on what helps

No label contrast above is significant in opposite directions on the two judges, but
CRIMSON's gains from labels are 2–7× GREEN's (Table 3). CRIMSON scores only abnormal
findings, and expected-benign findings have weight 0. Its authors designed it this way
so that radiologists' different styles of describing normal anatomy do not count. One
reading is that labels name findings, so a metric that ignores style shows their effect
more fully.

Retrieval restricted to the target's radiologist shows the opposite pattern. Here only
the corpus changes: examples come only from train studies read by the target's
radiologist (MIMIC-CXR's provider list; ranking otherwise unchanged). This experiment
predates the configuration above: the prompt has the indication but no prior report,
the labels are CheXbert-14 only, and it ran on MIMIC-CXR only.

**Table 4. Same-reader retrieval, MIMIC-CXR val (n = 960), paired differences.**

| Contrast | GREEN | CRIMSON |
|---|---|---|
| RO, same reader − RO | +.016 [+.008, +.024] | +.002 [−.016, +.020] |
| LaB-RAG, same reader − LaB-RAG | +.024 [+.016, +.032] | −.003 [−.016, +.011] |
| Oracle, same reader − oracle | +.020 [+.011, +.029] | +.013 [−.005, +.031] |
| LaB-RAG − RO (all readers) | −.001 [−.010, +.008] | +.035 [+.011, +.060] |

Same-reader examples raise GREEN in all three conditions but leave CRIMSON unchanged. In
this older configuration, labels leave GREEN unchanged and raise CRIMSON. Matching the
reader's style is something GREEN rewards and CRIMSON does not. In the current
configuration (Table 3), label gains show on both judges, and more strongly on CRIMSON.

---

## 4. Notes for your review (not paper text)

### Things to decide or check

1. **Val is both the development set and the set these numbers come from.** Every
   component was adopted from val results, cell after cell: the multiclass classifier,
   Findings+Impression, indication, prior-changes wording, device labels, incidental
   labels. This is also the best of many LaB-RAG cells, so expect some winner's curse.
   **Decided 2026-11-25:** stop evaluating on val; run the test split next.
2. **The classifier's epoch was selected on the same val studies we generate for.**
   The effect is probably negligible, since epochs 9–12 differ by < .0005 AUROC and
   there are no thresholds to tune. It should still be stated, or the selection moved
   to a held-out part of train.
3. **Generation noise vs the effect size.** Identical prompts differ by about ±.01 GREEN
   on MIMIC across runs. MIMIC's +.014 is only slightly above that, although pooling 2
   seeds helps. ReXGradient (+.011, tight CI, 8× the studies) is the stronger evidence.
   **Decided 2026-11-25:** 3 sampling seeds per main cell on test. CRIMSON is noisier
   between seeds: about ±.01 on both datasets, where GREEN on ReXGradient is ±.003. Its
   label effects are still several times that.
4. **Labels predate the labeler fixes.** These val results used the device and
   incidental rules from before ebc8e11 and the line-wrap fix 8a1f7fb. Test runs use
   the rebuilt labels, so test numbers come from somewhat different labels. Since then
   the rules have been checked against a blind LLM rater on a held-out sample
   ([labeler-rules-draft.md](labeler-rules-draft.md) §7): devices P 1.000 / R .980,
   incidentals P .957 / R .917. That draft also says §2.2 here could shrink to a
   summary that points to it.
5. **Two image representations.** Retrieval uses frozen RAD-DINO CLS while the
   classifier uses the fine-tuned model. This is deliberate (retrieval-only gets the
   same embedding), but it should be said explicitly so no one reads it as a mismatch.
6. **No Finding can appear next to device and incidental labels**, e.g.
   "No Finding, Support Devices, Low Lung Volumes". The rules allow it on purpose,
   because reports do this. The paper needs one sentence on it.
7. **SembScore and RadCliQ overlap with the labeler.** SembScore uses CheXbert, the
   labeler behind our labels, and RadCliQ includes SembScore; this is the largest gain
   in the table. The claim should rest on GREEN, CRIMSON, RaTEScore and RadGraph (all
   positive here).
8. **"Small model" framing.** The pooled classifier sees 480k labelled studies, and the
   same report text is used as training labels for the classifier (train split) and as
   examples (train split). This is fine for leakage, since val/test are never used for
   training or retrieval. But "labels from a small model" also means "a model trained on
   the corpus's own reports", and the paper should say so.
9. **MIMIC targets are 960 of 1,663 val studies** (both sections required). Mention it
   once, so readers don't compare it with ReXrank's MIMIC Findings set.
10. **The prior report helps both arms equally**, but its coverage differs a lot (61% vs
    26%), so dataset-level numbers mix studies with and without a prior. If you want a
    per-subset table, it is cheap to compute from the existing scores.
11. **Which judge leads, and the paper's message (open).** §3.4 is the strongest
    evidence for either framing. Labels look much larger on CRIMSON, ReXrank's primary
    metric. Same-reader style matching shows up only on GREEN. If the paper leads with
    CRIMSON, labels are the main finding and same-reader retrieval becomes a
    metric-limitations point. Table 4 is from the older configuration. Re-running
    same-reader retrieval on the current one would settle whether the pattern holds.
12. **CRIMSON failures** are 0.1–0.6% per condition. The only conditions that fail
    significantly more than retrieval-only are oracles: on ReXGradient 0.22% vs 0.11%,
    and MIMIC same-reader oracle 0.57%. They were not inspected, since the oracle is not
    a method and the share is tiny.

### Open directions (from your notes, 2026-11-25)

- **Label vs image matching.** Jaccard first, image cosine as a tiebreak, works as a
  hard filter to the exact label set. It costs image similarity: the mean cosine of the
  5 examples falls from .52 to .37 (MIMIC) and from .54 to .44 (ReXGradient), and only
  20% / 31% of examples are the same as retrieval-only's. Pursue this through the
  retrieval literature:
  - retrieve-then-rerank: image top-N, then a label rerank
  - score fusion (α·Jaccard + (1−α)·cosine) or reciprocal rank fusion
  - soft label overlap, weighted by classifier probability
  - MMR for diversity among the 5 examples
- **Same-patient retrieval for the style arm.** Valid with a temporal guard. On MIMIC
  val, the latest prior shares the target's radiologist 14% of the time (8% for a
  random train study). 37% of targets with a prior have some earlier same-reader study
  of the same patient. Same-patient reports also share content, so the arm needs a
  control that separates style from content.

