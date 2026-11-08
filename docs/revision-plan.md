# LaB-RAG Journal Revision Plan (v5)

Status: **DRAFT v5, 2026-11-07. Design only; nothing implemented yet.**

- **Target:** MDPI *Machine Learning and Knowledge Extraction* (MAKE), special issue "Multimodal Data Fusion for Complementary Feature Learning and Knowledge Discovery". Submit by April 2027.
- **Title:** unchanged, so Google Scholar keeps merging it with the arXiv preprint.
- **This version:** v5 applies Steven's review of v4.
  - Removed: BRAX, the memorization probe, the LLM-judge metrics, and the reviewer-debt section.
  - Image runs moved out of the core into one image × medical-training experiment (the old D and E). E is retired; the other experiment letters are unchanged.
  - Full ReXGradient test everywhere.
  - Transfer made symmetric.
  - Soft label filter for the rich vocabularies.
  - Items marked *(proposed)* answer questions from the review and still need confirmation.

---

## 1. Story

**Small models** predict structured labels from the image. They are cheap to train compared with serving or tuning the LLM; the default is classifiers over frozen image embeddings, but the claim does not depend on that form. The labels boost **frozen, text-only LLMs** for radiology report generation. The story is unchanged from the preprint. The old numbers do not need to be reproduced exactly.

The revision tests the obvious objection: *"LLMs keep getting better. Won't labels from small models become unnecessary?"* Each experiment measures **how much labels add as a function of X**, where X is something that makes the LLM stronger: scale, thinking, medical training, access to the image, or model age. The framing is **not** "labels always win".

- If the gain from labels shrinks as X grows, that is still a result.
- The cost argument holds either way: training a small model, compared with serving or fine-tuning a ≈400B model.

Fusion framing for the special issue: labels are a compact, interpretable, late-fused modality connecting separately pretrained vision and language models.

---

## 2. Datasets

| Dataset | Role |
|---|---|
| **MIMIC-CXR** | Primary. RRG test (official split), classifiers, retrieval corpus, training data for fine-tuning |
| **ReXGradient-160K** | Primary. RRG test (official patient-disjoint split, 67 sites; **full 10k test everywhere**), classifiers, retrieval corpus |
| **CheXpert Plus** | Secondary RRG test (Findings and Impression only; the test set is small) |
| **PadChest-GR** | Richer-labels test bed (§5). **Official split: 3,185 train / 455 val / 915 test.** English finding sentences; labels are in the PadChest vocabulary (all 154 verified) |
| **PadChest** | Labels only (its reports are stemmed Spanish). Trains the rich-label classifiers. Also used in encoder pretraining |
| **OpenI** | **Not an RRG test.** Its reports are public on the web, so LLM leakage can't be ruled out (argued from that fact, not from a probe). External classifier test |

**Dropped:** VinDr-CXR (no reports), BRAX (OpenI and the cross-dataset transfer runs already cover external validation).

**Sections:** Findings; Impression (generated from the image only, not conditioned on Findings); Findings+Impression (joint).

### Leakage controls
- **Our leakage-controlled RAD-DINO:** trained on the train splits of CheXpert Plus, MIMIC-CXR, ReXGradient and PadChest. It excludes every validation and test patient of those datasets, the PadChest-GR validation and test patients, and all of OpenI.
- **Classifiers:** fit on train, thresholds set on validation. Wherever predicted labels are used as training inputs, they are K-fold out-of-fold predictions.
- **Retrieval:** never returns the target study or another study from the same patient.
- **Audit in the paper:**
  - the public RAD-DINO `training_images.csv` checked against our test sets
  - the stated pretraining data of the baseline models

---

## 3. Conditions

| Condition | Image to model | Retrieved reports | Labels | What it isolates |
|---|:-:|:-:|:-:|---|
| No context | ✓ | – | – | the model alone. **Only defined when the model sees the image** (D) |
| Retrieval only (Standard RAG) | opt. | ✓ | – | retrieval without labels |
| Labels only | opt. | – | ✓ | labels without retrieval: the prompt lists the predicted labels, and the LLM writes the report from them |
| LaB-RAG | opt. | ✓ (label filter) | ✓ (simple format) | the full method |
| Template *(proposed)* | – | – | ✓ | labels with **no LLM**: each label is rendered into a fixed sentence ("Cardiomegaly." / "No pleural effusion."). This is the small model alone, the floor the LLM has to beat |

- **Text-only runs (the default)** use Retrieval only, Labels only, LaB-RAG, and the Template baseline. "No context" is omitted because a text-only LLM with no context gets no study-specific input.
- **Label source:** predicted by default. **Oracle** labels (extracted from the reference report) and **random** labels (sampled from label prevalence) are the controls.
- **Statistics:** 1 seed. Paired bootstrap over studies, with 95% CIs on the difference between conditions.
- **Rule:** no claim varies two factors unless the full cross of those factors is run.

---

## 4. Models and comparisons

### 4.1 Roster

| Family | Models |
|---|---|
| Qwen3.5 | 4B · **27B (main)** · 122B-A10B · 397B-A17B |
| gpt-oss | 20b · **120b (main)** |
| Gemma | Gemma 2 27B · Gemma 3 27B · Gemma 4 31B · MedGemma 27B (multimodal checkpoint) |
| Mistral | Mistral-7B-Instruct v0.1 · v0.3 (preprint anchor) · Ministral 3 8B · Mistral Small 4 |
| Others | OLMo 3 32B Think (fully open training data) · Muse Glimmer 30B |

- **Teacher candidates** (for distillation, chosen on validation, §6): Qwen3.5-397B, Mistral Large 4, Muse Spark Open.
- **Main models:** Qwen3.5-27B (multimodal, controllable thinking) and gpt-oss-120b (strictly text-only, with low/medium/high effort).
- **To check at implementation time:** checkpoint availability, licences, chat templates, and how each model enforces a thinking budget.

### 4.2 Experiments

**Datasets per experiment:**
- **Main models (B):** all three sections, on MIMIC-CXR and ReXGradient (full test).
- **All other experiments:** Findings only, on the MIMIC-CXR and ReXGradient full test sets, unless stated otherwise.

| # | Question | Design |
|---|---|---|
| **A. Reproduction** | Does the new stack reproduce the preprint? | Mistral-7B v0.3 with greedy decoding; the preprint's Standard RAG / filter only / format only / LaB-RAG configurations; MIMIC and CheXpert Plus × Findings and Impression. Reuses the surviving BioViL-T and GLoRIA embeddings and classifiers, and compares against the paper's tables (the old generations are gone). |
| **B. Core** | Do labels boost current reasoning LLMs? | Both main models, **text only**, × {retrieval only, labels only, LaB-RAG} × {predicted, oracle, random} labels, plus the template baseline. MIMIC and ReXGradient × all three sections. |
| **C. Scale × thinking** | How does the gain from labels change with model size and thinking? | Fully crossed. **Qwen3.5:** {4B, 27B, 122B, 397B} × {thinking off, plus 3 thinking budgets with the same token caps at every size} × {retrieval only, labels only, LaB-RAG}. **gpt-oss:** {20b, 120b} × {low, medium, high} × the same three conditions. |
| **D. Image × medical training** | Do labels still help when the model sees the image? Does medical tuning substitute for labels? | {Qwen3.5-27B, Gemma 3 27B, MedGemma 27B} × {text only, with image} × conditions. Image runs add **no context** (image only). Medical training is the Gemma 3 vs MedGemma contrast, in both input modes; this mirrors the preprint's Mistral vs BioMistral comparison. Using the same multimodal MedGemma checkpoint with and without the image keeps the cross clean. The Qwen text-only cells reuse B. |
| ~~E~~ | — | Merged into D. Gemma 4 31B stays in F and G only. |
| **F. Model age** | Does the gain from labels survive model progress at a fixed size? | Mistral-7B v0.1 → v0.3 → Ministral 3 8B; Gemma 2 27B → Gemma 3 27B → Gemma 4 31B. Text only, **thinking off** everywhere for comparability. A thinking-on point for Ministral 3 8B and Gemma 4 31B is an extra marker. |
| **G. Generality** | Does LaB-RAG help across many models? | Retrieval only vs LaB-RAG on every generator, including OLMo 3 32B Think, Muse Glimmer 30B and Mistral Small 4. Presented as one table or figure. |
| **H. Preprint ablations** | Filter, format, number of retrieved reports | Run fresh, **on the best configuration only**: 7 one-at-a-time variants. |
| **I. Encoders** | Does the encoder choice matter? | Our leakage-controlled RAD-DINO (default) vs public RAD-DINO, MedSigLIP, DINOv3, and BioViL-T / GLoRIA. All of them get classifier metrics; one main model also runs retrieval only vs LaB-RAG with each. |
| **J. Transfer** | Do the classifiers and corpus transfer to other datasets? | One main model × {retrieval only, LaB-RAG}. **Symmetric:** both MIMIC and ReXGradient act as the source, and each transfers to the other primary dataset plus CheXpert Plus and PadChest-GR (CheXbert-14 labels). Two settings: (1) **source only**: source classifiers and source corpus; (2) **corpus swap**: source classifiers with the target's train corpus. Caveat: CheXpert Plus Findings test is 61 studies, so its CIs are wide. |

### 4.3 Supporting analyses
- **Classifier quality:**
  - per-label AUROC and F1, plus retrieval label-precision@k, for every encoder × label set
  - test sets: MIMIC, ReXGradient and CheXpert Plus; external test on OpenI; PadChest-GR for the rich labels
- **Literature baselines:** 3–5 strong current public models (chosen using ReXrank; weights to be verified), on MIMIC and ReXGradient only. Modern models only; overlap with the preprint's baselines is fine. No ReXrank submission.

---

## 5. Richer labels (PadChest-GR)

- **Headline:** the official PadChest-GR test split (915 studies), Findings only.
  - The reference report is the study's English sentences (positive and negative).
  - The retrieval corpus is PadChest-GR train, with its **gold** labels. This mirrors how CheXbert labels extracted from reports serve the default pipeline.
  - Target labels are predicted by classifiers trained on PadChest (excluding the PadChest-GR validation and test patients). Oracle runs use the target's gold labels.
- **Label levels:**

| Level | Vocabulary |
|---|---|
| CheXbert-14 | the default label set (CheXbert run on the PadChest-GR sentences) |
| PadChest collapsed to 14 | control: same size as the default, different source |
| PadChest top-30 / top-60 findings | richer |
| PadChest findings + diagnoses + locations | richest |

- **Retrieval filter *(proposed)*:**
  - Exact label-set match stops working as the vocabulary grows: almost no corpus study shares an exact 60- or 250-label vector with the target.
  - At **every** level, use a soft filter: rank candidates by positive-label **Jaccard** overlap with the target, breaking ties by image similarity.
  - The existing `partial` filter counts shared positives only, so it favours corpus reports with many positives. Jaccard penalizes the extra positives.
  - Exact vs soft is compared at the CheXbert-14 level only, which links §5 back to the default pipeline.
- **Design:** both main models × {labels only, LaB-RAG} × {predicted, oracle} × each level.
- **Secondary:** MIMIC-CXR and ReXGradient, where corpus labels in the rich vocabularies come from classifiers run on the corpus images.
- **Note:** with the PadChest vocabularies, F1-CheXbert is no longer computed on the same label set used for boosting.

---

## 6. Fine-tuning and distillation: **TBD**

Settled so far:
- The student is Qwen3.5-4B (multimodal, and the same family as the main model).
- The teacher is chosen on **MIMIC + ReXGradient validation** (never test), using the label-independent metrics, from Qwen3.5-397B, Mistral Large 4 and Muse Spark Open. It can use up to 16×H200.
- Distillation is pure: frozen-teacher outputs only, with no ground truth or metric in the targets.
- LoRA, with out-of-fold predicted labels and self-excluding retrieval.

Deferred: the set of arms, joint vs per-section training, whether Impression is included, retrieval in the image arms, the evaluation sets.

---

## 7. Metrics
- **Suite:** the ReXrank metrics **minus the LLM-based judges**: BLEU, BERTScore, RadGraph F1, RaTEScore, SembScore, 1/RadCliQ-v1, plus F1-CheXbert.
  - **No LLM evaluators.** CRIMSON and GREEN are dropped. If a reviewer asks, GREEN is the one to add, because it is a fixed open-weight local model.
  - All metrics run locally.
- **Label-space overlap:** one short note, not an analysis.
  - F1-CheXbert and SembScore use CheXbert, the labeler that produces the boosting labels. RadCliQ includes SembScore.
  - These are reported with a table footnote. No claim rests on them alone.
  - The BLEU, BERTScore, RadGraph F1 and RaTEScore family is independent of the label set. (The restricted-RadGraph metric from v4 is dropped.)
- **Error breakdown:**
  - label accuracy → the LLM's adherence to the labels → end-to-end quality
  - this also explains why F1-CheXbert stays below 1 with oracle labels
- **Reasoning models:** only the final report is scored. Truncation and failure rates are reported.

---

## 8. Infrastructure
- **Envs:** fresh `labrag` and `labrag-eval` environments.
- **Stages:**
  - `embed` → `classify` (including out-of-fold) → `build-prompts` (hashed JSONL)
  - `generate`: an async client against `vllm serve`, resumable and cached
  - `evaluate`: per-study scores
- **GPU wrapper (sbatch):**
  - tensor-parallel size taken from `CUDA_VISIBLE_DEVICES`
  - free port, health check, teardown
  - multi-node serving for the large teacher and 397B models
- **Record per run:** model revision, vLLM version, sampling params, seed, reasoning settings.

---

## 9. Timeline
| Window | Milestones |
|---|---|
| Nov 2026 | finalize the design (§6); exclusion list → start leakage-controlled RAD-DINO training; envs and pipeline; reproduction (A); metric suite |
| Dec 2026 | classifiers; core (B) with public RAD-DINO, then swap to our checkpoint; scale × thinking (C) |
| Jan 2027 | D, F, G; transfer (J); richer-labels classifiers; teacher selection and teacher outputs |
| Feb 2027 | fine-tuning and distillation; richer-labels generation; H, I; literature baselines |
| Mar 2027 | analysis and writing (MDPI template); related-work update |
| early Apr 2027 | co-author review, submit |

---

## Appendix: rough generation budget

Assumptions:
- 1 seed.
- Test sizes in studies (F = Findings, I = Impression, F+I = Findings+Impression):
  - MIMIC test: F 2,347 · I 2,224 · F+I 1,624
  - ReXGradient test: 10,000 per section
  - CheXpert Plus: F 61 · I 200
  - PadChest-GR test: 915
- "Findings default" = MIMIC + ReXGradient full test, Findings only = 12,347 studies.

| Experiment | ≈ Generations |
|---|---|
| A. Reproduction | 19k |
| B. Core: 2 models × 7 condition/label cells × 36k studies (3 sections) | 507k |
| C. Scale × thinking (66 cells, minus cells shared with B) | 790k |
| D. Image × medical training (18 new cells) | 222k |
| F. Model age (21 new cells) | 259k |
| G. Generality: the 3 extra models only; the rest reuse runs above | 74k |
| H. Preprint ablations | 86k |
| I. Encoders (5 alternatives × 2 conditions) | 123k |
| J. Transfer (symmetric) | 57k |
| §5 Richer labels (headline + secondary) | 165k |
| Teacher selection | 15k |
| **Total, excluding fine-tuning and distillation** | **≈2.3M** |

- **Tokens and GPU-hours:** at roughly 2.5k output tokens per generation (with thinking), that is about 6B output tokens, or roughly 360–730 H200-hours. C (the 397B runs and the long thinking budgets) dominates.
- **Not counted:** evaluation, leakage-controlled RAD-DINO pretraining, teacher outputs for distillation, LoRA runs, the template baseline (no GPU).
- **Largest cut if needed:** in C, go from 3 thinking budgets to 2 (≈−200k). This is better than going back to a ReXGradient subset.
