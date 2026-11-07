# LaB-RAG Journal Revision Plan (v4)

Status: **DRAFT v4, 2026-11-07. Design only; nothing implemented yet.**

- **Target:** MDPI *Machine Learning and Knowledge Extraction* (MAKE), special issue "Multimodal Data Fusion for Complementary Feature Learning and Knowledge Discovery". Submit by April 2027.
- **Title:** unchanged, so Google Scholar keeps merging it with the arXiv preprint.
- **This version:** v4 replaces v3. It drops the dataset and model codes, organizes the model comparisons around one question per axis, and moves compute estimates to the appendix. Fine-tuning and distillation are **TBD** (§6).

---

## 1. Story

Small models (per-label classifiers over frozen image embeddings) predict structured labels. Those labels boost **frozen, text-only LLMs** for radiology report generation. The story is unchanged from the preprint. The old numbers do not need to be reproduced exactly.

The revision tests the obvious objection: *"LLMs keep getting better. Won't labels from small models become unnecessary?"* Each experiment measures **how much labels add as a function of X**, where X is something that makes the LLM stronger: scale, thinking, medical training, access to the image, or model age. The framing is **not** "labels always win".

- If the gain from labels shrinks as X grows, that is still a result.
- The cost argument holds either way: a ≈4B model plus logistic regression, compared with a ≈400B model.

Fusion framing for the special issue: labels are a compact, interpretable, late-fused modality connecting separately pretrained vision and language models.

---

## 2. Datasets

| Dataset | Role |
|---|---|
| **MIMIC-CXR** | Primary. RRG test (official split), classifiers, retrieval corpus, training data for fine-tuning |
| **ReXGradient-160K** | Primary. RRG test (official patient-disjoint split, 67 sites), classifiers, retrieval corpus |
| **CheXpert Plus** | Secondary RRG test (Findings and Impression only; the test set is small) |
| **PadChest-GR** | Richer-labels test bed (§5). **Official split: 3,185 train / 455 val / 915 test.** English finding sentences; labels are in the PadChest vocabulary (all 154 verified) |
| **PadChest** | Labels only (its reports are stemmed Spanish). Trains the rich-label classifiers. Also used in encoder pretraining |
| **OpenI** | **Not an RRG test.** Its reports are public on the web, so LLM leakage can't be ruled out. Used for classifier validation and the memorization probe |
| **BRAX** | External classifier test only |

**Dropped:** VinDr-CXR (no reports).

**Sections:** Findings; Impression (generated from the image only, not conditioned on Findings); Findings+Impression (joint).

### Leakage controls
- **Our leakage-controlled RAD-DINO:** trained on the train splits of CheXpert Plus, MIMIC-CXR, ReXGradient and PadChest. It excludes every validation and test patient of those datasets, the PadChest-GR validation and test patients, and all of OpenI and BRAX.
- **Classifiers:** fit on train, thresholds set on validation. Wherever predicted labels are used as training inputs, they are K-fold out-of-fold predictions.
- **Retrieval:** never returns the target study or another study from the same patient.
- **Audit in the paper:**
  - the public RAD-DINO `training_images.csv` checked against our test sets
  - the pretraining data of the baseline models
  - an LLM memorization probe (report prefix → verbatim continuation)
  - a search of OLMo 3's open pretraining corpus (Dolma 3) for report text

---

## 3. Conditions

| Condition | Retrieved reports | Labels |
|---|:-:|:-:|
| No context | – | – |
| Retrieval only (Standard RAG) | ✓ | – |
| Labels only | – | ✓ |
| LaB-RAG | ✓ (exact label filter) | ✓ (simple format) |

- **Label source:** predicted by default. **Oracle** labels (extracted from the reference report) and **random** labels (sampled from label prevalence) are the controls.
- **"With image":** the same conditions, with the image also passed to a multimodal model.
- **Statistics:** 1 seed. Paired bootstrap over studies, with 95% CIs on the difference between conditions.
- **Rule:** no claim varies two factors unless the full cross of those factors is run.

---

## 4. Models and comparisons

### 4.1 Roster (17 generators)

| Family | Models |
|---|---|
| Qwen3.5 | 4B · **27B (main)** · 122B-A10B · 397B-A17B |
| gpt-oss | 20b · **120b (main)** |
| Gemma | Gemma 2 27B · Gemma 3 27B · Gemma 4 31B · MedGemma 27B |
| Mistral | Mistral-7B-Instruct v0.1 · v0.3 (preprint anchor) · Ministral 3 8B · Mistral Small 4 |
| Others | OLMo 3 32B Think · Muse Glimmer 30B |

- **Teacher candidates** (for distillation, chosen on MIMIC validation, §6): Qwen3.5-397B, Mistral Large 4, Muse Spark Open.
- **Main models:** Qwen3.5-27B (multimodal, controllable thinking) and gpt-oss-120b (strictly text-only, with low/medium/high effort). Gemma 3 27B is not a main model because it has no reasoning-effort control.
- **To check at implementation time:** checkpoint availability, licences, chat templates, and how each model enforces a thinking budget.

### 4.2 Experiments

**Datasets per experiment:**
- **Main models:** all three sections, on MIMIC-CXR and ReXGradient.
- **All other experiments:** Findings only, on MIMIC-CXR test plus a 3,000-study site-stratified subset of ReXGradient test, unless stated otherwise.

| # | Question | Design |
|---|---|---|
| **A. Reproduction** | Does the new stack reproduce the preprint? | Mistral-7B v0.3 with greedy decoding; the preprint's Standard RAG / filter only / format only / LaB-RAG configurations; MIMIC and CheXpert Plus × Findings and Impression. Reuses the surviving BioViL-T and GLoRIA embeddings and classifiers, and compares against the paper's tables (the old generations are gone). |
| **B. Core** | Do labels boost current reasoning LLMs? | Both main models × all conditions, with predicted, oracle and random labels. Qwen3.5-27B is also run with the image. MIMIC and ReXGradient × all three sections. |
| **C. Scale × thinking** | How does the gain from labels change with model size and thinking? | Fully crossed. **Qwen3.5:** {4B, 27B, 122B, 397B} × {thinking off, plus 3 thinking budgets with the same token caps at every size} × {retrieval only, labels only, LaB-RAG}. **gpt-oss:** {20b, 120b} × {low, medium, high} × the same three conditions. |
| **D. Medical training** | Does domain tuning substitute for labels? | Gemma 3 27B vs MedGemma 27B × {no context, retrieval only, labels only, LaB-RAG}. Neither model has a reasoning mode. This mirrors the preprint's Mistral vs BioMistral comparison. |
| **E. Image** | Do labels still help when the model sees the image? | Qwen3.5-27B (in B), plus Gemma 3 27B, MedGemma 27B and Gemma 4 31B, each run with and without the image. |
| **F. Model age** | Does the gain from labels survive model progress at a fixed size? | Mistral-7B v0.1 → v0.3 → Ministral 3 8B; Gemma 2 27B → Gemma 3 27B → Gemma 4 31B. **Thinking off** everywhere, for comparability. A thinking-on point for Ministral 3 8B and Gemma 4 31B is an extra marker. |
| **G. Generality** | Does LaB-RAG help across many models? | Retrieval only vs LaB-RAG on every generator, including OLMo 3 32B Think, Muse Glimmer 30B and Mistral Small 4. Presented as one table or figure. |
| **H. Preprint ablations** | Filter, format, number of retrieved reports | Run fresh, **on the best configuration only**: 7 one-at-a-time variants. |
| **I. Encoders** | Does the encoder choice matter? | Our leakage-controlled RAD-DINO (default) vs public RAD-DINO, MedSigLIP, DINOv3, and BioViL-T / GLoRIA. All of them get classifier metrics; one main model also runs retrieval only vs LaB-RAG with each. |
| **J. Transfer** | Do the classifiers and corpus transfer to other datasets? | One main model × {retrieval only, LaB-RAG} in two settings: corpus swap, and source dataset only. Directions: MIMIC → ReXGradient, CheXpert Plus, PadChest-GR; ReXGradient → MIMIC. |

### 4.3 Supporting analyses
- **Classifier quality:**
  - per-label AUROC and F1, plus retrieval label-precision@k, for every encoder × label set
  - test sets: MIMIC, ReXGradient and CheXpert Plus; external tests on OpenI and BRAX; PadChest-GR for the rich labels
- **Literature baselines:** 3–5 strong current public models (chosen using ReXrank; weights to be verified), on MIMIC and ReXGradient only. The preprint's six baselines are cited from its tables. No ReXrank submission.
- **Memorization probe:** every text model, on 1k reports each from MIMIC, ReXGradient, CheXpert Plus and OpenI. Plus the corpus search for OLMo 3.

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

- **Design:** both main models × {labels only, LaB-RAG} × {predicted, oracle} × each level.
- **Secondary:** MIMIC-CXR and ReXGradient, where corpus labels in the rich vocabularies come from classifiers run on the corpus images.
- **Note:** with the PadChest vocabularies, F1-CheXbert is no longer computed on the same label set used for boosting.

---

## 6. Fine-tuning and distillation: **TBD**

Settled so far:
- The student is Qwen3.5-4B (multimodal, and the same family as the main model).
- The teacher is chosen on MIMIC validation, never on test, from Qwen3.5-397B, Mistral Large 4 and Muse Spark Open. It can use up to 16×H200.
- Distillation is pure: frozen-teacher outputs only, with no ground truth or metric in the targets.
- LoRA, with out-of-fold predicted labels and self-excluding retrieval.

Still to work through with Steven:
- the set of arms
- joint vs per-section training
- whether Impression is included
- retrieval in the image arms
- the evaluation sets

---

## 7. Metrics
- **Suite:** the ReXrank metrics (CRIMSON, 1/RadCliQ-v1, BLEU, BERTScore, SembScore, RadGraph F1, RaTEScore, GREEN). All judges run locally.
- **In-loop under CheXbert-14 boosting:** F1-CheXbert, SembScore and RadCliQ share the label space used for boosting, so they are reported but not used as primary evidence.
- **Primary claims:** label-independent metrics, plus RadGraph F1 restricted to entities outside the label set.
- **Error breakdown:**
  - label accuracy → the LLM's adherence to the labels → end-to-end quality
  - plus a template baseline (labels rendered into a fixed template, no LLM)
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
| Nov 2026 | finalize the design (§6); exclusion list → start leakage-controlled RAD-DINO training; envs and pipeline; reproduction (A); metric suite; memorization probe |
| Dec 2026 | classifiers; core (B) with public RAD-DINO, then swap to our checkpoint; scale × thinking (C) |
| Jan 2027 | D–G; transfer (J); richer-labels classifiers; teacher selection and teacher outputs |
| Feb 2027 | fine-tuning and distillation; richer-labels generation; H, I; literature baselines |
| Mar 2027 | analysis and writing (MDPI template); related-work update |
| early Apr 2027 | co-author review, submit |

---

## 10. Reviewer debts → where they are addressed
- **Per-label classifier performance and class imbalance:** §4.3.
- **Filter and format gains with predicted labels:** H, plus the oracle controls in B.
- **Why F1-CheXbert < 1 with oracle labels** (contradictory retrievals, or the LLM ignoring labels): the error breakdown (§7).
- **PromptMRG and Pragmatic-Llama:** literature baselines or discussion.
- **IU X-ray / OpenI:** excluded from RRG, with the reason backed by the memorization probe and the corpus search.
- **Missing citations:** related-work update.

---

## Appendix: rough generation budget

Assumptions:
- 1 seed.
- Test sizes in studies (F = Findings, I = Impression, F+I = Findings+Impression):
  - MIMIC test: F 2,347 · I 2,224 · F+I 1,624
  - ReXGradient: full test is 10,000 per section; the subset is 3,000
  - CheXpert Plus: F 61 · I 200
  - PadChest-GR test: 915
- "Findings default" = MIMIC test + ReXGradient subset, Findings only = 5,347 studies.

| Experiment | ≈ Generations |
|---|---|
| A. Reproduction | 19k |
| B. Core: 3 model runs (Qwen text, Qwen image, gpt-oss); predicted-label conditions on full ReXGradient, controls on the subset | 617k |
| C. Scale × thinking (48 Qwen cells + 18 gpt-oss cells) | 353k |
| D–F. Medical training, image, age (12 model runs × 4 conditions) | 257k |
| G. Generality: the 3 extra models only; the rest reuse runs above | 32k |
| H. Preprint ablations | 37k |
| I. Encoders (5 alternatives × 2 conditions) | 53k |
| J. Transfer | 25k |
| §5 Richer labels (headline + secondary) | 90k |
| Memorization probe + teacher selection | 80k |
| **Total, excluding fine-tuning and distillation** | **≈1.6M** |

- **Tokens and GPU-hours:** at roughly 2.5k output tokens per generation (with thinking), that is about 4B output tokens, or roughly 250–500 H200-hours. The 397B runs and the long thinking budgets dominate.
- **Not counted:** evaluation (LLM judges), leakage-controlled RAD-DINO pretraining, teacher outputs for distillation, LoRA runs.
- **Largest cut if needed:** in B, run the predicted-label conditions on the 3,000-study ReXGradient subset instead of the full 10,000 test. That saves about 360k generations.
