# LaB-RAG Journal Revision Plan (v6)

Status: **DRAFT v6, 2026-11-08. Design only; nothing implemented yet.**

- **Target:** MDPI *Machine Learning and Knowledge Extraction* (MAKE), special issue "Multimodal Data Fusion for Complementary Feature Learning and Knowledge Discovery". Submit by April 2027.
- **Title:** unchanged, so Google Scholar keeps merging it with the arXiv preprint.
- **This version:** v6 applies Steven's review of v5.
  - Story fixed: the "small models" are the label classifiers. The 4B fine-tuned model only shows that LaB-RAG works with fine-tuning.
  - OpenI dropped.
  - New condition: **filter only** (the preprint's naive-format arm). Template baseline and Jaccard filter confirmed.
  - D split back into D (image access) and E (medical training).
  - F uses engineered CoT prompting for older models instead of turning thinking off. CoT is the default for every model without native thinking.
  - Metrics aligned with ReXrank, including its LLM judges. F1-CheXbert only appears in the error breakdown.
  - Budget: a thinking-length pilot comes before C.
  - Items marked *(proposed)* still need confirmation.

---

## 1. Story

**Small models** predict structured labels from the image. By default these are classifiers over frozen image embeddings. The labels boost **frozen, text-only LLMs** for radiology report generation. The story is unchanged from the preprint. The old numbers do not need to be reproduced exactly.

The revision tests the obvious objection: *"LLMs keep getting better. Won't labels from small models become unnecessary?"* Each experiment measures **how much labels add as a function of X**, where X is something that makes the LLM stronger: scale, thinking, access to the image, medical training, or model age. The framing is **not** "labels always win".

- If the gain from labels shrinks as X grows, that is still a result.
- **Cost argument:** the label classifiers are orders of magnitude smaller than every LLM in the plan, including the 4B student in §6. They help LLMs at every size we test, from a 4B model to ≈400B.
- **Fine-tuning (§6)** only shows that LaB-RAG also works when the LLM is fine-tuned. It is not part of the cost argument.

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

**Dropped:**
- **VinDr-CXR:** it has no reports.
- **BRAX:** transfer (J) already covers external validation.
- **OpenI:** its reports are public on the web, so it was never an RRG test. Its only remaining role was external classifier testing, and the cross-dataset classifier tests (§4.3) already cover that for CheXbert-14. It has no labels in the PadChest vocabulary, so it can't test the rich-label classifiers either.

**Sections:** Findings; Impression (generated from the image only, not conditioned on Findings); Findings+Impression (joint).

### Leakage controls
- **Our leakage-controlled RAD-DINO:** trained on the train splits of CheXpert Plus, MIMIC-CXR, ReXGradient and PadChest. It excludes every validation and test patient of those datasets, plus the PadChest-GR validation and test patients.
- **Classifiers:** fit on train, thresholds set on validation. Wherever predicted labels are used as training inputs, they are K-fold out-of-fold predictions.
- **Retrieval:** never returns the target study or another study from the same patient.
- **Audit in the paper:**
  - the public RAD-DINO `training_images.csv` checked against our test sets
  - the stated pretraining data of the baseline models

---

## 3. Conditions

| Condition | Image to model | Retrieved reports | Labels in prompt | What it isolates |
|---|:-:|:-:|:-:|---|
| No context | ✓ | – | – | the model alone. **Only defined when the model sees the image** (D) |
| Retrieval only (Standard RAG) | opt. | ✓ (image similarity) | – | retrieval without labels |
| **Filter only** | opt. | ✓ (label filter) | – | labels used **only to choose the retrieved reports**. This is the preprint's "Label Filter only" arm (exact filter + naive prompt) |
| Labels only | opt. | – | ✓ | labels used **only in the prompt**, with no retrieval: the LLM writes the report from the predicted labels |
| LaB-RAG | opt. | ✓ (label filter) | ✓ (simple format) | the full method |
| Template | – | – | ✓ | labels with **no LLM**: each label is rendered into a fixed sentence ("Cardiomegaly." / "No pleural effusion."). This is the small model alone, the floor the LLM has to beat (cf. Pragmatic RRG, Nguyen et al. 2023) |

- **Text-only runs (the default)** use retrieval only, labels only, LaB-RAG and the template floor. **Filter only** is added in B. Other experiments keep the three main conditions to stay within budget. "No context" is omitted because a text-only LLM with no context gets no study-specific input.
- **Label source:** predicted by default. **Oracle** labels (extracted from the reference report) and **random** labels (sampled from label prevalence) are the controls.
- **Reasoning:**
  - Models with native thinking use it; the default budget is chosen in the pilot (Appendix).
  - Models without it (Mistral v0.1 and v0.3, Gemma 2, Gemma 3, MedGemma) get one **engineered CoT prompt**: reason first, then write the report after a fixed delimiter. The prompt is tuned on validation and shared across these models.
  - Only the text after the delimiter is scored.
  - A (reproduction) keeps the preprint prompts.
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
| Mistral | Mistral-7B-Instruct v0.1 · v0.3 (preprint anchor) · Ministral 3 8B (instruct and reasoning variants) · Mistral Small 4 |
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
| **B. Core** | Do labels boost current reasoning LLMs? | Both main models, **text only**, × {retrieval only, filter only, labels only, LaB-RAG} × {predicted, oracle, random} labels, plus the template floor. MIMIC and ReXGradient × all three sections. |
| **C. Scale × thinking** | How does the gain from labels change with model size and thinking? | Fully crossed. **Qwen3.5:** {4B, 27B, 122B, 397B} × {thinking off, plus 3 thinking budgets with the same token caps at every size} × {retrieval only, labels only, LaB-RAG}. **gpt-oss:** {20b, 120b} × {low, medium, high} × the same three conditions. Budgets come from the pilot (Appendix). |
| **D. Image access** | Do labels still help when the model sees the image? | {Qwen3.5-27B, MedGemma 27B} × {text only, with image} × {retrieval only, labels only, LaB-RAG}, plus **no context** in the image runs. Each model is compared with itself in the two input modes. MedGemma is the hardest case: a medical VLM that sees the image. The Qwen text-only cells reuse B. |
| **E. Medical training** | Does medical tuning substitute for labels? | Full 2×2: {Gemma 3 27B, MedGemma 27B} × {with image, text only} × the three conditions, plus **no context** in the image runs. **Headline: with image**, which is how MedGemma is meant to run and where its medical training (largely visual) can actually substitute for labels. **Secondary: text only**, which mirrors the preprint's Mistral vs BioMistral comparison. One question (medical training), reported in both input modes. The multimodal MedGemma checkpoint is used in both modes (a footnote notes the separate text-only checkpoint), so all MedGemma cells are shared with D. **Leakage caveat:** MedGemma's chest X-ray training includes MIMIC-CXR with no stated test exclusion, so ReXGradient is the clean read. |
| **F. Model age** | Does the gain from labels survive model progress at a fixed size? | Mistral-7B v0.1 → v0.3 → Ministral 3 8B; Gemma 2 27B → Gemma 3 27B → Gemma 4 31B. Text only. **Each model reasons the way its era allowed:** the engineered CoT prompt for models without native thinking, native thinking for Ministral 3 8B (reasoning variant) and Gemma 4 31B. The reasoning method counts as part of what a model generation is: older models can't think natively, and nobody runs CoT prompts on models that can. So it is not a second factor under the §3 rule. The Gemma 3 cells are shared with E. |
| **G. Generality** | Does LaB-RAG help across many models? | Retrieval only vs LaB-RAG on every generator, including OLMo 3 32B Think, Muse Glimmer 30B and Mistral Small 4. Presented as one table or figure. |
| **H. Preprint ablations** | Filter, format, number of retrieved reports | Run fresh, **on the best configuration only**: 7 one-at-a-time variants. Filter only is now a core condition (B). |
| **I. Encoders** | Does the encoder choice matter? | Our leakage-controlled RAD-DINO (default) vs public RAD-DINO, MedSigLIP, DINOv3, and BioViL-T / GLoRIA. All of them get classifier metrics; one main model also runs retrieval only vs LaB-RAG with each. |
| **J. Transfer** | Do the classifiers and corpus transfer to other datasets? | One main model × {retrieval only, LaB-RAG}. **Symmetric:** both MIMIC and ReXGradient act as the source, and each transfers to the other primary dataset plus CheXpert Plus and PadChest-GR (CheXbert-14 labels). Two settings: (1) **source only**: source classifiers and source corpus; (2) **corpus swap**: source classifiers with the target's train corpus. Caveat: CheXpert Plus Findings test is 61 studies, so its CIs are wide. |

### 4.3 Supporting analyses
- **Classifier quality:**
  - per-label AUROC and F1, plus retrieval label-precision@k, for every encoder × label set
  - in-domain test sets: MIMIC, ReXGradient and CheXpert Plus
  - **cross-dataset** (replaces OpenI): each source's classifiers on the other datasets, the same pairs as J
  - PadChest-GR test for the rich labels
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

- **Retrieval filter:**
  - Exact label-set match stops working as the vocabulary grows: almost no corpus study shares an exact 60- or 250-label vector with the target.
  - At **every** level, use a soft filter: rank candidates by positive-label **Jaccard** overlap with the target, breaking ties by image similarity.
  - The existing `partial` filter counts shared positives only, so it favours corpus reports with many positives. Jaccard penalizes the extra positives.
  - Exact vs soft is compared at the CheXbert-14 level only, which links §5 back to the default pipeline.
- **Design:** both main models × {labels only, LaB-RAG} × {predicted, oracle} × each level.
- **Secondary:** MIMIC-CXR and ReXGradient, where corpus labels in the rich vocabularies come from classifiers run on the corpus images.
- **Note:** with the PadChest vocabularies, the boosting labels no longer come from CheXbert, so the SembScore and RadCliQ overlap (§7) does not apply. This makes §5 a check on the CheXbert-14 results that does not share the labeler.

---

## 6. Fine-tuning and distillation: **TBD**

Purpose: show that LaB-RAG works when the LLM is fine-tuned. It is not a cost argument (§1).

Settled so far:
- The student is Qwen3.5-4B (multimodal, and the same family as the main model).
- The teacher is chosen on **MIMIC + ReXGradient validation** (never test), using the label-independent metrics, from Qwen3.5-397B, Mistral Large 4 and Muse Spark Open. It can use up to 16×H200.
- Distillation is pure: frozen-teacher outputs only, with no ground truth or metric in the targets.
- LoRA, with out-of-fold predicted labels and self-excluding retrieval.

Deferred: the set of arms, joint vs per-section training, whether Impression is included, retrieval in the image arms, the evaluation sets.

---

## 7. Metrics
- **Suite: the ReXrank metrics**: BLEU, BERTScore, SembScore, RadGraph F1, RaTEScore, 1/RadCliQ-v1, **GREEN** and **CRIMSON**.
  - Run locally where possible. GREEN uses a fixed open-weight model. **To check:** which judge model CRIMSON uses and whether it runs locally. If it needs an API, decide whether to keep it.
  - The LLM judges cost one generation per scored report, so they are counted in the budget (Appendix).
  - **F1-CheXbert is not in the suite** (ReXrank doesn't report it). It only appears in the error breakdown below.
- **Label-space overlap:** one table footnote.
  - SembScore uses CheXbert, the labeler that produces the boosting labels, and RadCliQ includes SembScore.
  - No claim rests on these two alone. BLEU, BERTScore, RadGraph F1, RaTEScore, GREEN and CRIMSON do not depend on the label set.
- **Error breakdown (where F1-CheXbert appears) *(proposed)*:**
  - Three stages: **label accuracy** (classifier vs CheXbert on the reference, §4.3) → **adherence** (CheXbert on the generated report vs the labels in the prompt) → **end-to-end quality** (the suite above).
  - With **oracle** labels, the prompt labels *are* CheXbert's labels for the reference report. So F1-CheXbert against the reference measures adherence directly. This answers the reviewers' question of why it stays below 1 (preprint: 0.942 on MIMIC Findings).
  - The gap from 1 has two sources:
    - **labeler noise:** CheXbert misreading generated text. The template baseline with oracle labels measures this floor: it is clean text rendered from the exact labels.
    - **LLM non-adherence:** omitted labels, hedging, or findings copied from retrieved reports.
  - A small sample of disagreements is sorted into these categories by hand.
- **Reasoning models:** only the final report is scored. Truncation and failure rates are reported.

---

## 8. Infrastructure
- **Envs:** fresh `labrag` and `labrag-eval` environments.
- **Stages:**
  - `embed` → `classify` (including out-of-fold) → `build-prompts` (hashed JSONL)
  - `generate`: an async client against `vllm serve`, resumable and cached
  - `evaluate`: per-study scores. The LLM judges are served the same way as the generators.
- **GPU wrapper (sbatch):**
  - tensor-parallel size taken from `CUDA_VISIBLE_DEVICES`
  - free port, health check, teardown
  - multi-node serving for the large teacher and 397B models
- **Record per run:** model revision, vLLM version, sampling params, seed, reasoning settings, thinking-token counts.

---

## 9. Timeline
| Window | Milestones |
|---|---|
| Nov 2026 | finalize the design (§6); exclusion list → start leakage-controlled RAD-DINO training; envs and pipeline; reproduction (A); metric suite; **thinking-length pilot**; CoT prompt for older models |
| Dec 2026 | classifiers; core (B) with public RAD-DINO, then swap to our checkpoint; scale × thinking (C) |
| Jan 2027 | D, E, F, G; transfer (J); richer-labels classifiers; teacher selection and teacher outputs |
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
| B. Core: 2 models × 10 condition/label cells × 36k studies (3 sections) | 724k |
| C. Scale × thinking (66 cells, minus cells shared with B) | 790k |
| D. Image access (11 new cells) | 136k |
| E. Medical training (7 new Gemma 3 cells; MedGemma shared with D) | 86k |
| F. Model age (15 new cells; Gemma 3 shared with E) | 185k |
| G. Generality: the 3 extra models only; the rest reuse runs above | 74k |
| H. Preprint ablations | 86k |
| I. Encoders (5 alternatives × 2 conditions) | 123k |
| J. Transfer (symmetric) | 57k |
| §5 Richer labels (headline + secondary) | 165k |
| Teacher selection | 15k |
| **Total, excluding fine-tuning and distillation** | **≈2.5M** |

### Thinking length is the main unknown
Steven's preliminary runs show that Qwen's highest thinking modes are slow. The v5 estimate assumed 2.5k output tokens per generation, which is probably too low for the top budget.

- **Pilot (Nov, before C):**
  - About 200 MIMIC validation studies per Qwen3.5 size × candidate budget, plus gpt-oss at each effort level.
  - Measure thinking tokens (mean and p95), truncation rate, H200 throughput, and validation quality.
- **Budgets are hard token caps** (e.g. ≈1k / 4k / 16k, set from the pilot), not the model's open-ended top mode. The top cap is where validation quality levels off or where cost becomes prohibitive, whichever comes first.
- **The default thinking setting for B** (and D–J) is a middle budget, not the top one.
- **Rough sensitivity:** if the top-budget cells average ≈10k tokens, those 148k generations alone add ≈1.5B tokens. The total would then be ≈7–8B output tokens, or ≈450–900 H200-hours. Re-estimate after the pilot.
- **Cuts, in order, if the pilot says it's too slow:**
  1. Top budget for retrieval only vs LaB-RAG only, dropping labels only (≈−49k of the most expensive generations).
  2. No top budget at 397B.
  3. 3 thinking budgets → 2 (≈−150k).
  4. Filter only in B for Findings only (≈−143k).

  All of these are better than going back to a ReXGradient subset.
- **Not counted:**
  - the LLM judges: ≈2.5M generations each, but short and on small models
  - the other evaluation metrics
  - leakage-controlled RAD-DINO pretraining
  - teacher outputs for distillation and the LoRA runs
  - the template floor (no GPU)
