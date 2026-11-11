# LaB-RAG Journal Revision Plan (v12)

Status: **v12, 2026-11-10.** Data prep stage.

- **Target:** MDPI *Machine Learning and Knowledge Extraction* (MAKE), special issue "Multimodal Data Fusion for Complementary Feature Learning and Knowledge Discovery". Submit by April 2027.
- **Title:** unchanged, so Google Scholar keeps merging it with the arXiv preprint.

---

## 1. Story

**Small models** predict structured labels from the image. By default these are classifiers over frozen image embeddings. The labels boost **frozen, text-only LLMs** for radiology report generation. The story is unchanged from the preprint. The old numbers do not need to be reproduced exactly.

The revision tests the obvious objection: *"LLMs keep getting better. Won't labels from small models become unnecessary?"* Each experiment measures **how much labels add as a function of X**, where X is something that makes the LLM stronger: scale, thinking, access to the image, medical training, or model age. The framing is **not** "labels always win".

- If the gain from labels shrinks as X grows, that is still a result.
- **Cost argument:** the label classifiers are orders of magnitude smaller than every LLM in the plan, including the 4B student in §6. They help LLMs at every size we test, from 0.8B to ≈400B.
- **Labels transfer cheaply (§5):** classifiers trained on another dataset (PadChest, from Spain, in its own label vocabulary) may improve generation on MIMIC-CXR and ReXGradient with no labels from those datasets and no tuning of the classifiers on them. That costs far less than fine-tuning an LLM on the target dataset.
- **Fine-tuning (§6)** only shows that LaB-RAG also works when the LLM is fine-tuned. §6 itself is not a cost argument; §5.2 borrows its text retrieval-only arms as a cost reference.

Fusion framing for the special issue: labels are a compact, interpretable, late-fused modality connecting separately pretrained vision and language models.

---

## 2. Datasets

| Dataset | Role |
|---|---|
| **MIMIC-CXR** | Primary. RRG test (official split), classifiers, retrieval corpus, training data for fine-tuning |
| **ReXGradient-160K** | Primary. RRG test (official patient-disjoint split, 67 sites; **full 10k test everywhere**), classifiers, retrieval corpus |
| **CheXpert Plus** | The **outside dataset**: never an RRG test. (1) **Pilot** (Appendix): every generation setting is chosen here, not on the evaluation datasets. (2) **Sole transfer source** for J (classifiers + corpus). Its official valid split has only 61 Findings studies (61 patients), so the preprint's CheXpert Plus test results are not re-run. |
| **PadChest-GR** | Richer-labels test bed (§5). **Official split: 3,185 train / 455 val / 915 test.** English finding sentences; labels are in the PadChest vocabulary (all 154 verified) |
| **PadChest** | Labels only (its reports are stemmed Spanish). Trains the rich-label and localized-label classifiers (§5). About 76% of its studies were labelled by the authors' RNN model, not by physicians. To state in the paper. |

**Sections:** Findings; Impression (generated from the image only, not conditioned on Findings); Findings+Impression (joint).

### Data provenance
- **Inputs are only the files the dataset authors released.** Every derived file (sections, labels, splits) is produced by code in this repo. Locally derived files from earlier work are never read.
- **MIMIC-CXR sections:** the dataset authors' own section parser (MIT-LCP `mimic-cxr`), used unmodified as a pinned submodule under `external/`.
- **CheXpert Plus reports:** the release has one report row per image, and images of the same study can carry slightly different anonymized text ([Stanford-AIMI/chexpert-plus#13](https://github.com/Stanford-AIMI/chexpert-plus/issues/13); the clinical content is the same). We keep one canonical text per study: the first image row in a fixed order. Labels are computed on that text.
- **MIMIC-CXR labelled test set** (`mimic-cxr-2.1.0-test-set-labeled.csv`, 687 studies): 9 of its studies are in the official validation split and 2 are not in MIMIC-CXR. It is filtered to official test.
- **Reference labels** (CheXbert-14 on the reference section) are computed in-repo, the same way for every dataset: one labeler at one version, on exactly the section text we generate and score. No author labels fit:
  - **MIMIC-CXR:** its labels come from the CheXpert and NegBio labelers (not CheXbert), on findings-or-impression text rather than a fixed section.
  - **CheXpert Plus:** its CheXbert labels are per image and can differ within a study ([#13](https://github.com/Stanford-AIMI/chexpert-plus/issues/13)).
  - **ReXGradient:** no author labels.
  - **PadChest-GR:** its gold labels are used as released (§5). J also needs CheXbert-14 labels on its Findings (§4.2), and the authors ship none, so those are computed in-repo too.

### Leakage controls
- **CheXpert Plus pilot split:** 400 patients from CheXpert Plus train, each with exactly one Findings study (17,972 of the 26,634 train patients qualify), drawn once with a fixed seed. 200 are the **pilot set** and 200 the **prompt-tuning set** (§3), so the CoT prompt isn't tuned on the studies that then choose between settings. Both are excluded from the CheXpert Plus classifiers and every retrieval corpus.
  - **Encoder caveat:** all 400 pilot patients are in public RAD-DINO's pretraining set, which contains all of CheXpert (both train and valid) as unlabelled images for self-supervised training. This is the same exposure the corpus images have, and the pilot only chooses generation settings and is never reported as a result. To state in the paper.
- **Encoder: public RAD-DINO, not retrained.** Its published `training_images.csv` (882,775 images) was checked against our evaluation sets:
  - **MIMIC-CXR:** no validation or test images.
  - **PadChest-GR:** no validation or test images.
  - **ReXGradient:** not in its training data at all.
  - It does include MIMIC-CXR train, PadChest and CheXpert images, so the classifier and corpus images were seen without labels during pretraining (§5.2).
- **Classifiers:** fit on train, thresholds set on validation. Wherever predicted labels are used as training inputs, they are K-fold out-of-fold predictions.
- **Retrieval:** never returns the target study or another study from the same patient.
- **Audit in the paper:**
  - the RAD-DINO `training_images.csv` check above
  - the stated pretraining data of the other encoders (I) and the baseline models

---

## 3. Conditions

| Condition | Image to model | Retrieved reports | Labels in prompt | What it isolates |
|---|:-:|:-:|:-:|---|
| No context | ✓ | – | – | the model alone. **Only possible when the model sees the image** (D), otherwise the model is not conditioned on any input. |
| Retrieval only (Standard RAG) | opt. | ✓ (image similarity) | – | retrieval without labels |
| Filter only | opt. | ✓ (label ranking) | – | labels used **only to choose the retrieved reports**. This is the preprint's "Label Filter only" arm (naive prompt), but with the Jaccard ranking |
| Labels only | opt. | – | ✓ | labels used **only in the prompt**, with no retrieval: the LLM writes the report from the predicted labels |
| LaB-RAG | opt. | ✓ (label ranking) | ✓ (simple format) | the full method |
| Template | – | – | ✓ | labels with **no LLM**: each label is rendered into a fixed sentence ("Cardiomegaly." / "No pleural effusion."). This is the small model alone, the floor the LLM has to beat (cf. Pragmatic RRG, Nguyen et al. 2023) |

- **Text-only runs (the default)** use retrieval only, LaB-RAG and the template floor. **Filter only** and **labels only** separate the two roles labels play (choosing the retrieved reports, and appearing in the prompt), so they run **in B only**. Every later experiment asks "does LaB-RAG still beat retrieval only as X grows?" "No context" is omitted because a text-only LLM with no context gets no study-specific input.
- **One LaB-RAG configuration everywhere:**
  - **Retrieval:** Jaccard-ranked label retrieval. Corpus studies are ranked by positive-label Jaccard overlap with the target's labels, with ties broken by image similarity. "No Finding" counts as a label, so a normal study has a non-empty label set.
  - **Prompt:** based on the preprint's simple label format, needs tuning in the pilot study.
  - The preprint's filter ablation (no filter / exact / partial) and format ablation (naive / simple / verbose / instruct) are **dropped**.
- **Label source:** predicted by default. **Oracle** labels (extracted from the reference report) and **random** labels (sampled from label prevalence) are the controls.
- **Reasoning:**
  - Models with native thinking use it; the default budget is chosen in the pilot (Appendix).
  - Models without it (Mistral v0.1 and v0.3, Gemma 2, Gemma 3, Qwen2.5) get one **engineered CoT prompt**: reason first, then write the report after a fixed delimiter. The prompt is tuned on the CheXpert Plus prompt-tuning set (§2) and shared across these models.
    - Only the text after the delimiter is scored.
  - **MedGemma reasoning: chosen empirically in the pilot.** MedGemma 27B **has a trained thinking mode** (both checkpoints).
    - **Google's documentation:** the official MedGemma notebooks (`Google-Health/medgemma`, `quick_start_with_hugging_face.ipynb`) say "MedGemma 1.5 4B and MedGemma 27B variants support thinking". Thinking is switched on by putting `SYSTEM INSTRUCTION: think silently if needed.` before the role line of the system prompt, e.g. "You are an expert radiologist."
    - **Output format:** the model writes `<unused94>thought\n…<unused95>` and then the final response. The notebooks split on `<unused95>`.
    - **Decoding:** greedy. The output cap goes from 300–1,000 tokens without thinking to 1,300–2,000 with it. There is no budget control beyond the cap.
    - **Google's caveat:** thinking "may not benefit some use cases" and "may potentially improve results on challenging text-only tasks". They say to validate it.
    - **Decision rule:** the pilot compares three settings on the CheXpert Plus pilot set: native thinking on, thinking off (direct answer), and our engineered CoT prompt. Each checkpoint uses its best setting in every D and E cell. The choice and the pilot numbers are reported.
    - **Note for E:** if MedGemma ends up with native thinking while Gemma 3 uses the CoT prompt, the difference in reasoning training counts as part of what "medical training" means for MedGemma. That's the same argument F uses for model generations.
    - **Implementation:** vLLM has no reasoning parser for this format, so split on `<unused95>` ourselves. The `<unused9x>` tokens must be kept when decoding.
- **Statistics:** 1 seed. Paired bootstrap over studies, with 95% CIs on the difference between conditions.
- **Rule:** no claim varies two factors unless the full cross of those factors is run.

---

## 4. Models and comparisons

### 4.1 Roster

| Family | Models |
|---|---|
| Qwen3.5 | 0.8B · 2B · 4B · 9B · **27B (main)** · 122B-A10B · 397B-A17B |
| Qwen (other releases, F only) | Qwen2.5-32B · Qwen3-32B · Qwen3.6-27B · Qwen3.8-27B |
| gpt-oss | 20b · **120b (main)** |
| Gemma | Gemma 2 27B · Gemma 3 27B · Gemma 4 31B · MedGemma 27B (multimodal and text-only checkpoints) |
| Mistral | Mistral-7B-Instruct v0.1 · v0.3 (preprint anchor) · Ministral 3 8B Reasoning · Mistral Small 4 |
| Others | Muse Glimmer 30B |

- **Teacher** for the reasoning traces in §6: Qwen3.5-397B (same family as the 4B student, so the thinking format and tokenizer match).
- **Precision:** every reported generation runs in **BF16**, including Qwen3.5-397B in C and the pilot. The only exception is the §6 teacher, which runs the official FP8 checkpoint (`Qwen/Qwen3.5-397B-A17B-FP8`, block-wise FP8 with block size 128). The teacher only produces training data that no result compares against, so the difference in precision never enters a comparison. To state in the paper.
- **Main models:** Qwen3.5-27B (multimodal, controllable thinking) and gpt-oss-120b (strictly text-only, with low/medium/high effort).
- **Why Qwen3.5 and not the newer Qwen3.6 / 3.8** (to state in the paper): Qwen3.5 is the newest Qwen release with open weights at every size from 0.8B to 397B. C needs that range, and §6 needs the 4B student and the 397B teacher from the same release. The open Qwen3.6 models are 27B and 35B-A3B only. The open Qwen3.8 models are 27B, a ≈180B MoE, and a 2.4T text-only model. The F Qwen line checks whether the main result carries over to the newest 27B.
- **MedGemma checkpoints: both, in the text-only cells.** MedGemma 27B has a multimodal checkpoint (`medgemma-27b-it`) and a separate text-only one (`medgemma-27b-text-it`, trained on medical text only). The MedGemma technical report's headline 27B numbers come from the text-only checkpoint, and it calls the multimodal 27B results preliminary.
  - The **multimodal checkpoint** runs in both input modes. D's within-model comparison then changes only the input, and the checkpoint matches Gemma 3 27B, which is also multimodal and runs text-only in E.
  - The **text-only checkpoint** also runs every text-only MedGemma cell in D and E: retrieval only and LaB-RAG, about 25k generations. It is reported next to the multimodal checkpoint, so text-only MedGemma is shown at its best.
- **Ministral 3 8B: one variant, the Reasoning one.** Mistral releases two post-trained 8B models. **Instruct** is tuned for chat (SFT + online DPO) and answers directly. **Reasoning** is post-trained separately from the base model (SFT on CoT data, then GRPO and online DPO), thinks before answering, and targets math, code and STEM. They also have different recommended sampling settings (Instruct temperature ≈0.1; Reasoning 0.7 / top-p 0.95). Only the Reasoning variant is used, in both F and G, which matches F's rule that each model reasons the way its era allowed. Sanity check in the pilot: Instruct with the CoT prompt on the same CheXpert Plus pilot studies. If Instruct is clearly better, it **replaces** Reasoning everywhere, so F and G never mix the two.
- **MedGemma 1.5 is out of scope.** It is released only as a 4B multimodal model (Jan 2026), while D and E need a 27B medical model matched to Gemma 3 27B. MedGemma 1 27B, in both checkpoints, is still the current MedGemma at that size. To say in the paper if asked.
- **To check at implementation time:** checkpoint availability, licences, chat templates, and how each model enforces a thinking budget.

### 4.2 Experiments

**Datasets per experiment:**
- **Main models (B):** all three sections, on MIMIC-CXR and ReXGradient.
- **All other experiments, including fine-tuning (§6):** Findings only, on the MIMIC-CXR and ReXGradient test sets, unless stated otherwise.

| # | Question | Design |
|---|---|---|
| **B. Core** | Do labels boost current reasoning LLMs? | Both main models, **text only**, × {retrieval only, filter only, labels only, LaB-RAG} × {predicted, oracle, random} labels, plus the template floor. MIMIC and ReXGradient × all three sections. |
| **C. Scale × thinking** | How does the gain from labels change with model size and thinking? | Fully crossed. **Qwen3.5:** {0.8B, 2B, 4B, 9B, 27B, 122B, 397B} × {thinking off, plus 3 thinking budgets with the same token caps at every size} × {retrieval only, LaB-RAG}. **gpt-oss:** {20b, 120b} × {low, medium, high} × the same two conditions. Budgets come from the pilot (Appendix). All Qwen3.5 sizes have the same thinking switch. The model cards and the guides disagree on whether the small models think by default, so `enable_thinking` is always set explicitly. The pilot checks that 0.8B and 2B don't loop or truncate under the larger caps. |
| **D. Image access** | Do labels still help when the model sees the image? | {Qwen3.5-27B, MedGemma 27B} × {text only, with image} × {retrieval only, LaB-RAG}, plus **no context** in the image runs. Each model is compared with itself in the two input modes. MedGemma is the hardest case: a medical VLM that sees the image. The Qwen text-only cells reuse B. |
| **E. Medical training** | Does medical tuning substitute for labels? | Full 2×2: {Gemma 3 27B, MedGemma 27B} × {with image, text only} × {retrieval only, LaB-RAG}, plus **no context** in the image runs. **Headline: with image**, which is how MedGemma is meant to run and where its medical training (largely visual) can actually substitute for labels. **Secondary: text only**, which mirrors the preprint's Mistral vs BioMistral comparison. One question (medical training), reported in both input modes. The text-only cells run both MedGemma checkpoints (§4.1). The MedGemma cells are shared with D. **Leakage caveat:** MedGemma's chest X-ray training includes MIMIC-CXR with no stated test exclusion, so ReXGradient is the clean read. |
| **F. Model age** | Does the gain from labels survive model progress at a fixed size? | Three lines, text only (release year in brackets):<br>• Mistral-7B v0.1 (2023) → v0.3 (2024) → Ministral 3 8B Reasoning (2025)<br>• Gemma 2 27B (2024) → Gemma 3 27B (2025) → Gemma 4 31B (2026)<br>• Qwen2.5-32B (2024) → Qwen3-32B (2025) → Qwen3.5-27B (2026) → Qwen3.6-27B (2026) → Qwen3.8-27B (2026). Qwen2 is left out because it has no model near 30B. Three of the Qwen releases come from 2026, so plot against release date, not release index.<br> **Each model reasons the way its era allowed:** the engineered CoT prompt for models without native thinking, native thinking for Ministral 3 8B Reasoning, Gemma 4 31B and Qwen3 onwards. The reasoning method counts as part of what a model generation is: older models can't think natively, and nobody runs CoT prompts on models that can. So it is not a second factor under the §3 rule. The Gemma 3 cells are shared with E; the Qwen3.5-27B cells are shared with B. |
| **G. Generality** | Does LaB-RAG help across many models? | Retrieval only vs LaB-RAG on every generator, including Muse Glimmer 30B and Mistral Small 4. Presented as one table or figure. |
| **H. Retrieval depth** | Does the gain from labels depend on how many reports are retrieved? | One main model at its default thinking setting. **Number of retrieved reports k ∈ {1, 3, 5 (default), 10, 20} × {retrieval only, LaB-RAG}**. Both conditions are swept, so the label gain can be read at each k (§3 rule). k = 20 is new: modern long-context models can use more retrieved reports than Mistral-7B could. The filter and format ablations are dropped (§3). |
| **I. Encoders** | Does the encoder choice matter? | Public RAD-DINO (default) vs MedSigLIP and DINOv3. All of them get classifier metrics; one main model also runs retrieval only vs LaB-RAG with each. The preprint's BioViL-T and GLoRIA are dropped: they were strongly dataset-specific in the preprint. (Google's CXR Foundation was considered and rejected: its model card calls it legacy and recommends MedSigLIP, and it runs only in TensorFlow.) **Leakage caveat:** MedSigLIP trains on MIMIC-CXR with no stated test exclusion, so ReXGradient is the clean read. |
| **J. Transfer** | Do the classifiers and corpus transfer to other datasets? | One main model × {retrieval only, LaB-RAG}. **Source: CheXpert Plus only** (classifiers trained on CheXpert Plus train, thresholds on its valid split; corpus = its ≈46k train Findings studies minus the pilot patients). **Targets:** MIMIC, ReXGradient and PadChest-GR (CheXbert-14 labels). Two settings: (1) **source only**: source classifiers and source corpus; (2) **corpus swap**: source classifiers with the target's train corpus. For MIMIC and ReXGradient, B's in-domain result on the same test set is the reference. Retrieval only in the corpus-swap setting doesn't use the classifiers, so it is shared with B and §5. Caveat to report: the source corpus is ≈3× smaller than MIMIC's or ReXGradient's. |

### 4.3 Supporting analyses
- **Classifier quality:**
  - per-label AUROC and F1, plus retrieval label-precision@k, for every encoder × label set
  - in-domain test sets: MIMIC and ReXGradient
  - **cross-dataset**: the CheXpert Plus classifiers on MIMIC, ReXGradient and PadChest-GR, the same pairs as J
  - PadChest-GR test for the rich labels and the localized labels (§5). The only dataset where localized labels can be checked, since MIMIC and ReXGradient have no gold localized labels.
- **Literature baselines:** 3–5 strong current public models (chosen using ReXrank; weights to be verified), on MIMIC and ReXGradient only. Modern models only; overlap with the preprint's baselines is fine. No ReXrank submission.

---

## 5. Richer labels

**Question:** do richer labels help more than the 14 CheXbert labels? "Richer" means two things:
- **a larger vocabulary** (up to all PadChest findings), and
- **localized labels**, which say where a finding is ("pleural effusion, left"), not only that it is there.

All rich labels come from classifiers trained on **PadChest**, the only dataset labelled in this vocabulary and with locations. They are tested twice:
- **§5.1 PadChest-GR:** in-domain, with gold labels.
- **§5.2 Transfer:** the same classifiers on MIMIC-CXR and ReXGradient, which have no PadChest labels at all.

### Label sets

| Vocabulary | Findings | Unlocalized | By side/zone | By location |
|---|---|:-:|:-:|:-:|
| CheXbert-14 | the default 14 labels | ✓ | n/a | n/a |
| PadChest-14 | PadChest collapsed to 14: same size as the default, different source | ✓ | ✓ | ✓ |
| PadChest-30 | the 30 most common PadChest findings | ✓ | later | later |
| PadChest-60 | the 60 most common PadChest findings | ✓ | ✓ | ✓ |
| PadChest-all | all PadChest findings | ✓ | later | later |

- **Localization columns** (cuts of the location tree, §5.3):
  - **By side/zone:** "pleural effusion, left", "opacity, left lower zone".
  - **By location:** the most specific location the data supports: "opacity, left lower lobe".
  - **n/a:** CheXbert has no locations.
- **9 label sets in all:** 5 unlocalized + 4 localized (PadChest-14 and PadChest-60, each cut two ways).
- **Decision for later:** whether to also localize PadChest-30 and PadChest-all, based on how the 4 localized sets do.
- **Same everywhere:**
  - **Classifier:** the default logistic regression on the global image embedding, for every label set, so localized vs unlocalized never also compares classifier types. A localization adapter on the encoder is out of scope for now.
  - **Training data:** PadChest, minus the PadChest-GR validation and test patients. Thresholds are set on PadChest-GR validation (gold labels). PadChest-GR is a subset of PadChest, so this exclusion is what keeps the PadChest-GR test images out of training. The encoder is already clean (§2), and PadChest itself is never a test set. The same exclusion protects J's transfer to PadChest-GR and the PadChest-GR rows in §4.3.
  - **Retrieval:** the default Jaccard ranking (§3). This is also why Jaccard is the default: exact label-set match stops working as the vocabulary grows, since almost no corpus study shares an exact 60-label (or larger) vector with the target. (The preprint's `partial` filter counts shared positives only, so it favours corpus reports with many positives. Jaccard penalizes the extra positives.)
- **No shared labeler:** with the PadChest label sets, the boosting labels no longer come from CheXbert, so the SembScore and RadCliQ overlap (§7) does not apply. §5 is a check on the CheXbert-14 results that does not share the labeler.

### 5.1 PadChest-GR (headline)
- **Test:** the official PadChest-GR test split (915 studies), Findings only. The reference report is the study's English sentences (positive and negative).
- **Corpus:** PadChest-GR train, with its **gold** labels. This mirrors how CheXbert labels extracted from reports serve the default pipeline.
- **Labels:** predicted by the PadChest classifiers; oracle runs use the target's gold labels (gold localized labels for the localized sets, §5.3).
- **Design:** both main models × LaB-RAG × {predicted, oracle} × each of the 9 label sets (localized included), plus one retrieval-only cell per model (retrieval only does not depend on the label set).
- **Reading:** larger vocabulary vs smaller; localized vs unlocalized at the same vocabulary.

### 5.2 Transfer to MIMIC-CXR and ReXGradient
Can classifiers cheaply trained on **another dataset**, in **a richer vocabulary**, improve generation on MIMIC-CXR and ReXGradient with no labels from them and no tuning of the classifiers on them? If so, it is a much cheaper way to adapt to a new dataset than fine-tuning an LLM on it (§6). This is a finding in its own right, not a secondary check.

- **Nothing from the target's labels is used.** The corpus labels are predicted by the same classifiers from the corpus images, so no report labeler is needed. The only target-domain input is the corpus reports, which every retrieval condition uses anyway.
  - Caveat to state: public RAD-DINO's self-supervised pretraining includes the unlabelled MIMIC-CXR train images (but no ReXGradient images, so ReXGradient is the clean read). The classifiers never see the target data.
- **Design:** MIMIC-CXR and ReXGradient Findings test, both main models, predicted labels only (these datasets have no gold PadChest labels):
  - retrieval only *(shared with B)*
  - LaB-RAG with in-domain CheXbert-14 labels *(shared with B)*: the in-domain reference
  - LaB-RAG with each PadChest label set, localized ones included
- **Reading:** PadChest labels vs retrieval only is the transfer gain. PadChest labels vs in-domain CheXbert-14 shows how much of the in-domain gain the foreign labels recover. Vocabulary size and localization are read as in §5.1.
- **Relation to J:** J transfers CheXbert-14 classifiers between datasets that use the **same** vocabulary. Here the source differs in country, report language and vocabulary.
- **Cost comparison with fine-tuning (2×2):** two ways to adapt the same Qwen3.5-4B to MIMIC-CXR and ReXGradient, each with and without reasoning.

| Adaptation | Thinking off | Teacher's thinking budget |
|---|:-:|:-:|
| Frozen 4B + LaB-RAG with PadChest labels *(2 new cells)* | ✓ | ✓ |
| Fine-tuned 4B, text retrieval only *(from §6)* | plain SFT | SFT with reasoning |

  - Reasoning modes are matched as in §6's frozen reference rows: plain SFT runs with thinking off, and SFT with reasoning thinks at the teacher's budget.
  - The frozen row uses a single PadChest label set: the one of the 8 (localized included) that is best on the CheXpert Plus pilot set (PadChest classifiers run on its images and its corpus), so §5.2 uses nothing from the target except the corpus reports.
  - Report the compute for each: minutes of classifier training vs H200-hours of LoRA (plus the teacher rationales for SFT with reasoning).
  - Fully crossed, so the effect of reasoning can be read for each adaptation method (§3 rule). Each method is a bundle (classifiers + label ranking + labels in the prompt, vs fine-tuned weights), so the claim is about the two methods as wholes.

### 5.3 Building the localized labels
- **Source:** PadChest's `LabelsLocalizationsBySentence` pairs each finding with the locations in the same sentence. PadChest-GR's per-sentence labels and locations give the gold localized labels.
- **Location tree:** PadChest's anatomic location hierarchy (Bustos et al. 2020, arXiv 1901.07441v2, Appendix A.1.3). No author file ships it, so it is transcribed into `resources/padchest_locations.tsv` and cited. A location such as "left lower lobe" has two parents: an anatomic one, which gives the side, and a lung-field zone, which gives upper, middle or lower. "By side/zone" and "by location" are two cuts of this tree.
- **Rare pairs back off.** PadChest has 3,608 finding × location pairs, only 197 of them in ≥100 images, and PadChest-GR validation (455 studies) is too small to set thresholds for rare pairs. A pair with too few examples moves up to its parent location ("left lower lobe" → "left lung" → "lung"), and with no location left it is the plain finding. The minimum count is set at implementation.
- **Pairs that repeat the finding are dropped** (e.g. cardiomegaly + cardiac, aortic elongation + aortic).
- **In retrieval,** a localized label also counts as its plain finding, so "effusion, left" still partly matches "effusion, right".
- **In the template floor,** a localized label renders as "Left pleural effusion."
- **Check first:** the global embedding may not keep left vs right. Measure side accuracy on PadChest-GR test (§4.3) before any generation runs.
- **Metrics:** the location-aware metrics (RadGraph F1, GREEN, CRIMSON) show what localization adds to the reports.

---

## 6. Fine-tuning

Purpose: show that LaB-RAG still works after the LLM is fine-tuned. This is not a cost argument (§1), although §5.2 uses the text retrieval-only arms as a cost reference.

- **Student:** Qwen3.5-4B (multimodal, same family as the main model), trained with LoRA.
- **Findings only**, for both training and evaluation (§4.2). Full Findings train is 152k MIMIC-CXR + 140k ReXGradient ≈ 292k studies.
- **Two recipes.** Both train on the same final text, the **reference report**. They differ only in whether a reasoning trace comes before it.
  - **Plain SFT:** target = the reference report. Run with thinking off at test time.
  - **SFT with reasoning:** target = a rationale, then the reference report. **Qwen3.5-397B** (the teacher) gets the student's exact input (including the image in the image arms) plus the reference report, and writes the reasoning that leads from the input to that report. This is the rationalization step of STaR (Zelikman et al. 2022); see also Distilling step-by-step (Hsieh et al. 2023). At test time the student thinks first, then writes the report.
- **Arms** (one trained model per cell, **10 in total**):

| Input | Plain SFT | SFT with reasoning |
|---|:-:|:-:|
| Text: retrieval only | ✓ | ✓ |
| Text: LaB-RAG | ✓ | ✓ |
| Image only *(reference row)* | ✓ | ✓ |
| Image + retrieval | ✓ | ✓ |
| Image + LaB-RAG | ✓ | ✓ |

- **Clean comparisons:**
  - **Labels:** retrieval only vs LaB-RAG, within each recipe and input mode.
  - **Reasoning:** plain SFT vs SFT with reasoning, for the same input.
  - **Image only** is the standard fine-tuned VLM for report generation. It's a reference row, not a claim: image only vs image + LaB-RAG changes both retrieval and labels at once (§3 rule).
- **Frozen reference rows,** matched by reasoning mode: the frozen Qwen3.5-4B with **thinking off** is the comparison for plain SFT, and with the **teacher's thinking budget** for SFT with reasoning. The text cells come from C, which already crosses 4B with thinking off and the thinking budgets. The image cells (3 inputs × 2 modes) are new and cheap.

Settled:
- Training inputs use **out-of-fold** predicted labels and self-excluding retrieval.
- **No teacher selection.** The student's report is always the reference, so choosing a teacher by the quality of its reports no longer makes sense. Qwen3.5-397B is used because its thinking format matches the student's. The teacher sees the reference, which reverses the earlier "pure distillation" rule.
- **Trace hygiene (no metric involved):** the teacher is told to reason only from the inputs. Traces that mention the reference or quote long spans of it are dropped, and a sample is checked by hand. Known limitation: a text-only input can't support every finding (devices, comparisons). Plain SFT already learns to produce such content; the rationale only makes it explicit.

- **Training-set size: measure first.** Plain SFT needs no teacher, so train it at 20k, 50k and full train (one or two arms) and watch validation quality. Then size the teacher run at the point where quality levels off. Both recipes always use the same training studies. At full train, the teacher run is 5 inputs × 292k ≈ 1.5M generations on Qwen3.5-397B, roughly 500–1,000 H200-hours at a rough throughput estimate. Steven considers that affordable if the learning curve justifies it.

- **Evaluation:** MIMIC-CXR and ReXGradient Findings test, predicted labels.
  - **Label-reliance check:** the models trained with labels are also tested with **oracle** and **random** labels, and compared with the frozen 4B (in the matching reasoning mode) under the same swap. If random labels hurt the fine-tuned model much more than the frozen one, fine-tuning taught it to trust labels blindly.
- **Hyperparameters:** one small learning-rate sweep on validation using one arm. Then the same LoRA settings and step count for every arm, with the checkpoint chosen by validation loss. Arms are not tuned one by one, so differences between them can't be attributed to tuning.

- **No transfer sets for the fine-tuned models.** PadChest-GR's references are sentence lists, so the metrics would mostly measure the style mismatch with MIMIC-CXR and ReXGradient. Transfer would also need its own choice of source vs target corpus. §6's claim is in-domain; J covers transfer for the frozen models.

---

## 7. Metrics
- **Suite: the ReXrank metrics**: BLEU, BERTScore, SembScore, RadGraph F1, RaTEScore, 1/RadCliQ-v1, **GREEN** and **CRIMSON**.
  - **Official implementations, one container per upstream:** CXR-Report-Metric (BLEU-2, BERTScore, SembScore, RadGraph F1, RadCliQ-v1), GREEN, RaTEScore and CRIMSON. They can't share an env: GREEN pins transformers 4.40, while CRIMSON needs transformers ≥5.3. ReXrank's own pipeline also splits them across two envs. RadEval reimplements all of them in one package, but its parity with ReXrank is unverified, so it is not used.
  - Everything runs locally. GREEN uses a fixed open-weight model. **CRIMSON:** by default the `crimson-score` package uses a fine-tuned MedGemma 4B judge (`rajpurkarlab/medgemma-4b-it-crimson`). GPT-5.2 is an optional API backend that we won't use. CRIMSON scores abnormal findings only, and can take patient age and indication as context. Caveat: the judge is from the MedGemma family, which is also a generator in D and E.
  - The LLM judges (GREEN, CRIMSON) generate one output per scored report. That is evaluation cost, not part of the generation budget, which counts only the reports the experiments generate (Appendix, "not counted").
  - **F1-CheXbert is not in the suite** (ReXrank doesn't report it). It only appears in the error breakdown below.
- **Label-space overlap:** one table footnote.
  - SembScore uses CheXbert, the labeler that produces the boosting labels, and RadCliQ includes SembScore.
  - No claim rests on these two alone. BLEU, BERTScore, RadGraph F1, RaTEScore, GREEN and CRIMSON do not depend on the label set.
- **Error breakdown (where F1-CheXbert appears) *(provisional; not critical for the main results, revisit later)*:**
  - Three stages: **label accuracy** (classifier vs CheXbert on the reference, §4.3) → **adherence** (CheXbert on the generated report vs the labels in the prompt) → **end-to-end quality** (the suite above).
  - With **oracle** labels, the prompt labels *are* CheXbert's labels for the reference report. So F1-CheXbert against the reference measures adherence directly. This answers the reviewers' question of why it stays below 1 (preprint: 0.942 on MIMIC Findings).
  - The gap from 1 has two sources:
    - **labeler noise:** CheXbert misreading generated text. The template baseline with oracle labels measures this floor: it is clean text rendered from the exact labels.
    - **LLM non-adherence:** omitted labels, hedging, or findings copied from retrieved reports.
  - A small sample of disagreements is sorted into these categories by hand.
- **Reasoning models:** only the final report is scored. Truncation and failure rates are reported.

---

## 8. Infrastructure
- **Envs:** one uv env (`uv.lock`) per run checkout; vLLM and the metrics (§7) run as containers.
- **Runner** (`slurm/submit.sh` + `slurm/job.sh`). The code, env and source data live on each node's local disk; only the shared storage below is common to all nodes:
  - the submitter pins the run to the current commit, refuses a dirty or unpushed tree, and keeps data-reading jobs off kg35-nvl02 (it has no copy of the data)
  - on each node, the job fetches that commit from GitHub into its own directory and builds the env from the commit's `uv.lock`. Runs at different commits never share code or env.
  - model weights and vLLM images are pre-pulled on every node; the job checks they are present at the pinned revision and digest, and never downloads them
  - **shared storage:** `/opt/gpu_working/steven/labrag-shared`, an NFS directory visible from every node. It holds both the intermediates later stages need and the main results. NFS is easy to overload, so jobs are written to go easy on it:
    - every job, on every node, works on local disk and copies its outputs to NFS once at the end, as a few large files (e.g. one compressed JSONL per generation cell), then renames them into place; never many small files, and never appending to a shared file
    - a job copies the NFS inputs it needs (embeddings, prompts) to local disk once at the start, rather than reading them repeatedly
    - no envs, uv cache, model weights, containers, vLLM caches or training checkpoints on NFS: they stay on local disk (final LoRA adapters are small and can go to NFS)
    - no file locks or SQLite on NFS
- **Stages:**
  - `embed` → `classify` (including out-of-fold) → `build-prompts` (hashed JSONL)
  - `generate`: an async client against `vllm serve`, resumable and cached
  - `evaluate`: per-study scores. The LLM judges are served the same way as the generators.
- **Hard cap: never more than 8 GPUs in use at once** across all our jobs. Of the cluster's 16 H200s, at most 8 are ours at any time. Steven enforces it when scheduling; the runner doesn't.
- **GPU wrapper (sbatch):**
  - tensor-parallel size taken from `CUDA_VISIBLE_DEVICES`
  - free port, health check, teardown
  - multi-node serving on **one rail** (tensor-parallel across nodes on the same rail is faster than crossing sockets within a node):
    - Qwen3.5-397B BF16: 8-way TP across all 4 nodes (C and the pilot)
    - Qwen3.5-122B BF16: 4-way TP across 2 nodes (≈245 GB is too tight for one NVLink pair)
    - §6 teacher, Qwen3.5-397B FP8: 4 GPUs on one rail across 2 nodes, so at most 2 servers at once under the 8-GPU cap
  - everything else on 1 GPU or one NVLink pair
- **Record per run:** model revision, vLLM version, sampling params, seed, reasoning settings, thinking-token counts.

---

## Appendix: rough generation budget

Assumptions:
- 1 seed.
- Test sizes in studies (F = Findings, I = Impression, F+I = Findings+Impression):
  - MIMIC test: F 2,347 · I 2,224 · F+I 1,624
  - ReXGradient test: 10,000 per section
  - PadChest-GR test: 915
- "Findings default" = MIMIC + ReXGradient full test, Findings only = 12,347 studies.

| Experiment | ≈ Generations |
|---|---|
| B. Core: 2 models × 10 condition/label cells × 36k studies (3 sections) | 724k |
| C. Scale × thinking (68 cells, minus 4 shared with B; the 0.8B–9B cells are cheap) | 790k |
| D. Image access (8 new cells) | 99k |
| E. Medical training (5 new Gemma 3 cells + 2 text-only MedGemma checkpoint cells; multimodal MedGemma shared with D) | 86k |
| F. Model age (18 new cells; Gemma 3 shared with E, Qwen3.5-27B with B) | 222k |
| G. Generality: the 2 extra models only; the rest reuse runs above | 49k |
| H. Retrieval depth (8 k cells) | 99k |
| I. Encoders (2 alternatives × 2 conditions) | 49k |
| J. Transfer (CheXpert Plus source → 3 targets; corpus-swap retrieval only shared with B and §5) | 40k |
| §5 Richer labels: §5.1 PadChest-GR 35k + §5.2 transfer 198k + the two 4B cost-comparison cells 25k (the 4 localized sets add ≈15k + 99k) | ≈258k |
| **Total, excluding fine-tuning (§6)** | **≈2.4M** |

### Thinking length is the main unknown
Steven's preliminary runs show that Qwen's highest thinking modes are slow. The v5 estimate assumed 2.5k output tokens per generation, which is probably too low for the top budget.

- **Pilot (before C), on CheXpert Plus only (§2):**
  - **Why CheXpert Plus:** every generation setting is chosen on a dataset we never evaluate on, so the MIMIC and ReXGradient results are not tuned to them. To state in the paper.
  - The 200-study CheXpert Plus pilot set, per Qwen3.5 size × candidate budget, plus gpt-oss at each effort level. Labels come from the CheXpert Plus classifiers (J), and the corpus is CheXpert Plus train minus the pilot patients.
  - The pilot uses public RAD-DINO, like every other experiment (see the encoder caveat in §2).
  - The 397B pilot cells run in BF16, like C (§4.1).
  - Measure thinking tokens (mean and p95), truncation rate, H200 throughput, and pilot quality.
  - The same pilot studies are used for **Ministral 3 8B Reasoning vs Instruct + CoT** (§4.1), the **§5.2 PadChest label set**, and the 0.8B/2B loop check (C).
  - **MedGemma reasoning setting:** native thinking on vs thinking off vs our CoT prompt, for both 27B checkpoints, on the same pilot studies. Measure pilot quality, thinking length and truncation at Google's caps, then pick per checkpoint (§3).
  - **Still tuned on target validation** (part of training on that data, not a generation setting): classifier thresholds, and §6's LoRA learning rate, checkpoint and training-set size.
- **Budgets are hard token caps** (e.g. ≈1k / 4k / 16k, set from the pilot), not the model's open-ended top mode. The top cap is where pilot quality levels off or where cost becomes prohibitive, whichever comes first.
- **The default thinking setting for B** (and D–J) is a middle budget, not the top one.
- **Rough sensitivity:** if the top-budget cells average ≈10k tokens, those ≈220k generations take ≈2.2B tokens, ≈1.7B more than the 2.5k-token assumption. The total would then be ≈7.5B output tokens, or ≈450–850 H200-hours (tokens from the 0.8B–9B models are far cheaper than the rest). Re-estimate after the pilot.
- **Qwen3.5-397B in BF16 takes our whole 8-GPU cap:** 8 GPUs on one rail across all 4 nodes, so only one server runs at a time, and nothing else of ours can run alongside it. C's 397B cells (≈99k generations) can't overlap with any other run, including the §6 teacher. Very rough estimate: about two days; re-estimate after the pilot.
- **Cuts, in order, if the pilot says it's too slow:**
  1. No top budget at 397B.
  2. 3 thinking budgets → 2 (≈−173k: 7 Qwen3.5 sizes × 2 conditions).
  3. Filter only in B for Findings only (≈−143k).

  All of these are better than going back to a ReXGradient subset.
- **Not counted:**
  - the LLM judges: ≈1.9M generations each, but short and on small models
  - the other evaluation metrics
  - §6: the teacher rationales, the LoRA runs, and the test generations of the fine-tuned arms and their new frozen-4B reference rows (the image cells and the oracle/random label-reliance cells)
  - the template floor (no GPU)
