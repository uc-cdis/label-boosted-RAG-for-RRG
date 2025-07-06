REPO_ROOT=/opt/gpudata/steven/label-boosted-RAG-for-RRG
MIMIC_CXR_DIR=/opt/gpudata/mimic-cxr
CHEXPERTPLUS_DIR=/opt/gpudata/chexpertplus
LABEL_DIR=/opt/gpudata/cxr-derived
BASE_OUTPUT_DIR=/opt/gpudata/labrag

set -e

# =================================
# Core Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type no-filter \
--prompt_type naive \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type naive \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_exact_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type no-filter \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-core/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# Filter Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type no-filter \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter/findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter/findings_top-5_mimic-cxr-biovilt-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type partial \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter/findings_top-5_mimic-cxr-biovilt-pred-label_partial_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-filter/findings_top-5_mimic-cxr-biovilt-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# Prompt Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type naive \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type verbose \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type instruct \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-prompt/findings_top-5_mimic-cxr-biovilt-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# LLM Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.1 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.1.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model BioMistral/BioMistral-7B \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_BioMistral-7B.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-llm/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_BioMistral-7B_METRICS.csv

# =================================
# Embedding Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-embedding

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-embedding/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-embedding/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-resnet50-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-resnet50.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-embedding

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-embedding/findings_top-5_mimic-cxr-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-embedding/findings_top-5_mimic-cxr-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# True Label Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-true-label

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-true-label/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-true-label/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv None \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-true-label

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-true-label/findings_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-true-label/findings_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# Top K Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 3 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k/findings_top-3_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k/findings_top-3_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k/findings_top-5_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type findings \
--k 10 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--true_label_csv $LABEL_DIR/mimic-findings-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/mimic-findings-biovilt-classifiers/pred_pr.csv \
--report_csv $MIMIC_CXR_DIR/mimic_cxr_sectioned.csv \
--feature_h5 $BASE_OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k/findings_top-10_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-mimic/exp-findings/exp-top-k/findings_top-10_mimic-cxr-biovilt-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv
