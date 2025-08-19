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
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-best-pred.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type naive \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-best-pred.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type no-filter \
--prompt_type simple \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-best-pred.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-best-pred.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core-best/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv
