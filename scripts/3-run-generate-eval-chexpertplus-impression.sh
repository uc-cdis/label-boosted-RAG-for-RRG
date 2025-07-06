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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_no-filter_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-core/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# Filter Experiments
# =================================

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter/impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter/impression_top-5_chexpertplus-gloria-pred-label_no-filter_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type partial \
--prompt_type simple \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter/impression_top-5_chexpertplus-gloria-pred-label_partial_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-filter/impression_top-5_chexpertplus-gloria-pred-label_partial_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# Prompt Experiments
# =================================

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_naive_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type verbose \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_verbose_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type instruct \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-prompt/impression_top-5_chexpertplus-gloria-pred-label_exact_instruct_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# LLM Experiments
# =================================

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.1 \
--filter_type exact \
--prompt_type simple \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.1.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.1_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model BioMistral/BioMistral-7B \
--filter_type exact \
--prompt_type simple \
--section_type impression \
--k 5 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_BioMistral-7B.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-llm/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_BioMistral-7B_METRICS.csv

# =================================
# Embedding Experiments
# =================================

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-embedding

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-embedding/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-embedding/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-resnet50-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-resnet50.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-embedding

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-embedding/impression_top-5_chexpertplus-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-embedding/impression_top-5_chexpertplus-resnet50-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# True Label Experiments
# =================================

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-true-label

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-true-label/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-true-label/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv None \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-true-label

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-true-label/impression_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-true-label/impression_top-5_true-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

# =================================
# Top K Experiments
# =================================

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type impression \
--k 3 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k/impression_top-3_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k/impression_top-3_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

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
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k/impression_top-5_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv

python $REPO_ROOT/rrg/generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--filter_type exact \
--prompt_type simple \
--section_type impression \
--k 10 \
--batch_size 32 \
--prompt_yaml $REPO_ROOT/rrg/prompts.yaml \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--true_label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--predicted_label_csv $BASE_OUTPUT_DIR/chexpertplus-impression-gloria-classifiers/pred_pr.csv \
--report_csv $CHEXPERTPLUS_DIR/report.csv \
--feature_h5 $BASE_OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_dir $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k

python $REPO_ROOT/rrg/eval.py \
--report_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k/impression_top-10_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3.csv \
--output_csv $BASE_OUTPUT_DIR/exp-chexpertplus/exp-impression/exp-top-k/impression_top-10_chexpertplus-gloria-pred-label_exact_simple_Mistral-7B-Instruct-v0.3_METRICS.csv
