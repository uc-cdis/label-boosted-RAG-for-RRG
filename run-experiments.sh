export CUDA_VISIBLE_DEVICES=1
# remember to set HF_TOKEN!

# Findings - Filter Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-filter
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-filter/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-filter/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type partial --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-filter
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-filter/Mistral-7B-Instruct-v0.3_partial_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-filter/Mistral-7B-Instruct-v0.3_partial_pred-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type no-filter --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-filter
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-filter/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-filter/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_findings_METRICS.csv

# Findings - Prompt Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type naive --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type verbose --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_verbose_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_verbose_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type instruct --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_instruct_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_instruct_top-5_findings_METRICS.csv

# Findings - Model Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-model
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-model/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-model/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.1 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-model
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-model/Mistral-7B-Instruct-v0.1_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-model/Mistral-7B-Instruct-v0.1_exact_pred-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model BioMistral/BioMistral-7B --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-model
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-model/BioMistral-7B_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-model/BioMistral-7B_exact_pred-label_simple_top-5_findings_METRICS.csv

# Findings - Label Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv  --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-label
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-label/Mistral-7B-Instruct-v0.3_exact_true-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-label/Mistral-7B-Instruct-v0.3_exact_true-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-label
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-label/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-label/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings_METRICS.csv

# Findings - Prompt/Filter Redundancy Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type no-filter --prompt_type naive --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_naive_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_naive_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type no-filter --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type naive --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_findings_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings.csv --output_csv /opt/gpudata/rrg-data-2/exp-findings/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings_METRICS.csv

# Impression - Filter Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-filter
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-filter/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-filter/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type partial --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-filter
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-filter/Mistral-7B-Instruct-v0.3_partial_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-filter/Mistral-7B-Instruct-v0.3_partial_pred-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type no-filter --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-filter
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-filter/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-filter/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_impression_METRICS.csv

# Impression - Prompt Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type naive --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type verbose --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_verbose_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_verbose_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type instruct --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-prompt
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_instruct_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-prompt/Mistral-7B-Instruct-v0.3_exact_pred-label_instruct_top-5_impression_METRICS.csv

# Impression - Model Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-model
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-model/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-model/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.1 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-model
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-model/Mistral-7B-Instruct-v0.1_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-model/Mistral-7B-Instruct-v0.1_exact_pred-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model BioMistral/BioMistral-7B --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-model
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-model/BioMistral-7B_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-model/BioMistral-7B_exact_pred-label_simple_top-5_impression_METRICS.csv

# Impression - Label Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv  --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-label
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-label/Mistral-7B-Instruct-v0.3_exact_true-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-label/Mistral-7B-Instruct-v0.3_exact_true-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-label
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-label/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-label/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression_METRICS.csv

# Impression - Prompt/Filter Redundancy Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type no-filter --prompt_type naive --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_naive_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_naive_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type no-filter --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_no-filter_pred-label_simple_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type naive --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_naive_top-5_impression_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression.csv --output_csv /opt/gpudata/rrg-data-2/exp-impression/exp-redundancy/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression_METRICS.csv

# Section Experiments
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type both --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-section
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-section/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_both.csv --output_csv /opt/gpudata/rrg-data-2/exp-section/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_both_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type findings-intersect --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-section
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-section/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings-intersect.csv --output_csv /opt/gpudata/rrg-data-2/exp-section/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_findings-intersect_METRICS.csv

python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/generate.py --model mistralai/Mistral-7B-Instruct-v0.3 --filter_type exact --prompt_type simple --section_type impression-intersect --k 5 --batch_size 32 --prompt_yaml /opt/gpudata/label-boosted-RAG-for-RRG/rrg/prompts.yaml --split_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv --metadata_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv --true_label_csv /opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv --predicted_label_csv /opt/gpudata/rrg-data-2/image-labels/pred_pr.csv --report_csv /opt/gpudata/mimic-cxr/mimic_cxr_sectioned.csv --feature_h5 /opt/gpudata/rrg-data-2/biovilt-features.h5 --output_dir /opt/gpudata/rrg-data-2/exp-section
python /opt/gpudata/label-boosted-RAG-for-RRG/rrg/eval.py --report_csv /opt/gpudata/rrg-data-2/exp-section/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression-intersect.csv --output_csv /opt/gpudata/rrg-data-2/exp-section/Mistral-7B-Instruct-v0.3_exact_pred-label_simple_top-5_impression-intersect_METRICS.csv
