REPO_ROOT=/opt/gpudata/steven/label-boosted-RAG-for-RRG
LABEL_DIR=/opt/gpudata/cxr-derived
MIMIC_CXR_DIR=/opt/gpudata/mimic-cxr
CHEXPERTPLUS_DIR=/opt/gpudata/chexpertplus
OUTPUT_DIR=/opt/gpudata/labrag

##############################################################################

# Findings, BioViL-T, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/mimic-findings-biovilt-lr-hyperparam \
--parallelize_labels 14

# Findings, ResNet50, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-resnet50.h5 \
--feature_key img_proj \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/mimic-findings-resnet50-lr-hyperparam \
--parallelize_labels 14

# Impression, BioViL-T, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/mimic-impression-biovilt-lr-hyperparam \
--parallelize_labels 14

# Impression, ResNet50, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-resnet50.h5 \
--feature_key img_proj \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/mimic-impression-resnet50-lr-hyperparam \
--parallelize_labels 14

##############################################################################

# Findings, GLoRIA, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-gloria.h5 \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/chexpertplus-findings-gloria-lr-hyperparam \
--parallelize_labels 14

# Findings, ResNet50, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-resnet50.h5 \
--feature_key img_proj \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/chexpertplus-findings-resnet50-lr-hyperparam \
--parallelize_labels 14

# Impression, GLoRIA, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-gloria.h5 \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/chexpertplus-impression-gloria-lr-hyperparam \
--parallelize_labels 14

# Impression, ResNet50, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-resnet50.h5 \
--feature_key img_proj \
--model_config $REPO_ROOT/configs/classify/lr-hyperparam.yaml \
--output_results $OUTPUT_DIR/chexpertplus-impression-resnet50-lr-hyperparam \
--parallelize_labels 14
