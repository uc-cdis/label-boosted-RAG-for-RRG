REPO_ROOT=/opt/gpudata/steven/label-boosted-RAG-for-RRG
LABEL_DIR=/opt/gpudata/cxr-derived
MIMIC_CXR_DIR=/opt/gpudata/mimic-cxr
CHEXPERTPLUS_DIR=/opt/gpudata/chexpertplus
OUTPUT_DIR=/opt/gpudata/labrag
NUM_WORKERS=4

##############################################################################

# Findings, BioViL-T, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_results $OUTPUT_DIR/mimic-findings-biovilt-classifiers \
--num_workers $NUM_WORKERS

# Findings, ResNet50, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-resnet50.h5 \
--feature_key img_proj \
--output_results $OUTPUT_DIR/mimic-findings-resnet50-classifiers \
--num_workers $NUM_WORKERS

# Impression, BioViL-T, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--output_results $OUTPUT_DIR/mimic-impression-biovilt-classifiers \
--num_workers $NUM_WORKERS

# Impression, ResNet50, MIMIC-CXR
python $REPO_ROOT/rrg/classify.py \
--split_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-split.csv \
--metadata_csv $MIMIC_CXR_DIR/mimic-cxr-2.0.0-metadata.csv \
--label_csv $LABEL_DIR/mimic-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/mimic-cxr-resnet50.h5 \
--feature_key img_proj \
--output_results $OUTPUT_DIR/mimic-impression-resnet50-classifiers \
--num_workers $NUM_WORKERS

##############################################################################

# Findings, GLoRIA, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_results $OUTPUT_DIR/chexpertplus-findings-gloria-classifiers \
--num_workers $NUM_WORKERS

# Findings, ResNet50, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-findings-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-resnet50.h5 \
--feature_key img_proj \
--output_results $OUTPUT_DIR/chexpertplus-findings-resnet50-classifiers \
--num_workers $NUM_WORKERS

# Impression, GLoRIA, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-gloria.h5 \
--output_results $OUTPUT_DIR/chexpertplus-impression-gloria-classifiers \
--num_workers $NUM_WORKERS

# Impression, ResNet50, CheXpert+
python $REPO_ROOT/rrg/classify.py \
--split_csv $CHEXPERTPLUS_DIR/split.csv \
--metadata_csv $CHEXPERTPLUS_DIR/metadata.csv \
--label_csv $LABEL_DIR/chexpertplus-impression-labels.csv \
--feature_h5 $OUTPUT_DIR/chexpertplus-resnet50.h5 \
--feature_key img_proj \
--output_results $OUTPUT_DIR/chexpertplus-impression-resnet50-classifiers \
--num_workers $NUM_WORKERS
