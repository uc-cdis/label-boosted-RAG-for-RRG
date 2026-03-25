REPO_ROOT=/opt/gpudata/trung/label-boosted-RAG-for-RRG
MIMIC_CXR_DIR=/opt/gpudata/mimic-cxr
CHEXPERTPLUS_DIR=/opt/gpudata/chexpertplus
OUTPUT_DIR=$REPO_ROOT/labrag

BATCH_SIZE=64
NUM_WORKERS=8

python $REPO_ROOT/rrg/extract.py \
--model_type biovil-t \
--input_path $MIMIC_CXR_DIR/files \
--file_ext ".jpg" \
--output_h5 $OUTPUT_DIR/mimic-cxr-biovilt.h5 \
--batch_size $BATCH_SIZE \
--num_workers $NUM_WORKERS

python $REPO_ROOT/rrg/extract.py \
--model_type resnet50 \
--input_path $MIMIC_CXR_DIR/files \
--file_ext ".jpg" \
--output_h5 $OUTPUT_DIR/mimic-cxr-resnet50.h5 \
--batch_size $BATCH_SIZE \
--num_workers $NUM_WORKERS

python $REPO_ROOT/rrg/extract.py \
--model_type gloria \
--model_path $REPO_ROOT/../../models/GLoRIA/chexpert_resnet50.ckpt \
--input_path $CHEXPERTPLUS_DIR/PNG \
--file_ext ".png" \
--output_h5 $OUTPUT_DIR/chexpertplus-gloria.h5 \
--batch_size $BATCH_SIZE \
--num_workers $NUM_WORKERS

python $REPO_ROOT/rrg/extract.py \
--model_type resnet50 \
--input_path $CHEXPERTPLUS_DIR/PNG \
--file_ext ".png" \
--output_h5 $OUTPUT_DIR/chexpertplus-resnet50.h5 \
--batch_size $BATCH_SIZE \
--num_workers $NUM_WORKERS