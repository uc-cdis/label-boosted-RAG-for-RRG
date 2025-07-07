REPO_ROOT=/opt/gpudata/steven/label-boosted-RAG-for-RRG
OUTPUT_DIR=/opt/gpudata/labrag
EXP_DIR=exp-baselines

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/cxrrepair_impression.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/cxrrepair_impression_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/cxrredone_impression.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/cxrredone_impression_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/xrem_impression.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/xrem_impression_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/cxrmate_impression.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/cxrmate_impression_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/chexagent_impression.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/chexagent_impression_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/cxrmate_findings.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/cxrmate_findings_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/chexagent_findings.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/chexagent_findings_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/rgrg_findings.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/rgrg_findings_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/cxrmate_both.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/cxrmate_both_METRICS.csv

python $REPO_ROOT/rrg/eval.py \
--report_csv $OUPUT_DIR/$EXP_DIR/chexagent_both.csv \
--output_csv $OUPUT_DIR/$EXP_DIR/chexagent_both_METRICS.csv