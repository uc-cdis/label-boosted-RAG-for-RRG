export CUDA_VISIBLE_DEVICES=1

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/cxrrepair_impression.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/cxrrepair_impression_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/cxrredone_impression.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/cxrredone_impression_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/xrem_impression.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/xrem_impression_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/cxrmate_impression.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/cxrmate_impression_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/chexagent_impression.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/chexagent_impression_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/cxrmate_findings.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/cxrmate_findings_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/chexagent_findings.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/chexagent_findings_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/rgrg_findings.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/rgrg_findings_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/cxrmate_both.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/cxrmate_both_METRICS.csv

python /opt/gpudata/steven/label-boosted-RAG-for-RRG/rrg/eval.py \
--report_csv /opt/gpudata/rrg-data-2/baselines/chexagent_both.csv \
--output_csv /opt/gpudata/rrg-data-2/baselines/chexagent_both_METRICS.csv