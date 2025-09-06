set -e
set -x

bash scripts/cpp-findings-exp/3-run-generate-eval-chexpertplus-findings-better-labels-and-label-subset.sh
bash scripts/cpp-findings-exp/3-run-generate-eval-chexpertplus-findings-better-labels.sh
bash scripts/cpp-findings-exp/3-run-generate-eval-chexpertplus-findings-label-subset.sh
bash scripts/cpp-findings-exp/3-run-generate-eval-chexpertplus-findings-original.sh
