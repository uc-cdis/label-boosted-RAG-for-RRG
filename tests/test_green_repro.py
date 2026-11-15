"""GREEN's upstream reproduction test (external/GREEN/tests/test_repro.py) through each
judge path, on upstream's exact inputs. Needs Docker and one GPU, so it runs only when
asked for, in a job:

  slurm/submit.sh --gres=gpu:1 -- pytest -p no:cacheprovider -m gpu tests/test_green_repro.py

(no cache: a job's commit checkout is never modified)
"""

import pytest

from labrag.stages import evaluate


@pytest.mark.gpu
@pytest.mark.parametrize("metric", ["green", "green-native"])
def test_upstream_repro(metric):
    evaluate.repro(metric)
