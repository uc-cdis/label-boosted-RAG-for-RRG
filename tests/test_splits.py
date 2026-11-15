"""`labrag splits` must rewrite every committed split file byte for byte."""

import pytest

from labrag import paths
from labrag.data import DATASETS


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_write_splits_reproduces_committed_files(tmp_path, dataset):
    adapter = DATASETS[dataset]
    if not (paths.CXR_ROOT / adapter.SOURCE_CSV).exists():
        pytest.skip(f"{dataset} source not mounted")
    adapter.write_splits(paths.CXR_ROOT, tmp_path)
    for name in adapter.SPLIT_FILES:
        assert (tmp_path / name).read_text() == (paths.SPLITS_DIR / name).read_text()
