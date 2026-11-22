"""Fine-tuning RAD-DINO with the attention head: recipe pieces, determinism, and the
finetune -> predict path on two fake datasets with a tiny backbone (CPU)."""

import types

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from labrag import artifacts, data, paths, weights
from labrag.classifiers import finetune
from labrag.stages import finetune as finetune_stage
from labrag.stages import predict

NAMES = ["Edema", "Support Devices", "No Finding"]
TINY = dict(lp_epochs=1, ft_epochs=2, batch_size=8)


def tiny_backbone():
    from transformers import Dinov2Config, Dinov2Model

    torch.manual_seed(123)
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        image_size=56,
        attn_implementation="eager",
    )
    return Dinov2Model(config)


def tiny_processor():
    from transformers import BitImageProcessor

    return BitImageProcessor(
        size={"shortest_edge": 56},
        crop_size={"height": 56, "width": 56},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )


def images(directory, n, seed=0):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        path = directory / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (70, 60), dtype=np.uint8), "L").save(path)
        out.append(path)
    return out


def test_layer_decay_groups():
    model = finetune.build(tiny_backbone(), 3, 1, False)
    groups = finetune.param_groups(model)
    p = finetune.PARAMS
    lrs = sorted({g["lr"] for g in groups[:-1]})
    # embeddings, 2 layers, final norm: 4 depths
    assert lrs == sorted(p["backbone_lr"] * p["layer_decay"] ** k for k in range(4))
    assert groups[-1]["lr"] == p["head_lr"]
    names = {id(t): n for n, t in model.backbone.named_parameters()}
    for g in groups[:-1]:
        for t in g["params"]:
            n = names[id(t)]
            assert n not in finetune.UNUSED
            no_decay = t.ndim < 2 or n.startswith("embeddings.")
            assert g["weight_decay"] == (0.0 if no_decay else p["weight_decay"])
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == len(list(model.parameters())) - len(finetune.UNUSED)


def test_schedule_warms_up_then_decays_to_zero():
    total = 100
    f = [finetune.lr_factor(s, total) for s in range(total)]
    assert f[0] < f[4] == 1.0  # warmup: 5 steps
    assert all(a >= b for a, b in zip(f[4:], f[5:]))
    assert f[-1] < 1e-3


def test_augment_is_a_function_of_its_draws():
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (300, 200), dtype=np.uint8), "L"
    )
    a = finetune.augment(image, np.random.default_rng([1, 2, 3]))
    b = finetune.augment(image, np.random.default_rng([1, 2, 3]))
    c = finetune.augment(image, np.random.default_rng([1, 2, 4]))
    assert a.size == (finetune.SIZE, finetune.SIZE)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_every_dataset_counts_equally():
    Y = np.array([[1], [0], [1], [0], [1], [0]])
    P = np.array([[0.9], [0.1], [0.9], [0.1], [0.1], [0.9]])
    mask = np.ones(Y.shape, dtype=bool)
    group = np.array(["a", "a", "a", "a", "b", "b"])  # a: AUROC 1, b: AUROC 0
    metric, by = finetune.select_metric(Y, P, mask, group)
    assert by == {"a": 1.0, "b": 0.0} and metric == 0.5


def test_datasets_join_in_registry_order():
    with pytest.raises(SystemExit, match="in the order"):
        finetune_stage.split_datasets("finetune", "mimic-chexpertplus")
    with pytest.raises(SystemExit, match="unknown"):
        finetune_stage.split_datasets("finetune", "chexpertplus-nih")
    assert finetune_stage.split_datasets("finetune", "chexpertplus-rexgradient") == [
        "chexpertplus",
        "rexgradient",
    ]


@pytest.mark.parametrize("world", [1, 2])
def test_training_is_deterministic(tmp_path, world):
    paths_ = images(tmp_path / "img", 40)
    rng = np.random.default_rng(1)
    Y = rng.integers(0, 2, (40, 6))
    mask = np.ones(Y.shape, dtype=bool)
    mask[::3, 3:] = False  # some studies lack the second section
    train, group = np.arange(40) < 24, np.array(["a", "b"] * 20)
    runs = []
    for _ in range(2):
        backbone = tiny_backbone()
        start = {k: v.clone() for k, v in backbone.state_dict().items()}
        state, P, history = finetune.train(
            None,
            paths_,
            Y,
            mask,
            train,
            ~train,
            group,
            2,
            7,
            tmp_path,
            world=world,
            backbone=backbone,
            processor=tiny_processor(),
            params=TINY,
        )
        runs.append((state, P, history))
    (s1, p1, h1), (s2, p2, h2) = runs
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert np.array_equal(p1, p2) and p1.shape == (16, 6)
    assert [h["phase"] for h in h1] == ["lp", "ft", "ft"]
    assert any(not torch.equal(start[k], s1[f"backbone.{k}"]) for k in start)
    assert finetune.PARAMS["ft_epochs"] == 10  # the override doesn't leak


@pytest.mark.parametrize("world", [1, 2])
def test_reader_training_is_deterministic(tmp_path, world):
    paths_ = images(tmp_path / "img", 40)
    Y = np.random.default_rng(1).integers(0, 4, (40, 6))
    mask = np.ones(Y.shape, dtype=bool)
    train, group = np.arange(40) < 24, np.array(["a", "b"] * 20)
    condition = np.arange(40) % 3
    runs = []
    for _ in range(2):
        state, P, _ = finetune.train(
            None, paths_, Y, mask, train, ~train, group, 2, 7, tmp_path,
            world=world, backbone=tiny_backbone(), processor=tiny_processor(),
            params=TINY, classes=4, condition=condition, conditions=3,
        )  # fmt: skip
        runs.append((state, P))
    (s1, p1), (s2, p2) = runs
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert np.array_equal(p1, p2)
    assert s1["head.condition"].abs().sum() > 0  # the reader vectors trained


def fake_dataset(cxr_root, name, n, seed):
    """An adapter over n fake studies: 60% train, 40% val; Findings for all,
    Impression for every other study; a lateral for every third study."""
    paths_ = images(cxr_root / name, n, seed)
    laterals = images(cxr_root / name / "lat", n, seed + 100)[::3]
    studies = pd.DataFrame(
        {
            "study_id": [f"{name}_{i}" for i in range(n)],
            "subject_id": [f"{name}_p{i}" for i in range(n)],
            "split": ["train" if i < 0.6 * n else "val" for i in range(n)],
            "image_path": [p.name for p in paths_],
        }
    )
    all_images = pd.DataFrame(
        {
            "study_id": [*studies.study_id, *studies.study_id[::3]],
            "view": ["PA"] * n + ["LATERAL"] * len(laterals),
            "image_path": [*studies.image_path, *(f"lat/{p.name}" for p in laterals)],
        }
    )
    adapter = types.SimpleNamespace(
        SOURCE_FILES=(),
        SPLIT_FILES=(),
        IMAGE_ROOT=name,
        SECTIONS=("findings", "impression"),
        VIEW_RANK={"PA": 0},
        FRONTAL_VIEWS=("PA",),
        LATERAL_VIEWS=("LATERAL",),
        load_studies=lambda root: studies.copy(),
        load_images=lambda root: all_images.copy(),
        classifier_roles=lambda s, splits_dir: s.split,
    )
    rng = np.random.default_rng(seed)
    for section, rows in (("findings", range(n)), ("impression", range(0, n, 2))):
        codes = rng.choice([1.0, 0.0, -1.0, np.nan], size=(len(rows), len(NAMES)))
        labels = pd.DataFrame(codes, columns=NAMES)
        labels.insert(0, "study_id", studies.study_id.iloc[list(rows)].to_numpy())

        def build(out, labels=labels):
            labels.to_csv(out / "labels.csv", index=False)

        artifacts.publish(
            f"labels/{name}/{section}/chexbert14",
            {"stage": "labels", "fake": name},
            {},
            build,
        )
    return adapter


def test_finetune_then_predict(tmp_path, monkeypatch):
    cxr_root = tmp_path / "cxr"
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    for name, n, seed in (("fakea", 30, 0), ("fakeb", 20, 1)):
        monkeypatch.setitem(data.DATASETS, name, fake_dataset(cxr_root, name, n, seed))
    monkeypatch.setattr(weights, "resolve", lambda name: (tmp_path, {"fake": name}))
    monkeypatch.setattr(finetune, "load_backbone", lambda d: tiny_backbone())
    monkeypatch.setattr(finetune, "load_processor", lambda d: tiny_processor())
    for k, v in TINY.items():
        monkeypatch.setitem(finetune.PARAMS, k, v)

    clf = finetune_stage.run(
        "fakea-fakeb", "findings-impression", seed=3, cxr_root=cxr_root
    )
    rel = "fakea-fakeb/raddino.ft-attn-seed3/findings-impression/chexbert14"
    assert clf == paths.SHARED_ROOT / "classifiers" / rel
    manifest = artifacts.read_manifest(clf)
    assert manifest["config"]["datasets"] == ["fakea", "fakeb"]
    assert set(manifest["val_mean_auroc_by_dataset"]) == {"fakea", "fakeb"}
    metrics = pd.read_csv(clf / "metrics.csv")
    assert set(metrics.dataset) == {"fakea", "fakeb"}
    assert set(metrics.section) == {"findings", "impression"}
    saved = torch.load(clf / "model.pt", weights_only=False)
    assert saved["threshold"].shape == (2, len(NAMES))

    out = predict.run("fakeb", rel, cxr_root=cxr_root)
    probs = artifacts.read_table(out / "probs.findings.csv")
    assert len(probs) == 20 and list(probs.columns[2:]) == NAMES
    config = artifacts.read_manifest(out)["config"]
    assert config["threshold"]["findings"] == "max F1 on this dataset's val"
    target = pd.read_csv(out / "metrics.csv")
    assert set(target.section) == {"findings", "impression"}


def test_class_index_and_positive_probs():
    codes = np.array([[1.0, 0.0], [-1.0, np.nan]])
    assert finetune.class_index(codes).tolist() == [[0, 1], [2, 3]]
    # 2 readouts x 4 classes x 3 labels: the positive class is index 0 of each block
    P = np.arange(24.0).reshape(1, 24)
    assert finetune.positive_probs(P, 2, 4).tolist() == [[0, 1, 2, 12, 13, 14]]


def test_multiclass_training_is_deterministic(tmp_path):
    paths_ = images(tmp_path / "img", 40)
    Y = np.random.default_rng(1).integers(0, 4, (40, 6))
    mask = np.ones(Y.shape, dtype=bool)
    mask[::3, 3:] = False
    train, group = np.arange(40) < 24, np.array(["a", "b"] * 20)
    runs = []
    for _ in range(2):
        state, P, _ = finetune.train(
            None,
            paths_,
            Y,
            mask,
            train,
            ~train,
            group,
            2,
            7,
            tmp_path,
            world=1,
            backbone=tiny_backbone(),
            processor=tiny_processor(),
            params=TINY,
            classes=4,
        )
        runs.append((state, P))
    (s1, p1), (s2, p2) = runs
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert np.array_equal(p1, p2) and p1.shape == (16, 2 * 4 * 3)
    # softmax over the classes of each section and label
    assert np.allclose(p1.reshape(16, 2, 4, 3).sum(2), 1)


def test_multiclass_finetune_then_predict(tmp_path, monkeypatch):
    cxr_root = tmp_path / "cxr"
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    for name, n, seed in (("fakea", 30, 0), ("fakeb", 20, 1)):
        monkeypatch.setitem(data.DATASETS, name, fake_dataset(cxr_root, name, n, seed))
    monkeypatch.setattr(weights, "resolve", lambda name: (tmp_path, {"fake": name}))
    monkeypatch.setattr(finetune, "load_backbone", lambda d: tiny_backbone())
    monkeypatch.setattr(finetune, "load_processor", lambda d: tiny_processor())
    for k, v in TINY.items():
        monkeypatch.setitem(finetune.PARAMS, k, v)

    with pytest.raises(SystemExit, match="own class"):
        finetune_stage.run(
            "fakea-fakeb", "findings", seed=3, multiclass=True, uncertain="ones"
        )
    clf = finetune_stage.run(
        "fakea-fakeb", "findings-impression", seed=3, multiclass=True, cxr_root=cxr_root
    )
    rel = "fakea-fakeb/raddino.ft-attn-seed3/findings-impression/chexbert14.mc"
    assert clf == paths.SHARED_ROOT / "classifiers" / rel
    saved = torch.load(clf / "model.pt", weights_only=False)
    assert saved["threshold"] is None and saved["classes"] == list(finetune.MULTICLASS)
    assert "threshold" not in pd.read_csv(clf / "metrics.csv")

    out = predict.run("fakeb", rel, cxr_root=cxr_root)
    pred = artifacts.read_table(out / "predictions.findings.csv")
    assert len(pred) == 20 and list(pred.columns[1:]) == NAMES
    values = pred[NAMES].to_numpy()
    assert set(values[~np.isnan(values)]) <= {1.0, 0.0, -1.0}
    total = sum(
        artifacts.read_table(out / f"probs.findings.{c}.csv")[NAMES].to_numpy()
        for c in finetune.MULTICLASS
    )
    assert np.allclose(total, 1)
    config = artifacts.read_manifest(out)["config"]
    assert config["threshold"] == "none: the most probable class"


def test_reader_index_keeps_frequent_train_readers():
    reader = np.array(["r1", "r1", "r2", None, "r1", "r3", "r2"], dtype=object)
    train = np.array([True, True, True, True, False, True, False])
    with pytest.MonkeyPatch.context() as m:
        m.setattr(finetune_stage, "MIN_READER_STUDIES", 2)
        vocab, index = finetune_stage.reader_index(reader, train)
    assert vocab == ["unknown", "r1"]  # r2 has 1 train study, r3 1
    assert index.tolist() == [1, 1, 0, 0, 1, 0, 0]


def test_reader_vectors_start_at_zero():
    torch.manual_seed(0)
    plain = finetune.attention.Head(3, 32, 2, 64, 0.0, 2)
    torch.manual_seed(0)
    cond = finetune.attention.Head(3, 32, 2, 64, 0.0, 2, conditions=4)
    x = torch.randn(5, 7, 32)
    assert torch.equal(plain(x), cond(x, torch.tensor([0, 1, 2, 3, 1])))


def test_reader_finetune_then_predict(tmp_path, monkeypatch):
    cxr_root = tmp_path / "cxr"
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    adapters = {}
    for name, n, seed in (("fakea", 30, 0), ("fakeb", 20, 1)):
        adapters[name] = fake_dataset(cxr_root, name, n, seed)
        monkeypatch.setitem(data.DATASETS, name, adapters[name])
    # fakea has readers: r0 on 12 of its 18 train studies, r1 on the rest
    ids = [f"fakea_{i}" for i in range(30)]
    reader = pd.Series(["r0" if i % 3 else "r1" for i in range(30)], index=ids)
    adapters["fakea"].load_readers = lambda root: reader
    adapters["fakea"].READER_FILES = ()
    monkeypatch.setattr(finetune_stage, "MIN_READER_STUDIES", 10)
    monkeypatch.setattr(weights, "resolve", lambda name: (tmp_path, {"fake": name}))
    monkeypatch.setattr(finetune, "load_backbone", lambda d: tiny_backbone())
    monkeypatch.setattr(finetune, "load_processor", lambda d: tiny_processor())
    for k, v in TINY.items():
        monkeypatch.setitem(finetune.PARAMS, k, v)

    clf = finetune_stage.run(
        "fakea-fakeb",
        "findings-impression",
        classifier="attn-reader",
        seed=3,
        multiclass=True,
        cxr_root=cxr_root,
    )
    rel = "fakea-fakeb/raddino.ft-attn-reader-seed3/findings-impression/chexbert14.mc"
    assert clf == paths.SHARED_ROOT / "classifiers" / rel
    saved = torch.load(clf / "model.pt", weights_only=False)
    assert saved["readers"] == ["unknown", "r0"]
    assert saved["state_dict"]["head.condition"].shape == (2, 32)
    assert artifacts.read_manifest(clf)["n_readers"] == 1

    for name, known in (("fakea", 20), ("fakeb", 0)):
        out = predict.run(name, rel, cxr_root=cxr_root)
        assert artifacts.read_manifest(out)["known_reader_studies"] == known


def test_study_images_preferred_frontal_then_frontals_then_laterals(monkeypatch):
    rows = [
        ("s1", "LATERAL", "a_lat.png"),
        ("s1", "AP", "b_ap.png"),
        ("s1", "UNKNOWN", "c_unknown.png"),
        ("s1", "PA", "d_pa.png"),  # the preferred frontal
        ("s1", "PA", "e_pa_corrupt.png"),
        ("s1", "PA", "f_pa.png"),
        ("s2", "AP", "s2_0.png"),
        *(("s2", "LATERAL", f"s2_{i}.png") for i in range(1, 6)),
    ]
    frame = pd.DataFrame(rows, columns=["study_id", "view", "image_path"])
    adapter = types.SimpleNamespace(
        VIEW_RANK={"PA": 0, "AP": 1},
        FRONTAL_VIEWS=("PA", "AP"),
        LATERAL_VIEWS=("LATERAL",),
        CORRUPT_IMAGES=frozenset({"e_pa_corrupt.png"}),
        load_images=lambda root: frame,
    )
    monkeypatch.setitem(data.DATASETS, "fake", adapter)
    got = data.study_images("fake", None, ["s2", "s1"], ["s2_0.png", "d_pa.png"])
    assert got == [
        ["s2_0.png", "s2_1.png", "s2_2.png", "s2_3.png"],  # at most MAX_IMAGES
        ["d_pa.png", "f_pa.png", "b_ap.png", "a_lat.png"],
    ]


@pytest.mark.parametrize("world", [1, 2])
def test_multi_image_training_is_deterministic(tmp_path, world):
    paths_ = images(tmp_path / "img", 60)
    rows = [paths_[i : i + 1 + i % 3] for i in range(40)]  # 1 to 3 images each
    Y = np.random.default_rng(1).integers(0, 4, (40, 6))
    mask = np.ones(Y.shape, dtype=bool)
    train, group = np.arange(40) < 24, np.array(["a", "b"] * 20)
    runs = []
    for _ in range(2):
        state, P, _ = finetune.train(
            None, rows, Y, mask, train, ~train, group, 2, 7, tmp_path,
            world=world, backbone=tiny_backbone(), processor=tiny_processor(),
            params=TINY, classes=4,
        )  # fmt: skip
        runs.append((state, P))
    (s1, p1), (s2, p2) = runs
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert np.array_equal(p1, p2) and p1.shape == (16, 2 * 4 * 3)


def test_one_image_study_reads_as_the_single_image_model():
    torch.manual_seed(0)
    model = finetune.build(tiny_backbone(), 3, 2, False, classes=4).eval()
    pixels = torch.randn(3, 3, 56, 56)
    with torch.no_grad():
        alone = model(pixels[:1])
        studies = model(pixels, counts=[1, 2])
    torch.testing.assert_close(studies[:1], alone)


def test_all_images_finetune_then_predict(tmp_path, monkeypatch):
    cxr_root = tmp_path / "cxr"
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    for name, n, seed in (("fakea", 30, 0), ("fakeb", 20, 1)):
        monkeypatch.setitem(data.DATASETS, name, fake_dataset(cxr_root, name, n, seed))
    monkeypatch.setattr(weights, "resolve", lambda name: (tmp_path, {"fake": name}))
    monkeypatch.setattr(finetune, "load_backbone", lambda d: tiny_backbone())
    monkeypatch.setattr(finetune, "load_processor", lambda d: tiny_processor())
    for k, v in TINY.items():
        monkeypatch.setitem(finetune.PARAMS, k, v)

    clf = finetune_stage.run(
        "fakea-fakeb",
        "findings-impression",
        classifier="attn-all-images",
        seed=3,
        multiclass=True,
        cxr_root=cxr_root,
    )
    rel = (
        "fakea-fakeb/raddino.ft-attn-all-images-seed3/findings-impression/chexbert14.mc"
    )
    assert clf == paths.SHARED_ROOT / "classifiers" / rel
    manifest = artifacts.read_manifest(clf)
    assert manifest["images_per_study"] == (30 + 10 + 20 + 7) / 50
    assert manifest["config"]["image"] == finetune_stage.ALL_IMAGES_IMAGE
    out = predict.run("fakeb", rel, cxr_root=cxr_root)
    assert len(artifacts.read_table(out / "predictions.findings.csv")) == 20
