"""Artifact publishing: manifest, skip, refuse, overwrite; weight checks."""

import json

import pytest

from labrag import artifacts, weights


def _build(text):
    def build(out):
        (out / "data.csv").write_text(text)
        return {"n_rows": 1}

    return build


def test_publish_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("LABRAG_COMMIT", "a" * 40)
    final = artifacts.publish(
        "x/y", {"k": 1}, {"src": {"sha256": "s"}}, _build("v1"), shared_root=tmp_path
    )
    assert final == tmp_path / "x" / "y"
    assert (final / "data.csv").read_text() == "v1"
    m = json.loads((final / "manifest.json").read_text())
    assert m["config"] == {"k": 1} and m["n_rows"] == 1
    assert m["outputs"] == {"data.csv": artifacts.sha256_file(final / "data.csv")}
    assert m["commit"] == {"sha": "a" * 40, "dirty": False}
    assert [p.name for p in (tmp_path / "x").iterdir()] == ["y"]  # no leftovers
    # same mode as a directory made normally (not the 0700 of the staging dir)
    (tmp_path / "plain").mkdir()
    assert final.stat().st_mode == (tmp_path / "plain").stat().st_mode


def test_same_config_skips_different_refuses_overwrite_replaces(tmp_path):
    publish = lambda cfg, text, **kw: artifacts.publish(  # noqa: E731
        "a", cfg, {}, _build(text), shared_root=tmp_path, **kw
    )
    publish({"k": 1}, "v1")
    publish({"k": 1}, "v2")  # skipped: same config
    assert (tmp_path / "a" / "data.csv").read_text() == "v1"
    with pytest.raises(SystemExit, match="different config"):
        publish({"k": 2}, "v2")
    publish({"k": 2}, "v2", overwrite=True)
    assert (tmp_path / "a" / "data.csv").read_text() == "v2"
    assert [p.name for p in tmp_path.iterdir()] == ["a"]


def test_refuses_dir_without_manifest(tmp_path):
    (tmp_path / "a").mkdir()
    with pytest.raises(SystemExit, match="no manifest"):
        artifacts.publish("a", {}, {}, _build("v"), shared_root=tmp_path)


def test_weights_resolve_checks_sha256_and_reads_revision(tmp_path, monkeypatch):
    model_dir = tmp_path / "org" / "repo"
    meta = model_dir / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (model_dir / "w.bin").write_bytes(b"weights")
    (meta / "w.bin.metadata").write_text("c0ffee\netag\n1.0\n")
    good = artifacts.sha256_file(model_dir / "w.bin")
    cfg = {"m": {"repo": "org/repo", "files": {"w.bin": good}}}
    monkeypatch.setattr(weights, "load_config", lambda: cfg)
    _, record = weights.resolve("m", models_root=tmp_path)
    assert record["files"]["w.bin"] == {"sha256": good, "revision": "c0ffee"}

    cfg["m"]["files"]["w.bin"] = "0" * 64
    with pytest.raises(SystemExit, match="sha256"):
        weights.resolve("m", models_root=tmp_path)
    cfg["m"]["files"] = {"missing.bin": good}
    with pytest.raises(SystemExit, match="missing"):
        weights.resolve("m", models_root=tmp_path)
