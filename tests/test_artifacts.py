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


def test_started_covers_work_done_before_publish(tmp_path):
    begun = artifacts.now()
    final = artifacts.publish(
        "a", {}, {}, _build("v"), shared_root=tmp_path, started=begun
    )
    m = json.loads((final / "manifest.json").read_text())
    assert m["started"] == begun and m["finished"] >= begun


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


def test_artifact_record_checks_outputs(tmp_path):
    final = artifacts.publish("a", {"k": 1}, {}, _build("v1"), shared_root=tmp_path)
    record = artifacts.artifact_record(final)
    assert record["path"] == str(final) and set(record["outputs"]) == {"data.csv"}
    (final / "data.csv").write_text("v2")
    with pytest.raises(SystemExit, match="CHANGED"):
        artifacts.artifact_record(final)
    with pytest.raises(SystemExit, match="MISSING"):
        artifacts.artifact_record(tmp_path / "nope")


def _chain(tmp_path, upstream_text="v1"):
    """Artifact "up", and "down" built from it (its record in down's inputs)."""
    up = artifacts.publish("up", {"k": 1}, {}, _build(upstream_text),
                           shared_root=tmp_path)  # fmt: skip
    inputs = {"up": artifacts.artifact_record(up)}
    down = artifacts.publish("down", {"k": 1}, inputs, _build("d"),
                             shared_root=tmp_path)  # fmt: skip
    return up, down


def test_upstream_rebuilt_to_the_same_bytes_keeps_downstream(tmp_path):
    up, _ = _chain(tmp_path)
    artifacts.publish("up", {"k": 2}, {}, _build("v1"), shared_root=tmp_path,
                      overwrite=True)  # fmt: skip
    assert artifacts.is_built("down", {"k": 1}, shared_root=tmp_path)
    artifacts.artifact_record(tmp_path / "down")


def test_upstream_with_other_outputs_makes_downstream_stale(tmp_path):
    up, down = _chain(tmp_path)
    artifacts.publish("up", {"k": 1}, {}, _build("v2"), shared_root=tmp_path,
                      overwrite=True)  # fmt: skip
    with pytest.raises(SystemExit, match="STALE: .*\n  - .*up has other outputs"):
        artifacts.is_built("down", {"k": 1}, shared_root=tmp_path)
    with pytest.raises(SystemExit, match="STALE INPUT"):  # nor read downstream
        artifacts.artifact_record(down)
    # overwrite rebuilds it, from the current upstream
    inputs = {"up": artifacts.artifact_record(up)}
    artifacts.publish("down", {"k": 1}, inputs, _build("d"), shared_root=tmp_path,
                      overwrite=True)  # fmt: skip
    assert artifacts.is_built("down", {"k": 1}, shared_root=tmp_path)


def test_upstream_gone_makes_downstream_stale(tmp_path):
    import shutil

    up, _ = _chain(tmp_path)
    shutil.rmtree(up)
    with pytest.raises(SystemExit, match="up is gone"):
        artifacts.is_built("down", {"k": 1}, shared_root=tmp_path)


def test_changed_weights_pin_makes_artifact_stale(tmp_path, monkeypatch):
    sha = "a" * 64
    cfg = {"m": {"repo": "org/repo", "files": {"w.bin": sha}},
           "m2": {"repo": "org/repo", "files": {"other.bin": sha}}}  # fmt: skip
    monkeypatch.setattr(weights, "load_config", lambda: cfg)
    record = {"repo": "org/repo", "dir": "/x",
              "files": {"w.bin": {"sha256": sha, "revision": None}}}  # fmt: skip
    artifacts.publish("a", {}, {"weights": {"m": record}}, _build("v"),
                      shared_root=tmp_path)  # fmt: skip
    assert artifacts.is_built("a", {}, shared_root=tmp_path)
    cfg["m"]["files"]["w.bin"] = "b" * 64  # m2 shares the repo, with other files
    with pytest.raises(SystemExit, match="pinned to other files"):
        artifacts.is_built("a", {}, shared_root=tmp_path)
    cfg.clear()
    with pytest.raises(SystemExit, match="no longer pinned"):
        artifacts.is_built("a", {}, shared_root=tmp_path)


def test_stale_tree_follows_the_chain(tmp_path):
    up, down = _chain(tmp_path)
    inputs = {"down": artifacts.artifact_record(down)}
    artifacts.publish("x/last", {}, inputs, _build("l"), shared_root=tmp_path)
    assert artifacts.stale_tree(tmp_path) == (3, {})
    artifacts.publish("up", {"k": 1}, {}, _build("v2"), shared_root=tmp_path,
                      overwrite=True)  # fmt: skip
    seen, stale = artifacts.stale_tree(tmp_path)
    assert seen == 3
    assert stale[down] == [f"{up} has other outputs now"]
    assert stale[tmp_path / "x" / "last"] == [f"{down} is stale"]
    assert up not in stale
