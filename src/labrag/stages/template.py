"""Stage ``template``: reports rendered from the labels a prompt artifact shows, no LLM.

Writes ``generations/<prompt path>/template/<set>/generations.jsonl.gz`` (study_id, set,
report), where generate writes, so evaluate and compare take it like any condition. It
is the floor an LLM has to beat: the small model alone (cf. Pragmatic RRG).

Findings+Impression only, from a version that shows the positive label names. Findings:
one sentence per shown label, in label order ("Pleural effusion."), NORMAL for No
Finding or if no label is shown. Impression: the shown CheXbert-14 pathologies (not No Finding,
Support Devices, device types or incidental findings) the same way, or NORMAL if none.
"""

import gzip
import json

from labrag import artifacts, paths
from labrag.data import JOINT
from labrag.stages.classify import NO_FINDING, NOT_PATHOLOGY

NAME = "template"
NORMAL = "No acute cardiopulmonary abnormality."


def run(prompts: str, target_set: str, overwrite: bool = False):
    rel = prompts.strip("/").removeprefix("prompts/")
    prompt_dir = paths.SHARED_ROOT / "prompts" / rel
    prompt_config = artifacts.read_manifest(prompt_dir)["config"]
    if prompt_config["section"] != JOINT:
        raise SystemExit(f"template: {JOINT} only")
    if "label_categories" in prompt_config or "labels_shown" in prompt_config:
        raise SystemExit("template: needs a prompt that shows the positive labels")
    if not prompt_config["condition"].startswith("labrag"):
        raise SystemExit("template: needs a label condition")
    config = {
        "stage": NAME,
        "prompts": f"prompts/{rel}",
        "set": target_set,
        "findings": f"one sentence per shown label; {NORMAL!r} for {NO_FINDING} or none",
        "impression": f"the shown pathologies, or {NORMAL!r}",
    }

    def build(out):
        with gzip.open(prompt_dir / "prompts.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        rows = [r for r in rows if r["set"] == target_set]
        if not rows:
            raise SystemExit(f"template: no targets in set {target_set!r}")
        records = [
            {"study_id": r["study_id"], "set": r["set"], "report": report(r["labels"])}
            for r in rows
        ]
        with open(out / "generations.jsonl.gz", "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                for r in records:
                    f.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
        return {"n_generations": len(records)}

    return artifacts.publish(
        f"generations/{rel}/{NAME}/{target_set}",
        config,
        {"prompts": artifacts.artifact_record(prompt_dir)},
        build,
        overwrite,
    )


def report(labels: list[str]) -> str:
    """The Findings+Impression text for a study's shown positive labels."""
    findings = [NORMAL if n == NO_FINDING else sentence(n) for n in labels] or [NORMAL]
    impression = [sentence(n) for n in labels if n not in NOT_PATHOLOGY] or [NORMAL]
    return f"Findings: {' '.join(findings)} Impression: {' '.join(impression)}"


def sentence(name: str) -> str:
    """A label name as a sentence: "PICC Line" -> "PICC line."."""
    words = name.split()
    rest = [w if w.isupper() else w.lower() for w in words[1:]]
    return " ".join([words[0], *rest]) + "."
