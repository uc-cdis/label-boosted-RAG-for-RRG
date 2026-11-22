"""Stage ``styles``: report-style clusters, a stand-in for the reader where the release
has none.

Writes ``styles/<dataset>/<section>/resid-k<K>/styles.csv``: study_id, style (a cluster
index) for every study with the section's labels. Fit on the ``train`` role only;
every study, val and test included, is assigned from its OWN report, so a target's
style comes from its reference: ``build-prompts --same style`` is a style oracle, like
the label oracle, not a method.

Features (``resid``): the report with sentences that state a finding dropped and
finding, side, severity words and numbers masked; TF-IDF 1-3-grams, SVD, plus log
length and sentence count; then the part linear in the study's labels (one-hot of
CheXbert's four values per label, least squares on train) is removed, so clusters
carry less content. KMeans with K clusters on train.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from labrag import artifacts, paths
from labrag.data import DATASETS, JOINT, add_joint, check_section

LABELER = "chexbert14"
PARAMS = {
    "k": 30,
    "tfidf": {"ngram_range": [1, 3], "min_df": 5, "sublinear_tf": True},
    "svd_dims": 100,
    "structure_scale": 0.1,  # length features weigh about 2 SVD dims
    "kmeans": {"n_init": 3},
    "seed": 0,
}

# Finding and device terms mark a sentence that states one; they and the side and
# severity qualifiers are masked in the rest.
FINDINGS = (
    r"cardiomegal\w*|enlarg\w*|widen\w*|opacit\w*|opacific\w*|consolidat\w*|"
    r"pneumon\w*|infect\w*|aspiration|atelecta\w*|collapse|effusion\w*|"
    r"pneumothora\w*|edema|congesti\w*|vascular\s+engorgement|nodul\w*|mass\w*|"
    r"lesion\w*|tumou?r|metasta\w*|granuloma\w*|fractur\w*|deformit\w*|thicken\w*|"
    r"scar\w*|fibros\w*|emphysem\w*|hyperinflat\w*|calcif\w*|tortuous|tortuosity|"
    r"hernia\w*|tube\w*|catheter\w*|line\w*|pacemaker\w*|pacer|lead\w*|wire\w*|"
    r"clip\w*|stent\w*|port|picc|valve\w*|prosthe\w*|drain\w*|sternotomy|device\w*|"
    r"cabg|icd|defibrillator|infiltrat\w*|abnormal\w*|density|densities|blunting|"
    r"fluid|air\s*space|airspace|interstitial|bronchiectas\w*|cavit\w*|elevat\w*|"
    r"hilar|mediastin\w*|cardiac\s+silhouette|heart\s+size"
)
QUALIFIERS = (
    r"right|left|bilateral\w*|lower|upper|basilar|basal|apical|lobe|lingula\w*|"
    r"base\w*|mild\w*|moderate\w*|severe\w*|small|large|trace|minimal\w*"
)
MASK_RE = re.compile(rf"\b({FINDINGS}|{QUALIFIERS})\b")
FINDING_RE = re.compile(rf"\b({FINDINGS})\b")
NEGATION_RE = re.compile(
    r"\b(no|without|not|negative|free of|clear|resolved|absent|unremarkable|normal|"
    r"within normal limits)\b"
)
SENTENCE_RE = re.compile(r"(?<=[.;?!])\s+")


def style_text(text: str) -> str:
    """Lowercased; a sentence with a finding term and no negation becomes DROPSENT
    (its count stays); finding and qualifier terms become FIND, numbers NUM."""
    out = []
    for sentence in SENTENCE_RE.split(text.lower()):
        if FINDING_RE.search(sentence) and not NEGATION_RE.search(sentence):
            out.append("DROPSENT")
            continue
        out.append(re.sub(r"\d+", "NUM", MASK_RE.sub("FIND", sentence)))
    return " ".join(out)


def features(texts: np.ndarray, codes: np.ndarray, train: np.ndarray) -> np.ndarray:
    """(studies, svd_dims + 2) unit rows: style TF-IDF SVD + structure, minus their
    least-squares fit on the labels' one-hot (fit on ``train`` rows)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    p = PARAMS
    style = np.array([style_text(t) for t in texts])
    vec = TfidfVectorizer(
        ngram_range=tuple(p["tfidf"]["ngram_range"]),
        min_df=p["tfidf"]["min_df"],
        sublinear_tf=p["tfidf"]["sublinear_tf"],
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vec.fit_transform(style[train])
    svd = TruncatedSVD(p["svd_dims"], random_state=p["seed"]).fit(X)
    Z = np.zeros((len(texts), p["svd_dims"]))
    Z[train] = svd.transform(X)
    Z[~train] = svd.transform(vec.transform(style[~train]))
    Z = normalize(Z)
    words = [len(t.split()) for t in texts]
    sentences = [len(SENTENCE_RE.split(t)) for t in texts]
    structure = np.log1p(np.c_[words, sentences])
    mean, std = structure[train].mean(0), structure[train].std(0)
    structure = (structure - mean) / std * p["structure_scale"]
    Z = normalize(np.c_[Z, structure])
    onehot = [codes == v for v in (1.0, -1.0, 0.0)] + [np.isnan(codes)]
    A = np.c_[np.concatenate(onehot, axis=1).astype(float), np.ones(len(codes))]
    coef, *_ = np.linalg.lstsq(A[train], Z[train], rcond=None)
    return normalize(Z - A @ coef)


def run(
    dataset: str,
    section: str,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("styles", dataset, section, joint=True)
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / LABELER
    inputs = {
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
    }
    config = {
        "stage": "styles",
        "dataset": dataset,
        "section": section,
        "labeler": LABELER,
        "features": "resid: style text (finding sentences dropped, finding/side/"
        "severity terms and numbers masked) TF-IDF SVD + log length and sentence "
        "count, minus the least-squares fit on the labels' one-hot",
        "fit": "train role; every study assigned from its own report",
        "params": PARAMS,
    }

    def build(out: Path) -> dict:
        from sklearn.cluster import KMeans

        labels = artifacts.read_table(lab_dir / "labels.csv").set_index("study_id")
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        if section == JOINT:
            studies = add_joint(studies)
        studies = studies.loc[labels.index]
        train = (
            adapter.classifier_roles(studies, paths.SPLITS_DIR) == "train"
        ).to_numpy()
        texts = studies[section].to_numpy()
        Z = features(texts, labels.to_numpy(dtype=float), train)
        km = KMeans(
            PARAMS["k"], n_init=PARAMS["kmeans"]["n_init"], random_state=PARAMS["seed"]
        ).fit(Z[train])
        style = np.empty(len(Z), dtype=int)
        style[train] = km.labels_
        style[~train] = km.predict(Z[~train])
        pd.DataFrame({"study_id": labels.index, "style": style}).to_csv(
            out / "styles.csv", index=False
        )
        sizes = np.bincount(style[train], minlength=PARAMS["k"])
        return {
            "n_studies": len(style),
            "n_train": int(train.sum()),
            "train_cluster_sizes": {"min": int(sizes.min()), "max": int(sizes.max())},
        }

    return artifacts.publish(
        relpath(dataset, section), config, inputs, build, overwrite
    )


def relpath(dataset: str, section: str) -> str:
    """The styles artifact of a dataset's section."""
    return f"styles/{dataset}/{section}/resid-k{PARAMS['k']}"
