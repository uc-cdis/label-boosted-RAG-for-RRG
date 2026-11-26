import pandas as pd

from labrag.stages import baseline


def test_rexgradient_upstream_id_is_the_study_instance_uid():
    to_upstream = baseline.MODELS["cxrmate-2"]["upstream_id"]["rexgradient"]
    assert to_upstream("pGRDN1_aGRDN2_s1.2.826.0.1") == "1.2.826.0.1"


def test_parity_counts_exact_sections_apart_for_sampled_studies(tmp_path):
    published = tmp_path / "published.csv"
    pd.DataFrame(
        {
            "findings": ["Clear.", "Effusion.", "Clear."],
            "impression": ["Normal.", "Effusion.", "Normal."],
            "study_id": ["1", "2", "3"],
        }
    ).to_csv(published, index=False)
    generated = pd.DataFrame(
        {
            "study_id": ["3", "1", "2"],
            "findings": ["Clear.", "Clear.", "Edema."],
            "impression": ["", "Normal.", "Effusion."],
            "n_images": ["2", "6", "1"],
        }
    ).set_index("study_id")
    assert baseline.parity(generated, published, max_images=5) == {
        "published": "published.csv",
        "all_images": {
            "n": 2,
            "findings_equal": 1,
            "impression_equal": 1,
            "both_equal": 0,
        },
        "sampled": {
            "n": 1,
            "findings_equal": 1,
            "impression_equal": 1,
            "both_equal": 1,
        },
    }
