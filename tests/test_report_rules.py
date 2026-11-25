"""Rule labelers on report text: device types, incidental findings, label severity."""

import pytest

from labrag.labelers import devices, incidentals, severity


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ET tube tip 4 cm above the carina.", {"Endotracheal Tube": 1.0}),
        ("NG tube enters the stomach.", {"Enteric Tube": 1.0}),
        # a PICC is not also a central line
        ("Left PICC tip in mid SVC.", {"PICC Line": 1.0}),
        ("Right IJ tip in low SVC.", {"Central Venous Catheter": 1.0}),
        ("The chest tube has been removed.", {"Chest Tube": 0.0}),
        ("No endotracheal tube.", {"Endotracheal Tube": 0.0}),
        # a removal in another clause doesn't touch the device that stays
        (
            "The ET tube has been removed, and the NG tube remains.",
            {"Endotracheal Tube": 0.0, "Enteric Tube": 1.0},
        ),
        # named once as removed, once as present: present
        (
            "Left chest tube removed. Right chest tube in place.",
            {"Chest Tube": 1.0},
        ),
        (
            "Median sternotomy wires and mediastinal clips.",
            {"Sternotomy Wires": 1.0, "Surgical Clips": 1.0},
        ),
        (
            "Left chest wall dual lead pacing device.",
            {"Pacemaker or Defibrillator": 1.0},
        ),
        # a pacer's electrode is not a monitoring lead
        (
            "Right-sided pacemaker with the tip of the electrode wire in the apex.",
            {"Pacemaker or Defibrillator": 1.0},
        ),
        ("EKG leads overlie the chest.", {"Monitoring Leads": 1.0}),
        ("The lungs are clear. No pneumothorax.", {}),
        ("", {}),
    ],
)
def test_device_types(text, expected):
    assert devices.label(text) == expected


def test_line_wraps_dont_split_patterns():
    # MIMIC-CXR and CheXpert Plus wrap report lines mid-phrase
    assert devices.label("Right chest\n tube in place.") == {"Chest Tube": 1.0}
    assert incidentals.label("Low lung\n volumes.") == {"Low Lung Volumes": 1.0}
    assert incidentals.label("Elevated  right hemidiaphragm.") == {
        "Elevated Hemidiaphragm": 1.0
    }


def test_device_table_has_every_name():
    table = devices.label_reports(["ET tube in place.", "Clear lungs."])
    assert list(table.columns) == devices.NAMES
    assert table["Endotracheal Tube"].tolist()[0] == 1.0
    assert table.iloc[1].isna().all()


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Degenerative changes of the thoracic spine.",
            {"Degenerative Spine Changes": 1.0},
        ),
        ("Calcification of the aortic knob.", {"Aortic Calcification": 1.0}),
        ("Atherosclerotic calcifications.", {"Aortic Calcification": 1.0}),
        ("The aorta is tortuous.", {"Tortuous or Dilated Aorta": 1.0}),
        ("Lung volumes are low.", {"Low Lung Volumes": 1.0}),
        ("No hiatal hernia.", {"Hiatal Hernia": 0.0}),
        # a negation in another clause doesn't touch the finding
        (
            "No pneumothorax, and the lungs are hyperinflated.",
            {"Emphysema or Hyperinflation": 1.0},
        ),
        ("Elevation of the right hemidiaphragm.", {"Elevated Hemidiaphragm": 1.0}),
        ("Calcified granuloma in the left upper lobe.", {"Calcified Granuloma": 1.0}),
        ("The lungs are clear. No pneumothorax.", {}),
        ("", {}),
    ],
)
def test_incidentals(text, expected):
    assert incidentals.label(text) == expected


def test_incidental_table_has_every_name():
    table = incidentals.label_reports(["Mild scoliosis.", "Clear lungs."])
    assert list(table.columns) == incidentals.NAMES
    assert table["Scoliosis or Kyphosis"].tolist()[0] == 1.0
    assert table.iloc[1].isna().all()


@pytest.mark.parametrize(
    "text, names, expected",
    [
        ("Small left pleural effusion.", ["Pleural Effusion"], {"Pleural Effusion": "small"}),
        ("Heart size is mildly enlarged.", ["Cardiomegaly"], {"Cardiomegaly": "mild"}),
        ("Mild to moderate pulmonary edema.", ["Edema"], {"Edema": "mild to moderate"}),
        ("Small-to-moderate right effusion.", ["Pleural Effusion"], {"Pleural Effusion": "small to moderate"}),
        ("Moderate-sized left effusion.", ["Pleural Effusion"], {"Pleural Effusion": "moderate"}),
        # each label takes the phrase in its own clause
        (
            "There is a small left pleural effusion and mild pulmonary edema.",
            ["Pleural Effusion", "Edema"],
            {"Pleural Effusion": "small", "Edema": "mild"},
        ),
        # a negated mention says nothing about size
        ("No large effusion. Effusion is present.", ["Pleural Effusion"], {}),
        ("Small effusion. Moderate effusion on the right.", ["Pleural Effusion"],
         {"Pleural Effusion": "small, moderate"}),
        # labels without severity words, or not asked for, are left out
        ("Cardiomegaly. Small effusion.", ["Cardiomegaly"], {}),
        ("Small nodule.", ["Lung Lesion"], {}),
    ],
)  # fmt: skip
def test_severity(text, names, expected):
    assert severity.severities(text, names) == expected
