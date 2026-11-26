from labrag.stages import template


def test_report_lists_labels_and_the_pathologies_in_the_impression():
    assert template.report(["Pleural Effusion", "Support Devices", "PICC Line"]) == (
        "Findings: Pleural effusion. Support devices. PICC line. "
        "Impression: Pleural effusion."
    )


def test_no_finding_or_no_label_is_normal():
    normal = template.NORMAL
    assert template.report([]) == f"Findings: {normal} Impression: {normal}"
    assert template.report(["No Finding", "Endotracheal Tube"]) == (
        f"Findings: {normal} Endotracheal tube. Impression: {normal}"
    )
    assert template.report(["Low Lung Volumes"]) == (
        f"Findings: Low lung volumes. Impression: {normal}"
    )
