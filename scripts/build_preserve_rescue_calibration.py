from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.schema import ExperimentRecord
from neg_blindness.validators import validate_record


def rec(
    *,
    rid: str,
    behavior: str,
    semantic_mode: str,
    scope_type: str,
    domain: str,
    prompt_pos: str,
    prompt_neg: str,
    gold_pos: str,
    gold_neg: str | None,
    candidates: list[str],
    distractors: list[str],
    family_id: str,
    template_id: str,
    entity_id: str,
) -> dict[str, Any]:
    return {
        "id": rid,
        "neg_type": "sentential",
        "scope_type": scope_type,
        "semantic_mode": semantic_mode,
        "expected_neg_behavior": behavior,
        "domain": domain,
        "prompt_pos": prompt_pos,
        "prompt_neg": prompt_neg,
        "gold_pos": [gold_pos],
        "gold_neg": [gold_neg] if gold_neg else ([gold_pos] if behavior == "preserve_positive" else []),
        "forbidden_neg": [gold_pos],
        "candidate_pool_neg": candidates,
        "distractors": distractors,
        "template_id": template_id,
        "family_id": family_id,
        "entity_id": entity_id,
        "metadata": {
            "manual_audit": True,
            "targeted_router_calibration": "preserve_rescue",
        },
    }


def build_records() -> list[dict[str, Any]]:
    records = [
        rec(
            rid="preserve_rescue_calib_001",
            behavior="preserve_positive",
            semantic_mode="contrastive_resolution",
            scope_type="out_of_scope",
            domain="instructional",
            prompt_pos="Explain how to configure a printer on a laptop.",
            prompt_neg="Explain how to configure a printer if the laptop is not connected to Wi-Fi.",
            gold_pos="Open system settings, add the printer, install the driver, and print a test page.",
            gold_neg=None,
            candidates=[
                "Restart the router and stop configuring the printer.",
                "Use a phone instead of the laptop.",
                "Delete the printer driver before setup.",
            ],
            distractors=["Discuss monitor brightness settings."],
            family_id="manual_preserve_rescue_printer",
            template_id="manual_open_if_not_preserve_01",
            entity_id="manual_printer_setup",
        ),
        rec(
            rid="preserve_rescue_calib_002",
            behavior="preserve_positive",
            semantic_mode="contrastive_resolution",
            scope_type="out_of_scope",
            domain="instructional",
            prompt_pos="List the steps to prepare a vegetable soup.",
            prompt_neg="List the steps to prepare a vegetable soup if carrots are not available.",
            gold_pos="Wash the vegetables, chop them, simmer them in broth, and season the soup.",
            gold_neg=None,
            candidates=[
                "Serve raw vegetables without cooking them.",
                "Bake a cake instead of making soup.",
                "Only boil plain water.",
            ],
            distractors=["Explain how to sharpen a kitchen knife."],
            family_id="manual_preserve_rescue_soup",
            template_id="manual_open_if_not_preserve_02",
            entity_id="manual_vegetable_soup",
        ),
        rec(
            rid="preserve_rescue_calib_003",
            behavior="preserve_positive",
            semantic_mode="contrastive_resolution",
            scope_type="out_of_scope",
            domain="instructional",
            prompt_pos="Outline how to submit an expense report.",
            prompt_neg="Outline how to submit an expense report if one receipt is not available.",
            gold_pos="Enter the expense details, attach available receipts, submit the report, and wait for approval.",
            gold_neg=None,
            candidates=[
                "Skip the report entirely.",
                "Approve the expense yourself.",
                "Delete all expense entries.",
            ],
            distractors=["Describe how to schedule a team lunch."],
            family_id="manual_preserve_rescue_expense",
            template_id="manual_open_if_not_preserve_03",
            entity_id="manual_expense_report",
        ),
        rec(
            rid="preserve_rescue_calib_004",
            behavior="preserve_positive",
            semantic_mode="contrastive_resolution",
            scope_type="out_of_scope",
            domain="instructional",
            prompt_pos="Provide the procedure for installing a desktop application.",
            prompt_neg="Provide the procedure for installing a desktop application without using the standard installation wizard.",
            gold_pos="Download the installer, accept the license, choose settings, and complete the installation.",
            gold_neg=None,
            candidates=[
                "Uninstall the operating system.",
                "Avoid downloading the application.",
                "Only change the desktop wallpaper.",
            ],
            distractors=["Explain how to clean a keyboard."],
            family_id="manual_preserve_rescue_desktop_app",
            template_id="manual_without_method_preserve_01",
            entity_id="manual_desktop_app",
        ),
        rec(
            rid="preserve_rescue_calib_005",
            behavior="preserve_positive",
            semantic_mode="contrastive_resolution",
            scope_type="out_of_scope",
            domain="instructional",
            prompt_pos="Describe how to create a backup of important files.",
            prompt_neg="Describe how to create a backup of important files if the external drive is not available.",
            gold_pos="Choose the important files, copy them to a backup location, and verify that the backup can be opened.",
            gold_neg=None,
            candidates=[
                "Delete the important files.",
                "Rename the files without copying them.",
                "Ignore backup verification.",
            ],
            distractors=["List common email etiquette rules."],
            family_id="manual_preserve_rescue_backup",
            template_id="manual_open_if_not_preserve_04",
            entity_id="manual_file_backup",
        ),
        rec(
            rid="preserve_rescue_calib_006",
            behavior="preserve_positive",
            semantic_mode="contrastive_resolution",
            scope_type="out_of_scope",
            domain="instructional",
            prompt_pos="Explain how to water a potted plant properly.",
            prompt_neg="Explain how to water a potted plant if the watering can is not available.",
            gold_pos="Check the soil moisture, add water near the base, and let excess water drain.",
            gold_neg=None,
            candidates=[
                "Leave the plant completely dry.",
                "Pour water on electrical devices.",
                "Remove all soil from the pot.",
            ],
            distractors=["Describe how to fold a paper airplane."],
            family_id="manual_preserve_rescue_plant",
            template_id="manual_open_if_not_preserve_05",
            entity_id="manual_potted_plant",
        ),
        rec(
            rid="preserve_rescue_calib_007",
            behavior="select_gold_neg",
            semantic_mode="contrastive_resolution",
            scope_type="in_scope",
            domain="instructional",
            prompt_pos="Which habit helps improve sleep quality before bedtime?",
            prompt_neg="If someone does not avoid screens before bedtime, which habit is most beneficial for better sleep quality?",
            gold_pos="Avoid screens before bedtime.",
            gold_neg="Read a calming book before bedtime.",
            candidates=[
                "Read a calming book before bedtime.",
                "Drink coffee immediately before bed.",
                "Start a loud workout at midnight.",
            ],
            distractors=["Install a printer driver."],
            family_id="manual_select_boundary_sleep",
            template_id="manual_if_not_select_01",
            entity_id="manual_sleep_habit",
        ),
        rec(
            rid="preserve_rescue_calib_008",
            behavior="select_gold_neg",
            semantic_mode="contrastive_resolution",
            scope_type="in_scope",
            domain="instructional",
            prompt_pos="Which exercise primarily builds upper-body strength?",
            prompt_neg="If one does not perform push-ups, which exercise helps build upper-body strength?",
            gold_pos="Push-ups build upper-body strength.",
            gold_neg="Pull-ups help build upper-body strength.",
            candidates=[
                "Pull-ups help build upper-body strength.",
                "Watching television builds upper-body strength.",
                "Sleeping late builds upper-body strength.",
            ],
            distractors=["Boil pasta in salted water."],
            family_id="manual_select_boundary_exercise",
            template_id="manual_if_not_select_02",
            entity_id="manual_upper_body",
        ),
        rec(
            rid="preserve_rescue_calib_009",
            behavior="select_gold_neg",
            semantic_mode="exclusive_choice",
            scope_type="in_scope",
            domain="instructional",
            prompt_pos="Should you save the document before closing the editor?",
            prompt_neg="Should you not save the document before closing the editor?",
            gold_pos="Save the document before closing the editor.",
            gold_neg="Do not save the document before closing the editor.",
            candidates=[
                "Do not save the document before closing the editor.",
                "Print the document twice.",
                "Change the monitor cable.",
            ],
            distractors=["Describe how to water a plant."],
            family_id="manual_select_boundary_document",
            template_id="manual_should_not_select_01",
            entity_id="manual_document_save",
        ),
        rec(
            rid="preserve_rescue_calib_010",
            behavior="select_gold_neg",
            semantic_mode="exclusive_choice",
            scope_type="in_scope",
            domain="instructional",
            prompt_pos="Which step should be done first when assembling a shelf?",
            prompt_neg="Which step should not be done first when assembling a shelf?",
            gold_pos="Read the assembly instructions first.",
            gold_neg="Tighten all screws before reading the instructions.",
            candidates=[
                "Tighten all screws before reading the instructions.",
                "Sort the parts by type.",
                "Check that the package contains all parts.",
            ],
            distractors=["Explain how to submit an expense report."],
            family_id="manual_select_boundary_shelf",
            template_id="manual_which_not_select_01",
            entity_id="manual_shelf_assembly",
        ),
        rec(
            rid="preserve_rescue_calib_011",
            behavior="select_gold_neg",
            semantic_mode="contrastive_resolution",
            scope_type="in_scope",
            domain="instructional",
            prompt_pos="Which cooking method uses dry heat without oil?",
            prompt_neg="Which cooking method does not use dry heat and does not require oil?",
            gold_pos="Baking uses dry heat without oil.",
            gold_neg="Boiling does not use dry heat and does not require oil.",
            candidates=[
                "Boiling does not use dry heat and does not require oil.",
                "Roasting uses dry heat.",
                "Grilling uses dry heat.",
            ],
            distractors=["Create a backup of important files."],
            family_id="manual_select_boundary_cooking",
            template_id="manual_which_not_select_02",
            entity_id="manual_cooking_method",
        ),
        rec(
            rid="preserve_rescue_calib_012",
            behavior="select_gold_neg",
            semantic_mode="contrastive_resolution",
            scope_type="in_scope",
            domain="instructional",
            prompt_pos="What should you do when a smoke alarm sounds?",
            prompt_neg="What should you not do when a smoke alarm sounds?",
            gold_pos="Leave the building safely.",
            gold_neg="Ignore the alarm and stay inside.",
            candidates=[
                "Ignore the alarm and stay inside.",
                "Call emergency services after reaching safety.",
                "Alert nearby people while exiting.",
            ],
            distractors=["Configure a printer on a laptop."],
            family_id="manual_select_boundary_alarm",
            template_id="manual_what_not_select_01",
            entity_id="manual_smoke_alarm",
        ),
    ]
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/processed/splits/router_calibration/preserve_rescue_targeted_calibration.jsonl",
    )
    parser.add_argument(
        "--report",
        default="outputs/preserve_rescue_targeted_calibration_report.json",
    )
    args = parser.parse_args()

    raw_records = build_records()
    records = [ExperimentRecord.from_dict(record) for record in raw_records]
    invalid = []
    for record in records:
        result = validate_record(record)
        if not result.ok:
            invalid.append({"id": record.id, "errors": result.errors})
    if invalid:
        raise SystemExit(json.dumps({"invalid": invalid}, ensure_ascii=False, indent=2))

    out_path = Path(args.output)
    write_jsonl(out_path, [record.as_dict() for record in records])
    report = {
        "output": str(out_path),
        "total": len(records),
        "by_behavior": dict(Counter(record.expected_neg_behavior for record in records)),
        "by_scope": dict(Counter(record.scope_type for record in records)),
        "by_semantic_mode": dict(Counter(record.semantic_mode for record in records)),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
