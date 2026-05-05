#!/usr/bin/env python3
"""Normalize EuroRad benchmark input data and mask obvious clinical-history leakage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DIFFICULTY_MAP = {
    "normal": "normal",
    "hard": "hard",
    "extrem hard": "extreme_hard",
    "extreme hard": "extreme_hard",
}

RARITY_MAP = {
    "common": "common",
    "rare": "rare",
    "extrem rare": "extreme_rare",
    "extreme rare": "extreme_rare",
}

REDACTED_CLINICAL_HISTORY: Dict[str, str] = {
    "14261": "A 55-year-old woman with a remote history of treated pulmonary infection presented with fever and lethargy. Lumbar puncture was performed. Cerebrospinal fluid showed pleocytosis, elevated protein, and low glucose; organism-specific microbiology is withheld.",
    "14755": "A 20-year-old female patient with a chronic haemoglobinopathy and recurrent strokes, previous encephalo-duro-arterio-myo-synangiosis (EDAMS) due to bilateral carotid artery obstruction, presented with altered mental status.",
    "15330": "A 16-year-old male patient with a known congenital metabolic disorder presented with a stroke-like event and transient paresis of the left arm. After an initial CT, an MRI was scheduled.",
    "16553": "A 23-year-old woman previously underwent resection of a benign left-pelvic nerve-sheath tumour and re-presented 8 months later with a large fluctuant swelling in the left hemipelvis. Further imaging was arranged with a view to drainage.",
    "17487": "An 18-year-old male with no significant past medical history presented with frontal headache, pyrexia, and generalised tonic-clonic seizures. Examination revealed no focal neurological deficit. Empiric antiviral therapy was started for suspected encephalitis; cerebrospinal fluid showed inflammatory changes, with organism-specific results withheld.",
    "17761": "A 48-year-old female presented with a 5-day history of worsening orthostatic headache with photophobia, neck stiffness, visual disturbances, and a normal neurological examination. She had a 6-month history of milder headaches, worse on coughing, with no significant trauma or surgical history. Further imaging was requested to investigate a pressure-related cause.",
    "18683": "A 49-year-old woman with previously treated thyroid malignancy and known systemic metastatic disease presented three years after diagnosis with blurred vision of the right eye.",
    "19358": "A 33-year-old woman with a previously treated triple-negative breast malignancy and prior simple mastectomy was lost to follow-up and presented 6 months later with a fungating, ulcerating right chest wall mass and right axillary lymphadenopathy.",
}


def norm_key(value: Any) -> str:
    return str(value or "").strip().lower()


def fix_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    redacted_ids: List[str] = []
    normalized_difficulty = 0
    normalized_rarity = 0

    for case in cases:
        old_difficulty = case.get("difficulty")
        new_difficulty = DIFFICULTY_MAP.get(norm_key(old_difficulty), old_difficulty)
        if new_difficulty != old_difficulty:
            case.setdefault("original_difficulty", old_difficulty)
            case["difficulty"] = new_difficulty
            normalized_difficulty += 1

        old_rarity = case.get("rarity")
        new_rarity = RARITY_MAP.get(norm_key(old_rarity), old_rarity)
        if new_rarity != old_rarity:
            case.setdefault("original_rarity", old_rarity)
            case["rarity"] = new_rarity
            normalized_rarity += 1

        cid = str(case.get("case_id") or "")
        if cid in REDACTED_CLINICAL_HISTORY:
            case.setdefault("original_clinical_history", case.get("clinical_history") or "")
            case["clinical_history"] = REDACTED_CLINICAL_HISTORY[cid]
            case["clinical_history_redacted"] = True
            case["clinical_history_redaction_reason"] = "Direct diagnosis/pathogen/procedure leakage was masked; original text preserved."
            redacted_ids.append(cid)

    return {
        "n_cases": len(cases),
        "difficulty_labels_normalized": normalized_difficulty,
        "rarity_labels_normalized": normalized_rarity,
        "clinical_histories_redacted": len(redacted_ids),
        "redacted_case_ids": redacted_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise TypeError(f"Expected list of cases, got {type(cases).__name__}")
    summary = fix_cases(cases)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
