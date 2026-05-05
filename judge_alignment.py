from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from judge import JudgeRunner
from pipeline_main import canonicalize_case, deduplicate_preserve_order, ensure_dir, normalize_key, normalize_space
from run_main import (
    build_provider,
    file_sha256,
    json_dumps,
    make_judge_model_call,
    now_timestamp,
    read_json,
    slugify,
    write_json,
    write_jsonl,
)


FINAL_SCORE_KEYS = ("diagnosis", "localization", "differential_list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-judge an existing EuroRad run and measure LLM-judge alignment against the source LLM judge."
    )
    parser.add_argument("--input-run", required=True, help="Path to a source run directory or benchmark_full.json.")
    parser.add_argument("--out-dir", required=True, help="Output root for alignment logs/results.")
    parser.add_argument("--data-path", default=None, help="Override dataset path. Defaults to source run_config.data_path.")
    parser.add_argument("--image-root", default=None, help="Override image root. Defaults to source run_config.image_root.")
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic", "gemini", "vertex", "qwen"])
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-max-output-tokens", type=int, default=4096)
    parser.add_argument("--judge-structured-output", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--limit", type=int, default=None, help="Only re-judge the first N source case results.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--gemini-thinking-level", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--gemini-api-version", default="v1beta")
    parser.add_argument("--qwen-transport", default="auto", choices=["auto", "responses", "chat"])
    parser.add_argument("--vertex-project", default=None)
    parser.add_argument("--vertex-region", default=None)
    return parser.parse_args()


def resolve_benchmark_full(input_run: Path) -> Path:
    if input_run.is_dir():
        candidate = input_run / "benchmark_full.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No benchmark_full.json found in {input_run}")
    if input_run.name != "benchmark_full.json":
        raise ValueError(f"--input-run must be a run directory or benchmark_full.json, got {input_run}")
    return input_run


def int_score(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def final_score_map(judge_payload: Dict[str, Any]) -> Dict[str, Optional[int]]:
    scores = judge_payload.get("final_scores") or {}
    return {key: int_score((scores.get(key) or {}).get("score")) for key in FINAL_SCORE_KEYS}


def trajectory_label_map(judge_payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in judge_payload.get("trajectory_labels") or []:
        diagnosis = normalize_space(item.get("diagnosis"))
        if not diagnosis:
            continue
        out[normalize_key(diagnosis)] = normalize_space(item.get("label") or "U").upper()[:1] or "U"
    return out


def trajectory_score_map(judge_payload: Dict[str, Any]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    for item in judge_payload.get("trajectory_scores") or []:
        diagnosis = normalize_space(item.get("diagnosis"))
        if not diagnosis:
            continue
        out[normalize_key(diagnosis)] = int_score(item.get("score"))
    return out


def increment_count(counts: Dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def compare_judges(gemini_judge: Dict[str, Any], gpt_judge: Dict[str, Any]) -> Dict[str, Any]:
    final_gemini = final_score_map(gemini_judge)
    final_gpt = final_score_map(gpt_judge)
    final_items: Dict[str, Dict[str, Any]] = {}
    final_matches = 0
    final_total = 0
    final_confusions: Dict[str, Dict[str, int]] = {key: {} for key in FINAL_SCORE_KEYS}
    for key in FINAL_SCORE_KEYS:
        gemini_score = final_gemini.get(key)
        gpt_score = final_gpt.get(key)
        match = gemini_score is not None and gpt_score is not None and gemini_score == gpt_score
        final_items[key] = {"gemini": gemini_score, "gpt": gpt_score, "match": match}
        final_total += 1
        final_matches += int(match)
        increment_count(final_confusions[key], f"{gemini_score}->{gpt_score}")

    labels_gemini = trajectory_label_map(gemini_judge)
    labels_gpt = trajectory_label_map(gpt_judge)
    label_keys = sorted(set(labels_gemini) | set(labels_gpt))
    label_items: List[Dict[str, Any]] = []
    label_matches = 0
    label_confusions: Dict[str, int] = {}
    for key in label_keys:
        gemini_label = labels_gemini.get(key)
        gpt_label = labels_gpt.get(key)
        match = gemini_label is not None and gpt_label is not None and gemini_label == gpt_label
        label_items.append({"diagnosis_key": key, "gemini": gemini_label, "gpt": gpt_label, "match": match})
        label_matches += int(match)
        increment_count(label_confusions, f"{gemini_label}->{gpt_label}")

    scores_gemini = trajectory_score_map(gemini_judge)
    scores_gpt = trajectory_score_map(gpt_judge)
    score_keys = sorted(set(scores_gemini) | set(scores_gpt))
    score_items: List[Dict[str, Any]] = []
    score_matches = 0
    score_abs_error_sum = 0.0
    score_abs_error_count = 0
    score_confusions: Dict[str, int] = {}
    for key in score_keys:
        gemini_score = scores_gemini.get(key)
        gpt_score = scores_gpt.get(key)
        match = gemini_score is not None and gpt_score is not None and gemini_score == gpt_score
        score_items.append({"diagnosis_key": key, "gemini": gemini_score, "gpt": gpt_score, "match": match})
        score_matches += int(match)
        increment_count(score_confusions, f"{gemini_score}->{gpt_score}")
        if gemini_score is not None and gpt_score is not None:
            score_abs_error_sum += abs(float(gemini_score) - float(gpt_score))
            score_abs_error_count += 1

    atomic_total = final_total + len(label_keys) + len(score_keys)
    atomic_matches = final_matches + label_matches + score_matches
    strict = atomic_total > 0 and atomic_matches == atomic_total

    return {
        "final_scores": final_items,
        "trajectory_labels": label_items,
        "trajectory_scores": score_items,
        "counts": {
            "final_total": final_total,
            "final_matches": final_matches,
            "trajectory_label_total": len(label_keys),
            "trajectory_label_matches": label_matches,
            "trajectory_score_total": len(score_keys),
            "trajectory_score_matches": score_matches,
            "atomic_total": atomic_total,
            "atomic_matches": atomic_matches,
        },
        "rates": {
            "final_score_agreement": safe_rate(final_matches, final_total),
            "trajectory_label_agreement": safe_rate(label_matches, len(label_keys)),
            "trajectory_score_agreement": safe_rate(score_matches, len(score_keys)),
            "overall_atomic_agreement": safe_rate(atomic_matches, atomic_total),
            "trajectory_score_mae": safe_rate(score_abs_error_sum, score_abs_error_count),
        },
        "strict_agreement": strict,
        "confusions": {
            "final_scores": final_confusions,
            "trajectory_labels": label_confusions,
            "trajectory_scores": score_confusions,
        },
    }


def safe_rate(num: float, den: int) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)


def aggregate_alignment(case_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        "final_matches": 0,
        "final_total": 0,
        "trajectory_label_matches": 0,
        "trajectory_label_total": 0,
        "trajectory_score_matches": 0,
        "trajectory_score_total": 0,
        "atomic_matches": 0,
        "atomic_total": 0,
        "strict_matches": 0,
        "strict_total": 0,
        "trajectory_score_abs_error_sum": 0.0,
        "trajectory_score_abs_error_count": 0,
    }
    final_confusions: Dict[str, Dict[str, int]] = {key: {} for key in FINAL_SCORE_KEYS}
    label_confusions: Dict[str, int] = {}
    score_confusions: Dict[str, int] = {}

    for record in case_records:
        if record.get("error"):
            continue
        alignment = record.get("alignment") or {}
        counts = alignment.get("counts") or {}
        totals["final_matches"] += int(counts.get("final_matches") or 0)
        totals["final_total"] += int(counts.get("final_total") or 0)
        totals["trajectory_label_matches"] += int(counts.get("trajectory_label_matches") or 0)
        totals["trajectory_label_total"] += int(counts.get("trajectory_label_total") or 0)
        totals["trajectory_score_matches"] += int(counts.get("trajectory_score_matches") or 0)
        totals["trajectory_score_total"] += int(counts.get("trajectory_score_total") or 0)
        totals["atomic_matches"] += int(counts.get("atomic_matches") or 0)
        totals["atomic_total"] += int(counts.get("atomic_total") or 0)
        totals["strict_matches"] += int(bool(alignment.get("strict_agreement")))
        totals["strict_total"] += 1

        for item in alignment.get("trajectory_scores") or []:
            gemini_score = item.get("gemini")
            gpt_score = item.get("gpt")
            if gemini_score is not None and gpt_score is not None:
                totals["trajectory_score_abs_error_sum"] += abs(float(gemini_score) - float(gpt_score))
                totals["trajectory_score_abs_error_count"] += 1

        confusions = alignment.get("confusions") or {}
        for key, bucket in (confusions.get("final_scores") or {}).items():
            final_confusions.setdefault(key, {})
            for confusion, value in bucket.items():
                final_confusions[key][confusion] = final_confusions[key].get(confusion, 0) + int(value)
        for confusion, value in (confusions.get("trajectory_labels") or {}).items():
            label_confusions[confusion] = label_confusions.get(confusion, 0) + int(value)
        for confusion, value in (confusions.get("trajectory_scores") or {}).items():
            score_confusions[confusion] = score_confusions.get(confusion, 0) + int(value)

    return {
        "n_cases": len(case_records),
        "n_success": sum(1 for record in case_records if not record.get("error")),
        "n_error": sum(1 for record in case_records if record.get("error")),
        "counts": totals,
        "rates": {
            "final_score_agreement": safe_rate(totals["final_matches"], totals["final_total"]),
            "trajectory_label_agreement": safe_rate(totals["trajectory_label_matches"], totals["trajectory_label_total"]),
            "trajectory_score_agreement": safe_rate(totals["trajectory_score_matches"], totals["trajectory_score_total"]),
            "overall_atomic_agreement": safe_rate(totals["atomic_matches"], totals["atomic_total"]),
            "case_strict_agreement_rate": safe_rate(totals["strict_matches"], totals["strict_total"]),
            "trajectory_score_mae": safe_rate(
                totals["trajectory_score_abs_error_sum"],
                totals["trajectory_score_abs_error_count"],
            ),
        },
        "confusions": {
            "final_scores": final_confusions,
            "trajectory_labels": dict(sorted(label_confusions.items())),
            "trajectory_scores": dict(sorted(score_confusions.items())),
        },
    }


def case_map_from_data(data_path: Path, image_root: Path, reveal_unit: Optional[str]) -> Dict[str, Any]:
    raw_cases = read_json(data_path)
    if not isinstance(raw_cases, list):
        raise RuntimeError(f"Expected list of cases in {data_path}, got {type(raw_cases)}")
    cases = [canonicalize_case(raw_case, image_root=image_root, reveal_unit=reveal_unit) for raw_case in raw_cases]
    return {str(case.case_id): case for case in cases}


def run() -> int:
    args = parse_args()
    input_full_path = resolve_benchmark_full(Path(args.input_run))
    source_run_dir = input_full_path.parent
    source_full = read_json(input_full_path)
    source_config = source_full.get("run_config") or {}
    source_results = list(source_full.get("case_results") or [])
    if args.limit is not None:
        source_results = source_results[: args.limit]

    data_path = Path(args.data_path or source_config.get("data_path") or "")
    image_root = Path(args.image_root or source_config.get("image_root") or "")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    out_root = Path(args.out_dir)
    run_name = (
        f"judge-alignment-{slugify(source_config.get('judge_model') or 'source-judge')}"
        f"-vs-{slugify(args.judge_model)}-{now_timestamp()}"
    )
    out_dir = out_root / run_name
    ensure_dir(out_dir)

    alignment_config = {
        "source_run_dir": str(source_run_dir),
        "source_benchmark_full": str(input_full_path),
        "source_benchmark_full_sha256": file_sha256(input_full_path),
        "source_provider": source_config.get("provider"),
        "source_target_model": source_config.get("target_model"),
        "source_judge_provider": source_config.get("judge_provider"),
        "source_judge_model": source_config.get("judge_model"),
        "data_path": str(data_path),
        "data_path_sha256": file_sha256(data_path),
        "image_root": str(image_root),
        "reveal_unit": source_config.get("reveal_unit"),
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_structured_output": args.judge_structured_output,
        "judge_max_output_tokens": args.judge_max_output_tokens,
        "timeout": args.timeout,
        "reasoning_effort": args.reasoning_effort,
        "limit": args.limit,
        "n_source_cases_selected": len(source_results),
        "alignment_metric_policy": "exact_match_on_final_scores_trajectory_labels_and_trajectory_scores",
    }
    write_json(out_dir / "alignment_config.json", alignment_config)

    cases_by_id = case_map_from_data(data_path, image_root, source_config.get("reveal_unit"))
    judge_provider = build_provider(
        provider_name=args.judge_provider,
        model_name=args.judge_model,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        timeout=args.timeout,
        max_output_tokens=args.judge_max_output_tokens,
        image_detail="auto",
        reasoning_effort=args.reasoning_effort,
        agent_structured_output="never",
        gemini_thinking_level=args.gemini_thinking_level,
        qwen_transport=args.qwen_transport,
        gemini_api_version=args.gemini_api_version,
        vertex_project=args.vertex_project,
        vertex_region=args.vertex_region,
    )
    judge_runner = JudgeRunner(
        model_call=make_judge_model_call(judge_provider, args.judge_structured_output),
        enable_llm=True,
        enable_rule=False,
        prompt_version="judge_v5_schema_aligned_trajectory_scores",
        rule_version="rule_v8_rubric_extracted_terms",
    )

    case_records: List[Dict[str, Any]] = []
    for index, source_record in enumerate(source_results, start=1):
        case_id = str(source_record.get("case_id"))
        try:
            source_judge = (((source_record.get("judge") or {}).get("by_mode") or {}).get("llm") or {})
            model_payload = source_judge.get("model_payload") or {}
            if not source_judge:
                raise RuntimeError("source record has no judge.by_mode.llm payload")
            if not model_payload:
                raise RuntimeError("source judge has no model_payload")
            case = cases_by_id.get(case_id)
            if case is None:
                raise RuntimeError(f"case_id not found in data_path: {case_id}")

            gpt_result = judge_runner.score_case(
                case=case,
                final_top1_diagnosis=normalize_space(model_payload.get("final_top1_diagnosis")),
                final_differential=list(model_payload.get("final_differential") or []),
                final_location=dict(model_payload.get("final_location") or {}),
                trajectory_unique_diagnoses=deduplicate_preserve_order(model_payload.get("trajectory_unique_diagnoses") or []),
            )
            gpt_judge = ((gpt_result.get("by_mode") or {}).get("llm") or {})
            if not gpt_judge:
                raise RuntimeError(f"GPT judge failed: {gpt_result.get('errors')}")

            alignment = compare_judges(source_judge, gpt_judge)
            record = {
                "case_index": index,
                "case_id": case_id,
                "alignment": alignment,
                "gemini_judge": source_judge,
                "gpt_judge": gpt_judge,
                "gpt_judge_container": gpt_result,
                "error": None,
            }
            print(
                f"[OK] case_id={case_id} "
                f"overall_atomic={alignment['rates']['overall_atomic_agreement']:.4f} "
                f"strict={alignment['strict_agreement']}",
                flush=True,
            )
        except Exception as exc:
            record = {
                "case_index": index,
                "case_id": case_id,
                "alignment": None,
                "gemini_judge": (((source_record.get("judge") or {}).get("by_mode") or {}).get("llm") or {}),
                "gpt_judge": None,
                "gpt_judge_container": None,
                "error": repr(exc),
            }
            print(f"[FAIL] case_id={case_id} error={exc}", file=sys.stderr, flush=True)
        case_records.append(record)

    summary = aggregate_alignment(case_records)
    full_payload = {
        "alignment_config": alignment_config,
        "alignment_summary": summary,
        "alignment_cases": case_records,
    }
    write_json(out_dir / "alignment_summary.json", summary)
    write_jsonl(out_dir / "alignment_cases.jsonl", case_records)
    write_json(out_dir / "alignment_full.json", full_payload)
    print(f"Saved alignment outputs to: {out_dir}", flush=True)
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
