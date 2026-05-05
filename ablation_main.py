#!/usr/bin/env python3
"""Run ablation settings for the EuroRad-style benchmark.

Implemented settings:
- history-only: clinical history only, no images.
- all-images-at-once: all requestable EuroRad-style evidence units are revealed in one call.
- random-order-reveal: evidence units are passively revealed one at a time in deterministic random order.
- gold-order-reveal: evidence units are passively revealed one at a time by preferred/gold order.
- oracle-findings: official request-based sequential setting, but matched evidence reveals image + metadata + oracle imaging findings.

The official sequential setting remains in run_main.py. This file reuses provider adapters,
canonicalization, judge setup, and metric utilities from run_main.py / pipeline_main.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from judge import JudgeRunner
from pipeline_main import (
    CanonicalCase,
    CaseResult,
    ExamBundle,
    ImagePayload,
    MatchResolution,
    RequestEvent,
    TurnRecord,
    aggregate_results,
    build_dataset_preflight,
    build_metric_notes,
    build_metrics_display,
    canonicalize_case,
    compute_case_metrics,
    compute_metric_status,
    deduplicate_preserve_order,
    normalize_key,
    normalize_match_text,
    normalize_space,
    normalize_turn_output,
    resolve_request_to_exam,
)
from run_main import (
    build_provider,
    file_sha256,
    make_judge_model_call,
    now_timestamp,
    provider_native_temperature_policy,
    read_json,
    slugify,
    summarize_clinical_history_redaction,
    summarize_judge_transport,
    summarize_request_outcomes,
    summarize_target_output_health,
    write_json,
)

PASSIVE_SETTINGS = {"history-only", "all-images-at-once", "random-order-reveal", "gold-order-reveal"}
ACTIVE_ORACLE_SETTINGS = {"oracle-findings"}
ALL_SETTINGS = ["history-only", "all-images-at-once", "random-order-reveal", "gold-order-reveal", "oracle-findings"]

ROUTE_NOT_APPLICABLE_METRICS = {
    "S_ER",
    "B_opt",
    "B_inv",
    "S_order",
    "optional_request_rate",
    "nonessential_request_rate_proxy",
}

PASSIVE_SYSTEM_PROMPT = """You are a rigorous radiology diagnostic agent.
Return STRICT JSON only. Do not use markdown, code fences, commentary, or extra keys.

Required JSON schema:
{
  "action": "stop",
  "requested_examination": "",
  "current_differential": [
    {"diagnosis": "string", "probability": 0.25},
    {"diagnosis": "string", "probability": 0.25},
    {"diagnosis": "string", "probability": 0.25},
    {"diagnosis": "string", "probability": 0.25}
  ],
  "final_location": {
    "laterality": "string",
    "region": "string",
    "substructure": "string"
  }
}

Rules:
- action must be "stop". These ablations do not allow active imaging requests.
- current_differential must contain exactly 4 UNIQUE diagnoses.
- Every probability must be in [0,1], and the 4 probabilities must sum to 1.
- Provide the best current diagnosis and lesion localization using only the evidence given so far.
- No candidate diagnosis list is given.
"""

ORACLE_SYSTEM_PROMPT = """You are a rigorous radiology diagnostic agent.
You are participating in a EuroRad-style hidden evidence benchmark.
Return STRICT JSON only. Do not use markdown, code fences, commentary, or extra keys.

Required JSON schema for every turn:
{
  "action": "request_exam" | "stop",
  "requested_examination": "free-form text; use an empty string when action=stop",
  "current_differential": [
    {"diagnosis": "string", "probability": 0.25},
    {"diagnosis": "string", "probability": 0.25},
    {"diagnosis": "string", "probability": 0.25},
    {"diagnosis": "string", "probability": 0.25}
  ],
  "final_location": {
    "laterality": "string",
    "region": "string",
    "substructure": "string"
  }
}

Rules:
- current_differential must contain exactly 4 UNIQUE diagnoses.
- Every probability must be in [0,1], and the 4 probabilities must sum to 1.
- If action=request_exam, request exactly one next EuroRad-style figure/protocol evidence unit.
- Use modality, anatomic region, and when relevant sequence/acquisition/contrast/timepoint (e.g. T2-weighted MRI spine, DWI/ADC MRI brain, CTA, post-contrast T1, non-contrast CT).
- If action=stop, final_location must describe the lesion location and requested_examination must be empty.
- Each request consumes one step, including unavailable, duplicate, or out-of-scope requests.
- Ambiguous/broad requests may be resolved best-effort by the matcher and are not treated as agent errors when an eligible candidate exists.
- In this oracle-findings ablation, matched evidence reveals images, minimal metadata, and oracle imaging findings for the matched evidence unit.
- No candidate diagnosis list is given.
"""


def format_exam_metadata_line(exam: ExamBundle, image_indices: Sequence[int]) -> str:
    index_text = ", ".join(str(i) for i in image_indices) if image_indices else "none"
    return (
        f"- images [{index_text}] | evidence_id={exam.exam_id} | figure={exam.figure} | "
        f"source_figures={','.join(exam.source_figures or [exam.figure])} | "
        f"modality={exam.modality or 'unspecified'} | acquisition={exam.acquisition or 'unspecified'} | "
        f"view={exam.view or 'unspecified'} | region={exam.region or 'unspecified'} | "
        f"contrast={exam.contrast or 'unspecified'} | time_past={exam.time_past or 'unspecified'}"
    )


def flatten_exam_images(exams: Sequence[ExamBundle]) -> Tuple[List[ImagePayload], List[str]]:
    images: List[ImagePayload] = []
    metadata_lines: List[str] = []
    next_index = 1
    for exam in exams:
        indices = list(range(next_index, next_index + len(exam.image_payloads)))
        images.extend(exam.image_payloads)
        next_index += len(exam.image_payloads)
        metadata_lines.append(format_exam_metadata_line(exam, indices))
    return images, metadata_lines


def build_history_only_prompt(case: CanonicalCase) -> str:
    return f"""Setting: History-only ablation.

Clinical history:
{case.clinical_history or '[none]'}

No imaging is attached and no imaging may be requested in this ablation. Use only the clinical history.

Task:
Return one final stop-turn JSON with your four-item differential diagnosis, probabilities, and final_location.
"""


def build_all_images_prompt(case: CanonicalCase, metadata_lines: Sequence[str]) -> str:
    metadata_block = "\n".join(metadata_lines) if metadata_lines else "- none"
    return f"""Setting: All-images-at-once ablation.

Clinical history:
{case.clinical_history or '[none]'}

All requestable EuroRad-style figure/protocol evidence units for this case are attached in one call. Expert captions, oracle imaging findings, and final answers are not provided. The list below gives only minimal evidence metadata and the 1-indexed order of attached images.

Attached evidence metadata:
{metadata_block}

Task:
Use the clinical history and all attached images to return one final stop-turn JSON with your four-item differential diagnosis, probabilities, and final_location.
"""


def build_passive_initial_prompt(case: CanonicalCase, setting: str, n_reveals: int) -> str:
    return f"""Setting: {setting} ablation.

Clinical history:
{case.clinical_history or '[none]'}

This ablation passively reveals EuroRad-style figure/protocol evidence units. You cannot request imaging. The reveal order is controlled by the benchmark. Total planned evidence reveals: {n_reveals}.

This is the history-only baseline turn before the first evidence unit is revealed.

Task:
Return one stop-turn JSON with your current four-item differential diagnosis, probabilities, and final_location.
"""


def build_passive_reveal_prompt(case: CanonicalCase, setting: str, exam: ExamBundle, step_index: int, total_steps: int) -> str:
    metadata_line = format_exam_metadata_line(exam, list(range(1, 1 + len(exam.image_payloads))))
    order_description = "deterministic random order" if setting == "random-order-reveal" else "preferred/gold order"
    return f"""Setting: {setting} ablation.

Clinical history reminder:
{case.clinical_history or '[none]'}

The benchmark is passively revealing evidence units in {order_description}. You cannot request imaging.

Reveal {step_index} / {total_steps}:
{metadata_line}

New images for this evidence unit are attached to this message. Expert captions, oracle imaging findings, and final answers are not provided.

Task:
Update your current four-item differential diagnosis, probabilities, and final_location using all evidence seen so far. Return STRICT JSON with action=stop.
"""


def stable_random_order(case: CanonicalCase, seed: int) -> List[ExamBundle]:
    exams = list(case.official_exam_pool)
    seed_material = f"{seed}:{case.case_id}:random-order-reveal".encode("utf-8")
    stable_seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    rng = random.Random(stable_seed)
    rng.shuffle(exams)
    return exams


def gold_order(case: CanonicalCase) -> List[ExamBundle]:
    indexed = list(enumerate(case.official_exam_pool))
    return [
        exam for _, exam in sorted(
            indexed,
            key=lambda item: (
                item[1].preferred_order is None,
                item[1].preferred_order if item[1].preferred_order is not None else 10**9,
                item[1].label != "essential",
                item[0],
            ),
        )
    ]


def figure_key_variants(figure: str) -> List[str]:
    text = normalize_space(figure)
    digits = "".join(ch for ch in text if ch.isdigit())
    variants = [text, text.replace("Figure", "Fig"), text.replace("Fig", "Figure")]
    if digits:
        variants.extend([f"Fig{digits}", f"Fig {digits}", f"Figure {digits}"])
    return deduplicate_preserve_order(variants)


def oracle_findings_for_exam(case: CanonicalCase, exam: ExamBundle) -> List[str]:
    raw = case.raw_case or {}
    raw_exams = raw.get("imaging_examinations") or []
    by_figure: Dict[str, Dict[str, Any]] = {}
    for raw_exam in raw_exams:
        fig = normalize_space(raw_exam.get("figure") or "")
        if fig:
            by_figure[fig] = raw_exam
    image_detail = raw.get("image_detail") if isinstance(raw.get("image_detail"), dict) else {}
    findings: List[str] = []
    for source_figure in exam.source_figures or [exam.figure]:
        raw_exam = by_figure.get(normalize_space(source_figure)) or {}
        key_findings = raw_exam.get("key_findings") or []
        if isinstance(key_findings, str):
            key_findings = [key_findings]
        for finding in key_findings:
            finding = normalize_space(finding)
            if finding:
                findings.append(f"{source_figure}: {finding}")
        if not key_findings:
            for variant in figure_key_variants(source_figure):
                detail = image_detail.get(variant)
                if detail:
                    findings.append(f"{source_figure}: {normalize_space(detail)}")
                    break
    return deduplicate_preserve_order(findings)


def format_oracle_findings(case: CanonicalCase, exam: ExamBundle) -> str:
    findings = oracle_findings_for_exam(case, exam)
    if not findings:
        return "- none available for this evidence unit"
    return "\n".join(f"- {finding}" for finding in findings)


def format_request_history(requests: Sequence[RequestEvent]) -> str:
    if not requests:
        return "- none"
    lines: List[str] = []
    for req in requests:
        if req.outcome == "matched":
            suffix = f"MATCHED {req.matched_figure}"
            if req.ambiguity_resolved:
                suffix += f" ({req.resolution_reason})"
            lines.append(f"- request #{req.request_index}: \"{req.request_text}\" -> {suffix}")
        else:
            lines.append(f"- request #{req.request_index}: \"{req.request_text}\" -> INVALID ({req.invalid_reason})")
    return "\n".join(lines)


def build_oracle_initial_prompt(case: CanonicalCase, budget: int) -> str:
    return f"""Setting: Oracle-findings ablation.

Clinical history:
{case.clinical_history or '[none]'}

Official setting reminders:
- The hidden EuroRad-style evidence inventory list is NOT revealed.
- Hidden available evidence units in this case: {len(case.official_exam_pool)}.
- You may request at most {budget} evidence units in total.
- Every request consumes one step, even if it is unavailable, duplicate, ambiguous/best-effort resolved, or out-of-scope.
- When a request matches an evidence unit, images, minimal metadata, and oracle imaging findings for that matched unit will be revealed.

This is the first decision turn. No imaging evidence has been revealed yet.
Task: output your current four-item differential with probabilities, then either request the next evidence unit or stop if you already have enough evidence.
Respond in STRICT JSON only.
"""


def build_oracle_update_prompt(case: CanonicalCase, requests: Sequence[RequestEvent], last_resolution: MatchResolution, budget: int) -> Tuple[str, List[ImagePayload]]:
    history_text = format_request_history(requests)
    images: List[ImagePayload] = []
    if last_resolution.outcome == "matched" and last_resolution.matched_exam is not None:
        exam = last_resolution.matched_exam
        images = exam.image_payloads
        resolution_block = f"""Previous request result:
- MATCHED evidence unit: {exam.figure}
- Source figures: {', '.join(exam.source_figures or [exam.figure])}
- Minimal metadata:
  - modality: {exam.modality or 'unspecified'}
  - acquisition: {exam.acquisition or 'unspecified'}
  - region: {exam.region or 'unspecified'}
  - contrast: {exam.contrast or 'unspecified'}
  - time_past: {exam.time_past or 'unspecified'}

Oracle imaging findings for this matched evidence unit:
{format_oracle_findings(case, exam)}

New images for this matched evidence unit are attached to this message.
"""
    else:
        resolution_block = f"""Previous request result:
- INVALID / UNMATCHED request
- Reason: {last_resolution.reason}
- No new evidence unit was revealed.
"""
    prompt = f"""Clinical history reminder:
{case.clinical_history or '[none]'}

Request budget used: {len(requests)} / {budget}
Remember: every request consumes one step, including unavailable, duplicate, or out-of-scope requests.

Request history so far:
{history_text}

{resolution_block}
Task:
Update your current four-item differential using all evidence seen so far, then choose the next action.
- If you still need evidence, set action=request_exam and request exactly one next EuroRad-style evidence unit.
- If you are ready to conclude, set action=stop and provide final_location.

Respond in STRICT JSON only.
"""
    return prompt, images


def build_oracle_forced_stop_prompt(case: CanonicalCase, requests: Sequence[RequestEvent], last_resolution: Optional[MatchResolution], budget: int) -> Tuple[str, List[ImagePayload]]:
    history_text = format_request_history(requests)
    images: List[ImagePayload] = []
    if last_resolution and last_resolution.outcome == "matched" and last_resolution.matched_exam is not None:
        exam = last_resolution.matched_exam
        images = exam.image_payloads
        resolution_block = f"""The final request reached the budget limit and matched:
- {exam.figure}
- Source figures: {', '.join(exam.source_figures or [exam.figure])}
- modality: {exam.modality or 'unspecified'}
- acquisition: {exam.acquisition or 'unspecified'}
- region: {exam.region or 'unspecified'}
- contrast: {exam.contrast or 'unspecified'}
- time_past: {exam.time_past or 'unspecified'}

Oracle imaging findings for this matched evidence unit:
{format_oracle_findings(case, exam)}

The corresponding images are attached to this message.
"""
    elif last_resolution is not None:
        resolution_block = f"""The final request reached the budget limit but did not match:
- Reason: {last_resolution.reason}
- No new evidence unit was revealed.
"""
    else:
        resolution_block = "No request-resolution update is available."
    prompt = f"""Clinical history reminder:
{case.clinical_history or '[none]'}

You have reached the maximum request budget ({budget} / {budget}).
You MUST stop now.

Request history:
{history_text}

{resolution_block}
Task:
Return a final stop-turn JSON. Set action=stop, provide your final four-item differential with probabilities, and include final_location.
Respond in STRICT JSON only.
"""
    return prompt, images


def mark_route_metrics_not_applicable(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    for name in ROUTE_NOT_APPLICABLE_METRICS:
        if name in out:
            out[name] = None
    out["num_requests"] = 0
    out["num_matched_requests"] = 0
    out["num_invalid_requests"] = 0
    out["num_ambiguous_resolved_requests"] = 0
    out["num_route_matched_requests"] = 0
    out["num_nonroute_matched_requests"] = 0
    out["num_optional_requests"] = 0
    out["ablation_route_metrics_applicable"] = False
    return out


def build_ablation_metric_status(metrics: Dict[str, Any], route_applicable: bool) -> Dict[str, Dict[str, Any]]:
    status = compute_metric_status(metrics)
    if not route_applicable:
        for name in ROUTE_NOT_APPLICABLE_METRICS:
            if name in metrics:
                status[name] = {"defined": False, "reason": "not_applicable_for_nonsequential_ablation"}
    return status


def trajectory_unique_diagnoses_from_turns(turns: Sequence[TurnRecord]) -> List[str]:
    return deduplicate_preserve_order(
        diagnosis
        for turn in turns
        for diagnosis in [item.get("diagnosis") for item in turn.current_differential]
        if isinstance(diagnosis, str) and normalize_space(diagnosis)
    )


class AblationPipeline:
    def __init__(
        self,
        *,
        target_session_factory,
        judge_runner: JudgeRunner,
        setting: str,
        request_budget: int,
        trajectory_horizon: int,
        diagnostic_threshold: float,
        seed: int,
    ) -> None:
        self.target_session_factory = target_session_factory
        self.judge_runner = judge_runner
        self.setting = setting
        self.request_budget = int(request_budget)
        self.trajectory_horizon = int(trajectory_horizon)
        self.diagnostic_threshold = float(diagnostic_threshold)
        self.seed = int(seed)

    def _score_and_pack(self, case: CanonicalCase, turns: List[TurnRecord], requests: List[RequestEvent], *, route_applicable: bool, debug_extra: Optional[Dict[str, Any]] = None) -> CaseResult:
        final_turn = turns[-1]
        final_differential = final_turn.current_differential
        final_top1_diagnosis = final_differential[0]["diagnosis"] if final_differential else ""
        final_location = final_turn.final_location or {"laterality": "", "region": "", "substructure": ""}
        judge_result = self.judge_runner.score_case(
            case=case,
            final_top1_diagnosis=final_top1_diagnosis,
            final_differential=final_differential,
            final_location=final_location,
            trajectory_unique_diagnoses=trajectory_unique_diagnoses_from_turns(turns),
        )
        metrics_by_mode: Dict[str, Dict[str, Any]] = {}
        metric_status_by_mode: Dict[str, Dict[str, Any]] = {}
        metrics_display_by_mode: Dict[str, Dict[str, Any]] = {}
        for mode_name, mode_judge in (judge_result.get("by_mode") or {}).items():
            raw_metrics = compute_case_metrics(
                case,
                turns,
                requests,
                mode_judge,
                request_budget=self.request_budget,
                trajectory_horizon=self.trajectory_horizon,
                diagnostic_threshold=self.diagnostic_threshold,
            )
            if not route_applicable:
                raw_metrics = mark_route_metrics_not_applicable(raw_metrics)
            metrics_by_mode[mode_name] = raw_metrics
            metric_status_by_mode[mode_name] = build_ablation_metric_status(raw_metrics, route_applicable)
            metrics_display_by_mode[mode_name] = build_metrics_display(raw_metrics, metric_status_by_mode[mode_name])
        default_metric_mode = judge_result.get("default_mode") or (next(iter(metrics_by_mode)) if metrics_by_mode else "")
        metrics_raw = metrics_by_mode.get(default_metric_mode, next(iter(metrics_by_mode.values())) if metrics_by_mode else {})
        metrics = metrics_display_by_mode.get(default_metric_mode, next(iter(metrics_display_by_mode.values())) if metrics_display_by_mode else {})
        debug = {
            "setting": self.setting,
            "route_metrics_applicable": route_applicable,
            "official_exam_pool": [exam.minimal_metadata() | {"exam_id": exam.exam_id, "label": exam.label, "preferred_order": exam.preferred_order} for exam in case.official_exam_pool],
            "excluded_exam_pool": [exam.minimal_metadata() | {"exam_id": exam.exam_id, "time_past": exam.time_past} for exam in case.excluded_exam_pool],
            "excluded_reasons_by_exam": case.excluded_reasons_by_exam,
        }
        if debug_extra:
            debug.update(debug_extra)
        return CaseResult(
            case_id=case.case_id,
            case_title=case.case_title,
            section=case.section,
            area_of_interest=case.area_of_interest,
            difficulty=case.difficulty,
            rarity=case.rarity,
            turns=[turn.to_dict() for turn in turns],
            requests=[req.to_dict() for req in requests],
            judge=judge_result,
            metrics=metrics,
            metrics_raw=metrics_raw,
            metrics_by_mode=metrics_display_by_mode,
            metrics_by_mode_raw=metrics_by_mode,
            metric_status_by_mode=metric_status_by_mode,
            default_metric_mode=default_metric_mode,
            debug=debug,
        )

    def _send_stop_turn(self, session, prompt_text: str, images: Sequence[ImagePayload], turn_index: int, prompt_kind: str) -> TurnRecord:
        reply = session.send(prompt_text, images)
        normalized_output, parse_warning = normalize_turn_output(reply.text, forced_stop=True)
        return TurnRecord(
            turn_index=turn_index,
            prompt_kind=prompt_kind,
            prompt_text=prompt_text,
            attached_images=[img.to_dict() for img in images],
            raw_model_text=reply.text,
            parsed_output=normalized_output,
            action="stop",
            requested_examination="",
            current_differential=normalized_output["current_differential"],
            final_location=normalized_output.get("final_location"),
            parse_warning=parse_warning,
            provider_meta=reply.provider_meta or {},
        )

    def run_passive_case(self, case: CanonicalCase) -> CaseResult:
        session = self.target_session_factory(PASSIVE_SYSTEM_PROMPT)
        turns: List[TurnRecord] = []
        requests: List[RequestEvent] = []
        debug_extra: Dict[str, Any] = {}
        if self.setting == "history-only":
            prompt = build_history_only_prompt(case)
            turns.append(self._send_stop_turn(session, prompt, [], 1, "history_only_final"))
        elif self.setting == "all-images-at-once":
            images, metadata_lines = flatten_exam_images(case.official_exam_pool)
            prompt = build_all_images_prompt(case, metadata_lines)
            turns.append(self._send_stop_turn(session, prompt, images, 1, "all_images_at_once_final"))
            debug_extra["all_images_exam_order"] = [exam.exam_id for exam in case.official_exam_pool]
        elif self.setting in {"random-order-reveal", "gold-order-reveal"}:
            full_order = stable_random_order(case, self.seed) if self.setting == "random-order-reveal" else gold_order(case)
            ordered_exams = full_order[: self.request_budget]
            debug_extra["reveal_order"] = [{"exam_id": exam.exam_id, "figure": exam.figure, "preferred_order": exam.preferred_order, "label": exam.label} for exam in ordered_exams]
            debug_extra["full_candidate_order"] = [{"exam_id": exam.exam_id, "figure": exam.figure, "preferred_order": exam.preferred_order, "label": exam.label} for exam in full_order]
            debug_extra["passive_reveal_budget"] = self.request_budget
            turns.append(self._send_stop_turn(session, build_passive_initial_prompt(case, self.setting, len(ordered_exams)), [], 1, f"{self.setting}_history_baseline"))
            for idx, exam in enumerate(ordered_exams, start=1):
                prompt = build_passive_reveal_prompt(case, self.setting, exam, idx, len(ordered_exams))
                turns.append(self._send_stop_turn(session, prompt, exam.image_payloads, len(turns) + 1, f"{self.setting}_reveal"))
        else:
            raise ValueError(f"Unsupported passive setting: {self.setting}")
        return self._score_and_pack(case, turns, requests, route_applicable=False, debug_extra=debug_extra)

    def run_oracle_case(self, case: CanonicalCase) -> CaseResult:
        session = self.target_session_factory(ORACLE_SYSTEM_PROMPT)
        turns: List[TurnRecord] = []
        requests: List[RequestEvent] = []
        attempted_request_texts: set[str] = set()
        revealed_exam_ids: set[str] = set()
        last_resolution: Optional[MatchResolution] = None
        while True:
            if not turns:
                prompt_kind = "oracle_initial"
                prompt_text = build_oracle_initial_prompt(case, self.request_budget)
                images: List[ImagePayload] = []
                forced_stop = False
            elif len(requests) >= self.request_budget:
                prompt_kind = "oracle_forced_stop"
                prompt_text, images = build_oracle_forced_stop_prompt(case, requests, last_resolution, self.request_budget)
                forced_stop = True
            else:
                prompt_kind = "oracle_after_resolution"
                prompt_text, images = build_oracle_update_prompt(case, requests, last_resolution or MatchResolution(outcome="invalid", reason="missing"), self.request_budget)
                forced_stop = False
            reply = session.send(prompt_text, images)
            normalized_output, parse_warning = normalize_turn_output(reply.text, forced_stop=forced_stop)
            turn = TurnRecord(
                turn_index=len(turns) + 1,
                prompt_kind=prompt_kind,
                prompt_text=prompt_text,
                attached_images=[img.to_dict() for img in images],
                raw_model_text=reply.text,
                parsed_output=normalized_output,
                action=normalized_output["action"],
                requested_examination=normalized_output.get("requested_examination"),
                current_differential=normalized_output["current_differential"],
                final_location=normalized_output.get("final_location"),
                parse_warning=parse_warning,
                provider_meta=reply.provider_meta or {},
            )
            turns.append(turn)
            if normalized_output["action"] == "stop":
                break
            request_text = normalized_output.get("requested_examination") or ""
            resolution = resolve_request_to_exam(
                request_text,
                case.official_exam_pool,
                case.excluded_exam_pool,
                revealed_exam_ids,
                attempted_request_texts,
            )
            normalized_request = normalize_match_text(request_text)
            if normalized_request:
                attempted_request_texts.add(normalized_request)
            if resolution.outcome == "matched" and resolution.matched_exam is not None:
                revealed_exam_ids.add(resolution.matched_exam.exam_id)
            request_event = RequestEvent(
                request_index=len(requests) + 1,
                originating_turn_index=turn.turn_index,
                request_text=request_text,
                normalized_request_text=normalize_match_text(request_text),
                outcome=resolution.outcome,
                invalid_reason=None if resolution.outcome == "matched" else resolution.reason,
                resolution_reason=resolution.reason,
                ambiguity_resolved=bool(resolution.outcome == "matched" and "ambiguous" in normalize_key(resolution.reason)),
                matched_exam_id=resolution.matched_exam.exam_id if resolution.matched_exam is not None else None,
                matched_figure=resolution.matched_exam.figure if resolution.matched_exam is not None else None,
                match_score=resolution.match_score,
                candidate_scores=resolution.candidate_scores,
            )
            requests.append(request_event)
            last_resolution = resolution
        return self._score_and_pack(case, turns, requests, route_applicable=True, debug_extra={"oracle_findings_enabled": True})

    def run_case(self, case: CanonicalCase) -> CaseResult:
        if self.setting in PASSIVE_SETTINGS:
            return self.run_passive_case(case)
        if self.setting in ACTIVE_ORACLE_SETTINGS:
            return self.run_oracle_case(case)
        raise ValueError(f"Unsupported ablation setting: {self.setting}")


def compute_judge_mode_agreement(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dx_abs_diffs: List[float] = []
    loc_abs_diffs: List[float] = []
    ddx_abs_diffs: List[float] = []
    trajectory_label_agreements: List[float] = []
    trajectory_score_maes: List[float] = []
    for record in case_results:
        by_mode = record.get("metrics_by_mode_raw") or {}
        if "llm" in by_mode and "rule" in by_mode:
            for key, sink in [("S_dx", dx_abs_diffs), ("S_loc", loc_abs_diffs), ("S_ddx", ddx_abs_diffs)]:
                a = by_mode["llm"].get(key)
                b = by_mode["rule"].get(key)
                if a is not None and b is not None:
                    sink.append(abs(float(a) - float(b)))
        agreement_payload = ((record.get("judge") or {}).get("agreement") or {})
        agreement = agreement_payload.get("trajectory_label_agreement")
        if agreement is not None:
            trajectory_label_agreements.append(float(agreement))
        score_mae = agreement_payload.get("trajectory_score_mae")
        if score_mae is not None:
            trajectory_score_maes.append(float(score_mae))

    def mean_payload(values: Sequence[float], reason: str = "no_comparable_cases") -> Dict[str, Any]:
        return {"value": sum(values) / len(values) if values else 0.0, "defined": bool(values), "n_compared": len(values), "reason": "defined" if values else reason}

    return {
        "S_dx_mae_llm_vs_rule": mean_payload(dx_abs_diffs),
        "S_loc_mae_llm_vs_rule": mean_payload(loc_abs_diffs),
        "S_ddx_mae_llm_vs_rule": mean_payload(ddx_abs_diffs),
        "avg_trajectory_label_agreement": mean_payload(trajectory_label_agreements, "no_cases_with_trajectory_labels_in_both_modes"),
        "avg_trajectory_score_mae": mean_payload(trajectory_score_maes, "no_cases_with_trajectory_scores_in_both_modes"),
    }


def build_ablation_code_version() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = ["ablation_main.py", "pipeline_main.py", "run_main.py", "judge.py"]
    return {
        "schema_version": "ablation_benchmark_v2_random_gold_oracle",
        "settings_implemented": ALL_SETTINGS,
        "official_sequential_entrypoint": "run_main.py",
        "judge_schema_version": "judge_v5_schema_aligned_trajectory_scores",
        "rule_scorer_version": "rule_v8_rubric_extracted_terms",
        "oracle_findings_policy": "matched_evidence_unit_key_findings_only_v1",
        "passive_reveal_policy": "history_baseline_turn_then_up_to_budget_evidence_units_per_turn_v1",
        "temperature_policy": provider_native_temperature_policy(),
        "file_sha256": {name: file_sha256(root / name) for name in files if (root / name).exists()},
    }


def filter_raw_cases(raw_cases: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = raw_cases
    if args.case_id:
        wanted = {str(case_id) for case_id in args.case_id}
        selected = [case for case in selected if str(case.get("case_id")) in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def run_one_setting(args: argparse.Namespace, setting: str, canonical_cases: Sequence[CanonicalCase], target_provider, judge_runner, run_dir: Path) -> Dict[str, Any]:
    pipeline = AblationPipeline(
        target_session_factory=lambda system_prompt: target_provider.create_session(system_prompt),
        judge_runner=judge_runner,
        setting=setting,
        request_budget=args.budget,
        trajectory_horizon=args.trajectory_horizon if args.trajectory_horizon is not None else args.budget + 2,
        diagnostic_threshold=args.diagnostic_threshold,
        seed=args.seed,
    )
    case_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for case in canonical_cases:
        try:
            result = pipeline.run_case(case)
            case_results.append(result.to_dict())
            write_json(run_dir / "benchmark_full.partial.json", build_payload(args, setting, canonical_cases, case_results, failures, is_final=False))
        except Exception as exc:
            failures.append({"case_id": case.case_id, "error": repr(exc)})
            if not args.continue_on_error:
                raise
    payload = build_payload(args, setting, canonical_cases, case_results, failures, is_final=True)
    write_json(run_dir / "benchmark_full.json", payload)
    summary = dict(payload)
    summary.pop("case_results", None)
    write_json(run_dir / "benchmark_summary.json", summary)
    return summary


class _AggregateObj:
    def __init__(self, payload: Dict[str, Any], metric_mode: str) -> None:
        self.__dict__.update(payload)
        self.metrics = (payload.get("metrics_by_mode_raw") or {}).get(metric_mode) or {}
        self.metric_status = (payload.get("metric_status_by_mode") or {}).get(metric_mode) or {}
        self.metrics_display = (payload.get("metrics_by_mode") or {}).get(metric_mode) or {}


def build_payload(args: argparse.Namespace, setting: str, canonical_cases: Sequence[CanonicalCase], case_results: Sequence[Dict[str, Any]], failures: Sequence[Dict[str, Any]], *, is_final: bool) -> Dict[str, Any]:
    enabled_modes = sorted({mode for record in case_results for mode in ((record.get("metrics_by_mode_raw") or {}).keys())})
    aggregate_by_mode: Dict[str, Any] = {}
    table2_rows: Dict[str, Any] = {}
    model_label = args.model_label or f"{args.target_model} / {setting}"
    for mode in enabled_modes:
        aggregate_by_mode[mode] = aggregate_results([_AggregateObj(record, mode) for record in case_results], model_label=model_label, seed=args.seed)
        table2_rows[mode] = aggregate_by_mode[mode].get("table2_row")
    route_note = "Route/request metrics are not applicable for passive non-request ablations." if setting in PASSIVE_SETTINGS else "Route/request metrics are applicable for oracle-findings because it remains request-based sequential."
    return {
        "run_progress": {"is_final": is_final, "completed_cases": len(case_results), "failed_cases": len(failures), "attempted_cases": len(case_results) + len(failures), "total_cases": len(canonical_cases)},
        "run_config": {
            "setting": setting,
            "data_path": str(args.data_path),
            "data_path_sha256": file_sha256(Path(args.data_path)),
            "input_data_name": Path(args.data_path).name,
            "image_root": str(args.image_root),
            "reveal_unit": args.reveal_unit,
            "clinical_history_redaction": summarize_clinical_history_redaction([case.raw_case for case in canonical_cases]),
            "provider": args.provider,
            "target_model": args.target_model,
            "judge_provider": args.judge_provider if args.judge_modes in {"both", "llm"} else None,
            "judge_model": args.judge_model if args.judge_modes in {"both", "llm"} else None,
            "judge_modes": args.judge_modes,
            "judge_structured_output": args.judge_structured_output if args.judge_modes in {"both", "llm"} else None,
            "budget": args.budget,
            "trajectory_horizon": args.trajectory_horizon if args.trajectory_horizon is not None else args.budget + 2,
            "trajectory_horizon_policy": "explicit" if args.trajectory_horizon is not None else "default_budget_plus_2",
            "diagnostic_threshold": args.diagnostic_threshold,
            "limit": args.limit,
            "case_id": args.case_id,
            "seed": args.seed,
            "temperature_policy": provider_native_temperature_policy(),
            "max_output_tokens": args.max_output_tokens,
            "judge_max_output_tokens": args.judge_max_output_tokens,
            "timeout": args.timeout,
            "image_detail": args.image_detail,
            "reasoning_effort": args.reasoning_effort,
            "agent_structured_output": args.agent_structured_output,
            "gemini_thinking_level": args.gemini_thinking_level,
            "qwen_transport": args.qwen_transport,
            "gemini_api_version": args.gemini_api_version,
        },
        "run_health": {"has_blocking_preflight_error": False, "preflight_errors": [], "preflight_warnings": []},
        "dataset_preflight": build_dataset_preflight(canonical_cases),
        "failures": list(failures),
        "enabled_scoring_modes": enabled_modes,
        "default_scoring_mode": "llm" if "llm" in enabled_modes else (enabled_modes[0] if enabled_modes else None),
        "table2_rows": table2_rows,
        "aggregate_by_mode": aggregate_by_mode,
        "judge_mode_agreement": compute_judge_mode_agreement(case_results),
        "judge_transport": summarize_judge_transport(case_results),
        "target_output_health": summarize_target_output_health(case_results),
        "request_outcome_counts": summarize_request_outcomes(case_results),
        "metric_notes": build_metric_notes() | {"ablation_note": route_note},
        "code_version": build_ablation_code_version(),
        "case_results": list(case_results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EuroRad-style benchmark ablation settings.")
    parser.add_argument("--setting", default="all", choices=["all", "both"] + ALL_SETTINGS, help="Ablation setting to run. 'both' is backward-compatible shorthand for history-only + all-images-at-once; 'all' runs all implemented ablations.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini", "vertex", "qwen"])
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--judge-provider", default="vertex", choices=["openai", "anthropic", "gemini", "vertex", "qwen"])
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-modes", default="both", choices=["both", "llm", "rule"])
    parser.add_argument("--judge-structured-output", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--budget", type=int, default=6, help="Request budget for oracle-findings and reference trajectory horizon for passive ablations.")
    parser.add_argument("--reveal-unit", default="eurorad", choices=["eurorad", "figure"])
    parser.add_argument("--trajectory-horizon", type=int, default=None)
    parser.add_argument("--diagnostic-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None, help="Deprecated; accepted for compatibility but ignored. Provider defaults are always used.")
    parser.add_argument("--judge-temperature", type=float, default=None, help="Deprecated; accepted for compatibility but ignored. Provider defaults are always used.")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--image-detail", default="auto", choices=["low", "high", "auto", "original"])
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--agent-structured-output", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--gemini-thinking-level", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--qwen-transport", default="auto", choices=["auto", "responses", "chat"])
    parser.add_argument("--gemini-api-version", default="v1beta")
    parser.add_argument("--vertex-project", default=None)
    parser.add_argument("--vertex-region", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-preflight-errors", action="store_true")
    parser.add_argument("--model-label", default=None)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    args.data_path = Path(args.data_path)
    args.image_root = Path(args.image_root)
    args.out_dir = Path(args.out_dir)
    raw_cases = read_json(args.data_path)
    if not isinstance(raw_cases, list):
        raise RuntimeError(f"Expected a list of cases in {args.data_path}, got {type(raw_cases)}")
    raw_cases = filter_raw_cases(raw_cases, args)
    canonical_cases = [canonicalize_case(raw_case, image_root=args.image_root, reveal_unit=args.reveal_unit) for raw_case in raw_cases]
    dataset_preflight = build_dataset_preflight(canonical_cases)
    preflight_errors = [issue for issue in dataset_preflight.get("issues", []) if issue.get("severity") == "error"]
    if preflight_errors and not args.allow_preflight_errors:
        raise RuntimeError(f"Blocking dataset preflight errors: {preflight_errors}")

    target_provider = build_provider(
        provider_name=args.provider,
        model_name=args.target_model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_output_tokens=args.max_output_tokens,
        image_detail=args.image_detail,
        reasoning_effort=args.reasoning_effort,
        agent_structured_output=args.agent_structured_output,
        gemini_thinking_level=args.gemini_thinking_level,
        qwen_transport=args.qwen_transport,
        gemini_api_version=args.gemini_api_version,
        vertex_project=args.vertex_project,
        vertex_region=args.vertex_region,
    )
    enable_llm_judge = args.judge_modes in {"both", "llm"}
    enable_rule_judge = args.judge_modes in {"both", "rule"}
    judge_provider = None
    if enable_llm_judge:
        judge_provider = build_provider(
            provider_name=args.judge_provider,
            model_name=args.judge_model,
            api_key=args.judge_api_key or args.api_key,
            base_url=args.judge_base_url or args.base_url,
            timeout=args.timeout,
            max_output_tokens=args.judge_max_output_tokens,
            image_detail=args.image_detail,
            reasoning_effort=args.reasoning_effort,
            agent_structured_output="never",
            gemini_thinking_level=args.gemini_thinking_level,
            qwen_transport=args.qwen_transport,
            gemini_api_version=args.gemini_api_version,
            vertex_project=args.vertex_project,
            vertex_region=args.vertex_region,
        )
    judge_runner = JudgeRunner(
        model_call=make_judge_model_call(judge_provider, args.judge_structured_output) if judge_provider is not None else None,
        enable_llm=enable_llm_judge,
        enable_rule=enable_rule_judge,
        prompt_version="judge_v5_schema_aligned_trajectory_scores",
        rule_version="rule_v8_rubric_extracted_terms",
    )

    if args.setting == "all":
        settings = ALL_SETTINGS
    elif args.setting == "both":
        settings = ["history-only", "all-images-at-once"]
    else:
        settings = [args.setting]
    parent = args.out_dir / f"ablation-{slugify(args.provider)}-{slugify(args.target_model)}-{now_timestamp()}"
    parent.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Any] = {}
    for setting in settings:
        setting_dir = parent / setting
        setting_dir.mkdir(parents=True, exist_ok=True)
        summaries[setting] = run_one_setting(args, setting, canonical_cases, target_provider, judge_runner, setting_dir)
    write_json(parent / "ablation_summary.json", {"settings": settings, "summaries": summaries})
    print(f"[DONE] Ablation outputs saved to: {parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
