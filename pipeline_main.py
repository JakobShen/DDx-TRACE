
from __future__ import annotations

import json
import math
import mimetypes
import os
import random
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


# =========================
# Generic helpers
# =========================

def normalize_space(text: Optional[Any]) -> str:
    if text is None:
        raw = ""
    elif isinstance(text, str):
        raw = text
    else:
        raw = str(text)
    return re.sub(r"\s+", " ", raw.strip())


def normalize_key(text: Optional[str]) -> str:
    text = normalize_space(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    return text


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            if math.isnan(float(value)) or math.isinf(float(value)):
                return default
            return float(value)
        text = str(value).strip().replace("%", "")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def renormalize_probabilities(values: Sequence[float]) -> List[float]:
    cleaned = [max(0.0, float(v)) for v in values]
    total = sum(cleaned)
    if not cleaned:
        return []
    if total <= 0:
        return [1.0 / len(cleaned) for _ in cleaned]
    return [v / total for v in cleaned]


def extract_json_from_text(text: str) -> Optional[Any]:
    """Best-effort JSON extractor for model outputs."""
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None

    # direct parse
    for candidate in [raw]:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # strip fenced blocks
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # locate the largest {...} or [...] span
    spans: List[str] = []
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        starts = [m.start() for m in re.finditer(re.escape(start_char), raw)]
        ends = [m.start() for m in re.finditer(re.escape(end_char), raw)]
        if starts and ends:
            for s in starts[:8]:
                for e in reversed(ends[-8:]):
                    if e > s:
                        spans.append(raw[s:e + 1])
                        break
    spans = sorted(spans, key=len, reverse=True)
    for candidate in spans:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Python-literal fallback
    try:
        import ast

        candidate = raw
        candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.I)
        candidate = re.sub(r"\btrue\b", "True", candidate, flags=re.I)
        candidate = re.sub(r"\bfalse\b", "False", candidate, flags=re.I)
        obj = ast.literal_eval(candidate)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    return None


def split_candidates(raw: Optional[str]) -> List[str]:
    text = normalize_space(raw)
    if not text:
        return []
    parts = re.split(r"[,\n;]+", text)
    out: List[str] = []
    seen: set[str] = set()
    for part in parts:
        piece = normalize_space(part)
        if not piece:
            continue
        key = normalize_key(piece)
        if key not in seen:
            out.append(piece)
            seen.add(key)
    return out


def top_level_strings(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [normalize_space(x) for x in obj if isinstance(x, str) and normalize_space(x)]
    if isinstance(obj, dict):
        out: List[str] = []
        for _, value in obj.items():
            if isinstance(value, str) and normalize_space(value):
                out.append(normalize_space(value))
        return out
    return []


def detect_mime_type(path_or_url: str) -> str:
    mime, _ = mimetypes.guess_type(path_or_url)
    return mime or "image/jpeg"


# =========================
# Provider-facing payloads
# =========================

@dataclass
class ImagePayload:
    label: str
    path: Optional[str] = None
    url: Optional[str] = None
    mime_type: str = "image/jpeg"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "url": self.url,
            "mime_type": self.mime_type,
        }


@dataclass
class ModelReply:
    text: str
    raw: Dict[str, Any] = field(default_factory=dict)
    provider_meta: Dict[str, Any] = field(default_factory=dict)


class SessionProtocol(Protocol):
    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        ...


# =========================
# Benchmark-domain schema
# =========================

@dataclass
class ExamBundle:
    exam_id: str
    figure: str
    preferred_order: Optional[int]
    label: str  # essential | optional
    modality: str
    acquisition: str
    view: str
    region: str
    contrast: str
    time_past: Optional[str]
    image_payloads: List[ImagePayload]
    matcher_summary: str
    matcher_tokens: set[str]
    source_figures: List[str] = field(default_factory=list)
    source_exam_count: int = 1
    route_evaluable: bool = True
    evidence_modality_class: str = "radiology"
    time_past_raw: Optional[str] = None
    time_past_interpretable: bool = True

    def minimal_metadata(self) -> Dict[str, Any]:
        return {
            "figure": self.figure,
            "source_figures": list(self.source_figures or [self.figure]),
            "modality": self.modality or "unspecified",
            "acquisition": self.acquisition or "unspecified",
            "view": self.view or "unspecified",
            "region": self.region or "unspecified",
            "contrast": self.contrast or "unspecified",
            "time_past": self.time_past or "unspecified",
            "time_past_raw": self.time_past_raw,
            "time_past_interpretable": self.time_past_interpretable,
            "route_evaluable": self.route_evaluable,
            "evidence_modality_class": self.evidence_modality_class,
        }


@dataclass
class CanonicalCase:
    case_id: str
    case_title: str
    section: str
    area_of_interest: str
    question: str
    clinical_history: str
    final_diagnosis: str
    reference_ddx_options: List[str]
    diagnosis_rubric: Dict[str, Any]
    localization_rubric: Dict[str, Any]
    difficulty: str
    rarity: str
    official_exam_pool: List[ExamBundle]
    excluded_exam_pool: List[ExamBundle]
    gold_labels_by_exam: Dict[str, str]
    gold_order_by_exam: Dict[str, int]
    excluded_reasons_by_exam: Dict[str, str]
    raw_case: Dict[str, Any] = field(repr=False)

    def get_exam(self, exam_id: str) -> Optional[ExamBundle]:
        for exam in self.official_exam_pool:
            if exam.exam_id == exam_id:
                return exam
        return None


@dataclass
class RequestEvent:
    request_index: int
    originating_turn_index: int
    request_text: str
    normalized_request_text: str
    outcome: str  # matched | invalid
    invalid_reason: Optional[str] = None
    resolution_reason: Optional[str] = None
    ambiguity_resolved: bool = False
    matched_exam_id: Optional[str] = None
    matched_figure: Optional[str] = None
    match_score: Optional[float] = None
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnRecord:
    turn_index: int
    prompt_kind: str  # initial | after_resolution | forced_stop
    prompt_text: str
    attached_images: List[Dict[str, Any]]
    raw_model_text: str
    parsed_output: Dict[str, Any]
    action: str
    requested_examination: Optional[str]
    current_differential: List[Dict[str, Any]]
    final_location: Optional[Dict[str, str]]
    parse_warning: Optional[str] = None
    provider_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class MatchResolution:
    outcome: str  # matched | invalid
    reason: str
    matched_exam: Optional[ExamBundle] = None
    match_score: Optional[float] = None
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class CaseResult:
    case_id: str
    case_title: str
    section: str
    area_of_interest: str
    difficulty: str
    rarity: str
    turns: List[Dict[str, Any]]
    requests: List[Dict[str, Any]]
    judge: Dict[str, Any]
    metrics: Dict[str, Any]
    metrics_raw: Dict[str, Any]
    metrics_by_mode: Dict[str, Dict[str, Any]]
    metrics_by_mode_raw: Dict[str, Dict[str, Any]]
    metric_status_by_mode: Dict[str, Dict[str, Any]]
    default_metric_mode: str
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# Dataset adaptation
# =========================

NULL_STRINGS = {"", "null", "none", "na", "n/a", "nan"}
DEFAULT_REVEAL_UNIT = os.environ.get("EURORAD_REVEAL_UNIT", "eurorad")

FOLLOWUP_TEXT_RE = re.compile(
    r"\b("
    r"follow[- ]?up|"
    r"post[- ]?treatment|post[- ]?therapy|post[- ]?operative|postoperative|post[- ]?op|"
    r"after treatment|after therapy|after surgery|"
    r"pre[- ]?and[- ]?post|pre and post|"
    r"comparison|control study|delayed|later|prior|previous|past|months?|years?|weeks?|days?"
    r")\b",
    flags=re.I,
)
PRIOR_TEXT_RE = re.compile(r"\b(prior|previous|past)\b", flags=re.I)

def is_nullish_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return normalize_key(value) in NULL_STRINGS
    return False


def time_past_number(raw_time_past: Any) -> Optional[float]:
    if is_nullish_text(raw_time_past) or isinstance(raw_time_past, bool):
        return None
    if isinstance(raw_time_past, (int, float)):
        value = float(raw_time_past)
        return value if math.isfinite(value) else None
    raw = normalize_space(raw_time_past)
    if not re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?", raw):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def is_prior_time_past(raw_time_past: Any) -> bool:
    value = time_past_number(raw_time_past)
    return value is not None and value > 0


def is_future_time_past(raw_time_past: Any) -> bool:
    value = time_past_number(raw_time_past)
    return value is not None and value < 0


def parse_mcq_options(raw_case: Dict[str, Any]) -> List[str]:
    mcq = (raw_case.get("item") or {}).get("mcq") or {}
    options = mcq.get("options")
    candidates = top_level_strings(options)
    cleaned: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if is_nullish_text(candidate):
            continue
        key = normalize_key(candidate)
        if key not in seen:
            cleaned.append(candidate)
            seen.add(key)
    if cleaned:
        return cleaned

    ddx = split_candidates(raw_case.get("differential_diagnosis"))
    return ddx

def resolve_local_image_path(image_root: Optional[Path], rel_path: Optional[str]) -> Optional[Path]:
    """Resolve a relative image path robustly.

    Supports both:
    - --image-root pointing at the project root, where rel_path is like images/12789/f1_a.jpg
    - --image-root pointing directly at the images directory, where the same rel_path should resolve to 12789/f1_a.jpg
    """
    if not image_root or not rel_path:
        return None
    rel_norm = str(rel_path).replace("\\", "/").strip().lstrip("/")
    if not rel_norm:
        return None
    rel = Path(rel_norm)
    candidates: List[Path] = []
    candidates.append(image_root / rel)

    parts = rel.parts
    if parts:
        first = parts[0].lower()
        root_name = image_root.name.lower()
        if first == root_name and len(parts) > 1:
            candidates.append(image_root.joinpath(*parts[1:]))
        if first != "images":
            candidates.append(image_root / "images" / rel)
        else:
            candidates.append(image_root.parent / rel)
            if len(parts) > 1:
                candidates.append(image_root.joinpath(*parts[1:]))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def build_root_image_index(raw_case: Dict[str, Any], image_root: Optional[Path]) -> Dict[str, List[ImagePayload]]:
    """Build a figure-index from root-level image entries using LOCAL files only.

    The official benchmark is images-only and is expected to run with a real local --image-root.
    We therefore do not use remote URL fallback inside the main pipeline.
    """
    index: Dict[str, List[ImagePayload]] = {}
    for entry in raw_case.get("images") or []:
        name = normalize_space(entry.get("name"))
        caption = normalize_space(entry.get("caption"))
        figure_match = re.search(r"figure\s*(\d+)", name, flags=re.I) or re.search(r"figure\s*(\d+)", caption, flags=re.I)
        if not figure_match:
            continue
        figure = f"Figure {int(figure_match.group(1))}"
        image_path = entry.get("path")
        candidate = resolve_local_image_path(image_root, image_path)
        if candidate is None:
            continue
        payload = ImagePayload(
            label=name or figure,
            path=str(candidate),
            url=None,
            mime_type=detect_mime_type(str(candidate)),
        )
        index.setdefault(figure, []).append(payload)
    return index


def standardize_exam_label(raw_label: Optional[str]) -> str:
    label = normalize_key(raw_label)
    if label == "essential":
        return "essential"
    return "optional"


def is_followup_like(exam: Dict[str, Any]) -> bool:
    meta = exam.get("meta_info") or {}
    if is_future_time_past(meta.get("time_past")):
        return True
    if is_prior_time_past(meta.get("time_past")):
        return False
    text = " ".join(
        normalize_space(x)
        for x in [
            meta.get("time_past"),
            meta.get("acquisition"),
            meta.get("view"),
            meta.get("imaged_region"),
            exam.get("image_caption"),
        ]
        if not is_nullish_text(x)
    )
    return bool(FOLLOWUP_TEXT_RE.search(text))


def is_exam_bundle_followup_like(exam: ExamBundle) -> bool:
    time_value = exam.time_past_raw if exam.time_past_raw is not None else exam.time_past
    if is_future_time_past(time_value):
        return True
    if is_prior_time_past(time_value):
        return False
    text = " ".join(
        normalize_space(x)
        for x in [exam.time_past, exam.acquisition, exam.view, exam.region, exam.matcher_summary]
        if not is_nullish_text(x)
    )
    return bool(FOLLOWUP_TEXT_RE.search(text))


def is_exam_bundle_prior_timepoint(exam: ExamBundle) -> bool:
    time_value = exam.time_past_raw if exam.time_past_raw is not None else exam.time_past
    return is_prior_time_past(time_value)


def is_exam_bundle_future_timepoint(exam: ExamBundle) -> bool:
    time_value = exam.time_past_raw if exam.time_past_raw is not None else exam.time_past
    return is_future_time_past(time_value)


def normalize_time_past(raw_time_past: Any) -> Tuple[Optional[str], Optional[str], bool]:
    if is_nullish_text(raw_time_past):
        return None, None, True
    raw = normalize_space(raw_time_past)
    if time_past_number(raw_time_past) is not None:
        return raw, raw, True
    return raw, raw, True


def classify_evidence_modality(modality: str, acquisition: str = "") -> Tuple[str, bool]:
    text = normalize_match_text(" ".join([modality, acquisition])) if "normalize_match_text" in globals() else normalize_key(" ".join([modality, acquisition]))
    non_route = {"histopathology", "pathology", "microscopy", "photo", "clinical", "illustration", "drawing", "diagram"}
    if any(tok in text.split() for tok in non_route) or any(tok in text for tok in ["histopathology", "microscopy", "illustration"]):
        return "non_imaging", False
    return "radiology", True


def resolve_exam_images(
    raw_case: Dict[str, Any],
    exam: Dict[str, Any],
    image_root: Optional[Path],
    root_image_index: Dict[str, List[ImagePayload]],
) -> List[ImagePayload]:
    payloads: List[ImagePayload] = []
    for rel_path in exam.get("image_paths") or []:
        rel_path = str(rel_path)
        candidate = resolve_local_image_path(image_root, rel_path)
        if candidate is None:
            continue
        payloads.append(
            ImagePayload(
                label=f"{exam.get('figure')}::{Path(rel_path).name}",
                path=str(candidate),
                url=None,
                mime_type=detect_mime_type(str(candidate)),
            )
        )

    if payloads:
        return payloads

    figure = normalize_space(exam.get("figure"))
    if figure and figure in root_image_index:
        return [payload for payload in root_image_index[figure] if payload.path]

    return []


def _join_unique(values: Iterable[Any], *, sep: str = "; ") -> str:
    return sep.join(deduplicate_preserve_order(normalize_space(v) for v in values if normalize_space(v)))


def _canonical_study_modality(modality: str, acquisition: str) -> str:
    text = normalize_match_text(" ".join([modality, acquisition]))
    tokens = set(text.split())
    if "dsa" in tokens or "catheter" in tokens:
        return "DSA"
    if "cta" in tokens or ({"ct", "angiography"} <= tokens) or ({"ct", "angiogram"} <= tokens):
        return "CTA"
    if "mra" in tokens:
        return "MRA"
    if "mrv" in tokens or "venography" in tokens:
        return "MRV"
    if "petct" in tokens or ({"pet", "ct"} <= tokens):
        return "PETCT"
    if "scintigraphy" in tokens or "nuclear" in tokens or "tc99m" in tokens or "technetium" in tokens:
        return "NM"
    if "xray" in tokens or "radiograph" in tokens or "plainfilm" in tokens:
        return "XR"
    if "ultrasound" in tokens or "sonography" in tokens or "doppler" in tokens or "us" in tokens:
        return "US"
    if "ct" in tokens:
        return "CT"
    if "mri" in tokens or "mr" in tokens or tokens & {"dwi", "adc", "flair", "swi", "gre", "t1", "t2", "stir"}:
        return "MRI"
    if "histopathology" in tokens or "pathology" in tokens or "microscopy" in tokens:
        return "PATH"
    if "photo" in tokens or "clinical" in tokens:
        return "PHOTO"
    if "illustration" in tokens or "diagram" in tokens or "drawing" in tokens:
        return "ILLUSTRATION"
    return normalize_space(modality or "unspecified").upper()


def _canonical_study_region(region: str, matcher_tokens: set[str]) -> str:
    tokens = set(matcher_tokens)
    region_text = normalize_match_text(region)
    regions = canonical_regions_from_tokens(tokens | set(region_text.split()))
    if "paranasalsinuses" in regions:
        return "paranasal_sinuses"
    if "temporalbone" in regions:
        return "temporal_bone"
    if "brain" in regions or "head" in regions or "posteriorfossa" in regions:
        return "brain_head"
    if {"cspine", "tspine", "lspine", "lssp"} & regions:
        spine_order = ["cspine", "tspine", "lspine", "lssp"]
        present = [x for x in spine_order if x in regions]
        return "spine_" + "_".join(present) if present else "spine"
    if "orbit" in regions:
        return "orbit"
    if "headneck" in regions:
        return "headneck"
    if "chest" in regions:
        return "chest"
    if "abdomen" in regions:
        return "abdomen"
    if "pelvis" in regions:
        return "pelvis"
    if "vertebralartery" in regions:
        return "vascular"
    return region_text or "unspecified"


def _canonical_study_timepoint(bundle: ExamBundle) -> str:
    if is_exam_bundle_prior_timepoint(bundle):
        return "prior_patient_provided"
    if is_exam_bundle_followup_like(bundle):
        text = normalize_match_text(bundle.time_past or bundle.matcher_summary)
        if "post" in text or "after" in text or "follow" in text or "therapy" in text or "treatment" in text:
            return "followup_or_post_treatment"
        return "delayed_or_time_specified"
    return "initial_or_unspecified"


def _canonical_study_subtype(modality: str, acquisition: str, matcher_tokens: set[str]) -> str:
    tokens = set(matcher_tokens) | set(normalize_match_text(acquisition).split())
    if modality == "MRI":
        if "spectroscopy" in tokens or "mrs" in tokens:
            return "spectroscopy"
        return "routine_mri"
    if modality == "CT":
        if "perfusion" in tokens:
            return "perfusion"
        return "routine_ct"
    return modality.lower()


def _study_group_key(bundle: ExamBundle) -> Tuple[str, str, str, str]:
    modality = _canonical_study_modality(bundle.modality, bundle.acquisition)
    region = _canonical_study_region(bundle.region, bundle.matcher_tokens)
    timepoint = _canonical_study_timepoint(bundle)
    subtype = _canonical_study_subtype(modality, bundle.acquisition, bundle.matcher_tokens)
    return modality, region, timepoint, subtype


def group_exam_bundles_into_studies(figure_bundles: Sequence[ExamBundle]) -> List[ExamBundle]:
    grouped: Dict[Tuple[str, str, str, str], List[ExamBundle]] = {}
    order: List[Tuple[str, str, str, str]] = []
    for bundle in figure_bundles:
        key = _study_group_key(bundle)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(bundle)

    study_bundles: List[ExamBundle] = []
    for group_index, key in enumerate(order, start=1):
        parts = grouped[key]
        source_figures = [part.figure for part in parts]
        figure_label = "+".join(source_figures)
        preferred_orders = [part.preferred_order for part in parts if part.preferred_order is not None]
        preferred_order = min(preferred_orders) if preferred_orders else None
        label = "essential" if any(part.label == "essential" for part in parts) else "optional"
        image_payloads = [payload for part in parts for payload in part.image_payloads]
        modality = _join_unique(part.modality for part in parts) or key[0]
        acquisition = _join_unique(part.acquisition for part in parts)
        view = _join_unique(part.view for part in parts)
        region = _join_unique(part.region for part in parts) or key[1]
        contrast_values = deduplicate_preserve_order(part.contrast for part in parts if normalize_space(part.contrast))
        contrast = "MIXED" if len({normalize_key(x) for x in contrast_values}) > 1 else (contrast_values[0] if contrast_values else "")
        time_values = deduplicate_preserve_order(part.time_past for part in parts if normalize_space(part.time_past))
        time_past = _join_unique(time_values)
        raw_time_values = deduplicate_preserve_order(part.time_past_raw for part in parts if normalize_space(part.time_past_raw))
        time_past_raw = _join_unique(raw_time_values) or None
        route_evaluable = any(part.route_evaluable for part in parts)
        evidence_modality_class = "radiology" if route_evaluable else "non_imaging"
        time_past_interpretable = all(part.time_past_interpretable for part in parts)
        matcher_summary = " | ".join([
            f"Study {group_index}", figure_label, modality or "unspecified modality",
            acquisition or "unspecified acquisition", view or "unspecified view",
            region or "unspecified region", contrast or "unspecified contrast",
            time_past or "initial_or_unspecified", _join_unique(part.matcher_summary for part in parts),
        ])
        matcher_tokens: set[str] = set()
        for part in parts:
            matcher_tokens.update(part.matcher_tokens)
        matcher_tokens.update(tokenize_match_text(" ".join([key[0], key[1], key[2], key[3], matcher_summary])))
        study_bundles.append(
            ExamBundle(
                exam_id=f"Study {group_index}",
                figure=figure_label,
                preferred_order=preferred_order,
                label=label,
                modality=modality,
                acquisition=acquisition,
                view=view,
                region=region,
                contrast=contrast,
                time_past=time_past or None,
                image_payloads=image_payloads,
                matcher_summary=matcher_summary,
                matcher_tokens=matcher_tokens,
                source_figures=source_figures,
                source_exam_count=len(parts),
                route_evaluable=route_evaluable,
                evidence_modality_class=evidence_modality_class,
                time_past_raw=time_past_raw,
                time_past_interpretable=time_past_interpretable,
            )
        )
    return study_bundles


def canonicalize_case(raw_case: Dict[str, Any], image_root: Optional[Path], *, reveal_unit: Optional[str] = None) -> CanonicalCase:
    case_id = normalize_space(raw_case.get("case_id") or raw_case.get("id") or raw_case.get("case_url") or "unknown_case")
    case_title = normalize_space(raw_case.get("case_title") or "")
    section = normalize_space(raw_case.get("section") or "")
    area_of_interest = normalize_space(raw_case.get("area_of_interest") or "")
    question = normalize_space(((raw_case.get("item") or {}).get("stem")) or "")
    clinical_history = normalize_space(raw_case.get("clinical_history") or "")
    final_diagnosis = normalize_space(
        raw_case.get("final_diagnosis")
        or (((raw_case.get("item") or {}).get("open_ended") or {}).get("reference_answer"))
        or ""
    )

    rubric_block = raw_case.get("rubric_0_to_3") or (((raw_case.get("item") or {}).get("open_ended") or {}).get("rubric_0_to_3")) or {}
    diagnosis_rubric = rubric_block.get("Diagnosis") or {}
    localization_rubric = rubric_block.get("Localisation") or {}

    reference_ddx_options = parse_mcq_options(raw_case)
    root_image_index = build_root_image_index(raw_case, image_root)

    labels_by_figure: Dict[str, str] = {}
    step_entries = raw_case.get("diagnostic_steps") or []
    for step in step_entries:
        figure = normalize_space(step.get("source"))
        if not figure:
            continue
        labels_by_figure[figure] = standardize_exam_label(step.get("essential"))

    figure_exam_pool: List[ExamBundle] = []
    excluded_exam_pool: List[ExamBundle] = []
    excluded_reasons_by_exam: Dict[str, str] = {}

    for idx, exam in enumerate(raw_case.get("imaging_examinations") or [], start=1):
        figure = normalize_space(exam.get("figure") or f"Figure {idx}")
        meta = exam.get("meta_info") or {}
        modality = normalize_space(meta.get("modality") or exam.get("modality") or "")
        acquisition = normalize_space(meta.get("acquisition") or "")
        view = normalize_space(meta.get("view") or "")
        region = normalize_space(meta.get("imaged_region") or "")
        contrast = normalize_space(meta.get("contrast") or "")
        time_past, time_past_raw, time_past_interpretable = normalize_time_past(meta.get("time_past"))
        preferred_order_raw = exam.get("preferred_order")
        preferred_order = int(preferred_order_raw) if preferred_order_raw not in (None, "") else None
        label = labels_by_figure.get(figure, "optional")
        image_payloads = resolve_exam_images(raw_case, exam, image_root, root_image_index)

        matcher_summary = " | ".join(
            [
                figure,
                modality or "unspecified modality",
                acquisition or "unspecified acquisition",
                view or "unspecified view",
                region or "unspecified region",
                contrast or "unspecified contrast",
                time_past or "initial_or_unspecified",
                normalize_space(exam.get("image_caption") or ""),
            ]
        )
        matcher_tokens = build_exam_matcher_tokens(
            modality,
            acquisition,
            view,
            region,
            contrast,
            figure,
            extra_text=" ".join([time_past or "", normalize_space(exam.get("image_caption") or "")]),
        )
        evidence_modality_class, route_evaluable = classify_evidence_modality(modality, acquisition)

        bundle = ExamBundle(
            exam_id=figure,
            figure=figure,
            preferred_order=preferred_order,
            label=label,
            modality=modality,
            acquisition=acquisition,
            view=view,
            region=region,
            contrast=contrast,
            time_past=time_past,
            image_payloads=image_payloads,
            matcher_summary=matcher_summary,
            matcher_tokens=matcher_tokens,
            source_figures=[figure],
            source_exam_count=1,
            route_evaluable=route_evaluable,
            evidence_modality_class=evidence_modality_class,
            time_past_raw=time_past_raw,
            time_past_interpretable=time_past_interpretable,
        )

        if is_exam_bundle_future_timepoint(bundle):
            excluded_exam_pool.append(bundle)
            excluded_reasons_by_exam[bundle.exam_id] = "future_followup_time_past"
        elif not bundle.image_payloads:
            excluded_exam_pool.append(bundle)
            excluded_reasons_by_exam[bundle.exam_id] = "missing_local_images"
        else:
            figure_exam_pool.append(bundle)

    reveal_unit_key = normalize_key(reveal_unit or DEFAULT_REVEAL_UNIT)
    if reveal_unit_key in {"eurorad", "euro_rad", "eurorad_figure", "raw", "figure", "figure_level", "figure-level"}:
        # EuroRad-style default: keep each source imaging_examination / figure-protocol entry
        # as one requestable evidence unit. Do not merge T1/T2/FLAIR/STIR/etc. into
        # a clinical study-level bundle.
        official_exam_pool = figure_exam_pool
    else:
        raise ValueError(f"Unsupported reveal_unit={reveal_unit!r}. Use 'eurorad' or 'figure'.")

    gold_order_by_exam = {
        exam.exam_id: int(exam.preferred_order)
        for exam in official_exam_pool
        if exam.preferred_order is not None
    }
    gold_labels_by_exam = {exam.exam_id: exam.label for exam in official_exam_pool}

    return CanonicalCase(
        case_id=case_id,
        case_title=case_title,
        section=section,
        area_of_interest=area_of_interest,
        question=question,
        clinical_history=clinical_history,
        final_diagnosis=final_diagnosis,
        reference_ddx_options=reference_ddx_options,
        diagnosis_rubric=diagnosis_rubric,
        localization_rubric=localization_rubric,
        difficulty=normalize_space(raw_case.get("difficulty") or ""),
        rarity=normalize_space(raw_case.get("rarity") or ""),
        official_exam_pool=official_exam_pool,
        excluded_exam_pool=excluded_exam_pool,
        gold_labels_by_exam=gold_labels_by_exam,
        gold_order_by_exam=gold_order_by_exam,
        excluded_reasons_by_exam=excluded_reasons_by_exam,
        raw_case=raw_case,
    )


# =========================
# Matcher
# =========================

PHRASE_REPLACEMENTS = {
    "with and without contrast": "mixedcontrast",
    "with/without contrast": "mixedcontrast",
    "pre and post contrast": "mixedcontrast",
    "pre- and post-contrast": "mixedcontrast",
    "pre and post gadolinium": "mixedcontrast gadolinium",
    "pre- and post-gadolinium": "mixedcontrast gadolinium",
    "magnetic resonance imaging": "mri",
    "magnetic resonance angiography": "mra",
    "magnetic resonance venography": "mrv",
    "computed tomography angiography": "cta",
    "ct angiography": "cta",
    "ct angiogram": "cta",
    "angio ct": "cta",
    "computed tomography": "ct",
    "plain film": "radiograph",
    "plain radiograph": "radiograph",
    "radiography": "radiograph",
    "x-ray": "xray",
    "diffusion weighted imaging": "dwi",
    "diffusion-weighted imaging": "dwi",
    "apparent diffusion coefficient": "adc",
    "fluid attenuated inversion recovery": "flair",
    "susceptibility weighted imaging": "swi",
    "susceptibility-weighted imaging": "swi",
    "gradient echo": "gre",
    "t1 weighted": "t1",
    "t1-weighted": "t1",
    "t1wi": "t1",
    "t2 weighted": "t2",
    "t2-weighted": "t2",
    "t2wi": "t2",
    "post gadolinium": "postcontrast",
    "post-contrast": "postcontrast",
    "post contrast": "postcontrast",
    "contrast enhanced": "postcontrast",
    "contrast-enhanced": "postcontrast",
    "gadolinium enhanced": "postcontrast",
    "gadolinium-enhanced": "postcontrast",
    "without contrast": "noncontrast",
    "non contrast": "noncontrast",
    "non-contrast": "noncontrast",
    "non enhanced": "noncontrast",
    "non-enhanced": "noncontrast",
    "unenhanced": "noncontrast",
    "head and neck": "headneck",
    "posterior fossa": "posteriorfossa",
    "skull base": "skullbase",
    "temporal bone": "temporalbone",
    "paranasal sinuses": "paranasalsinuses",
    "cervical spine": "cspine",
    "thoracic spine": "tspine",
    "lumbar spine": "lspine",
    "lumbosacral spine": "lssp",
    "dorsal spine": "tspine",
    "vertebral artery": "vertebralartery",
    "digital subtraction angiography": "dsa",
    "catheter angiography": "dsa",
    "cerebral angiography": "dsa",
    "angiography dsa": "dsa",
    "x ray": "xray",
    "pet ct": "petct",
    "pet/ct": "petct",
    "tc-99m": "tc99m",
    "tc 99m": "tc99m",
    "technetium-99m": "tc99m",
    "mr angiography": "mra",
    "mr venography": "mrv",
    "mri/mra": "mra",
}

MODALITY_ALIASES = {
    "mri": {"mri", "mr", "dwi", "adc", "flair", "swi", "gre", "t1", "t2", "stir", "spectroscopy"},
    "mra": {"mra"},
    "mrv": {"mrv"},
    "ct": {"ct"},
    "cta": {"cta"},
    "dsa": {"dsa", "angiography"},
    "xray": {"xray", "radiograph", "radiography", "plainfilm", "plain"},
    "ultrasound": {"ultrasound", "us", "sonography", "doppler"},
    "petct": {"petct", "pet"},
    "nuclear_medicine": {"nuclear", "scintigraphy", "technetium", "tc99m"},
    "photo": {"photo", "clinical"},
    "illustration": {"illustration", "drawing", "diagram"},
    "histopathology": {"histopathology", "pathology", "microscopy"},
    "fluoroscopy": {"fluoroscopy"},
}

REGION_HINTS = {
    "brain": {"brain", "head", "intracranial", "cranial", "cerebral", "cerebellar", "brainstem", "medulla", "pons", "temporal", "frontal", "parietal", "occipital", "posteriorfossa"},
    "head": {"head", "brain", "skull", "cranium", "cranial", "intracranial", "skullbase"},
    "orbit": {"orbit", "orbits", "ocular"},
    "temporalbone": {"temporalbone", "petrous"},
    "paranasalsinuses": {"paranasalsinuses", "sinus", "sinuses"},
    "headneck": {"headneck", "neck", "cervical"},
    "cspine": {"cspine", "cervical", "spine", "cord"},
    "tspine": {"tspine", "thoracic", "dorsal", "spine", "cord"},
    "lspine": {"lspine", "lumbar", "spine", "cord"},
    "lssp": {"lssp", "lumbosacral", "sacral", "spine"},
    "posteriorfossa": {"posteriorfossa", "brainstem", "cerebellum", "medulla", "pons"},
    "chest": {"chest", "thorax", "thoracic", "lung", "mediastinum"},
    "abdomen": {"abdomen", "abdominal", "hepatic", "liver", "renal", "kidney"},
    "pelvis": {"pelvis", "pelvic", "hip", "sacrum"},
    "vertebralartery": {"vertebralartery", "vertebral", "vascular", "artery", "arterial"},
}

ACQUISITION_HINTS = {
    "dwi": {"dwi", "adc", "restricted", "diffusion"},
    "flair": {"flair"},
    "swi": {"swi", "susceptibility", "gre", "gradient"},
    "t1": {"t1"},
    "t2": {"t2"},
    "stir": {"stir"},
    "spectroscopy": {"spectroscopy", "mrs"},
    "cta": {"cta", "angiogram", "angiography", "angioct", "mip"},
    "mra": {"mra", "angiography"},
    "mrv": {"mrv", "venography"},
    "dsa": {"dsa", "angiography", "catheter"},
    "radiograph": {"radiograph", "xray"},
    "ultrasound": {"ultrasound", "doppler", "sonography"},
    "nuclear_medicine": {"nuclear", "scintigraphy", "technetium", "tc99m"},
    "bonewindow": {"bonewindow", "bone", "algorithm"},
    "postcontrast": {"postcontrast", "contrast"},
    "noncontrast": {"noncontrast"},
    "mixedcontrast": {"mixedcontrast"},
}

CONTRAST_HINTS = {
    "yes": {"yes", "postcontrast", "contrast"},
    "no": {"no", "noncontrast"},
    "mixed": {"yes", "no", "postcontrast", "noncontrast", "contrast", "mixedcontrast"},
}

ANATOMIC_COMPOUND_REGIONS = {"head", "brain", "neck", "headneck", "chest", "abdomen", "pelvis", "cspine", "tspine", "lspine", "lssp", "spine", "orbit", "paranasalsinuses", "temporalbone"}
SPECIALIZED_ACQUISITIONS = {"cta", "mra", "mrv", "dsa", "spectroscopy", "nuclear_medicine", "bonewindow"}
SPINE_LEVELS = {"cspine", "tspine", "lspine", "lssp"}
SEQUENCE_ACQUISITIONS = {"t1", "t2", "flair", "dwi", "swi", "stir", "spectroscopy", "bonewindow", "radiograph", "ultrasound", "nuclear_medicine"}


def spine_level_conflict(req_regions: set[str], ex_regions: set[str]) -> bool:
    req_spine = req_regions & SPINE_LEVELS
    ex_spine = ex_regions & SPINE_LEVELS
    if not req_spine or not ex_spine:
        return False
    # Lumbar and lumbosacral are close enough for this matcher.
    if {"lspine", "lssp"} & req_spine and {"lspine", "lssp"} & ex_spine:
        return False
    return not bool(req_spine & ex_spine)


def normalize_match_text(text: str) -> str:
    text = normalize_key(text)
    text = text.replace("–", "-").replace("—", "-")
    for src, dst in sorted(PHRASE_REPLACEMENTS.items(), key=lambda item: -len(item[0])):
        text = text.replace(src, dst)
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[(),:;+]", " ", text)
    text = re.sub(r"[^a-z0-9_ \-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_match_text(text: str) -> set[str]:
    norm = normalize_match_text(text)
    tokens = set(norm.split())
    expanded = set(tokens)
    for token in list(tokens):
        if token.startswith("figure"):
            expanded.add("figure")
        if token.endswith("weighted"):
            expanded.add(token[:-8])
    return {tok for tok in expanded if tok}


def canonical_modalities_from_tokens(tokens: set[str]) -> set[str]:
    out: set[str] = set()
    for canonical, aliases in MODALITY_ALIASES.items():
        if tokens & aliases:
            out.add(canonical)
    if "cta" in tokens:
        out.add("ct")
    if any(tok in tokens for tok in {"dwi", "adc", "flair", "swi", "gre", "t1", "t2", "stir", "spectroscopy", "mra", "mrv"}):
        out.add("mri")
    return out


def canonical_regions_from_tokens(tokens: set[str]) -> set[str]:
    out: set[str] = set()
    for canonical, aliases in REGION_HINTS.items():
        if tokens & aliases:
            out.add(canonical)
    spine_level_tokens = {"cervical", "thoracic", "dorsal", "lumbar", "lumbosacral", "sacral", "cspine", "tspine", "lspine", "lssp"}
    if "spine" in tokens:
        out.add("spine")
        if not (tokens & spine_level_tokens):
            out.difference_update({"cspine", "tspine", "lspine", "lssp"})
    return out


def canonical_acquisitions_from_tokens(tokens: set[str]) -> set[str]:
    out: set[str] = set()
    for canonical, aliases in ACQUISITION_HINTS.items():
        if tokens & aliases:
            out.add(canonical)
    return out


def canonical_contrast_from_tokens(tokens: set[str]) -> Optional[str]:
    if "mixedcontrast" in tokens:
        return "mixed"
    if "postcontrast" in tokens or "contrast" in tokens:
        if "noncontrast" in tokens:
            return "mixed"
        return "yes"
    if "noncontrast" in tokens:
        return "no"
    return None


def build_exam_matcher_tokens(modality: str, acquisition: str, view: str, region: str, contrast: str, figure: str, *, extra_text: str = "") -> set[str]:
    composite = " ".join([modality, acquisition, view, region, contrast, figure, extra_text])
    return tokenize_match_text(composite)


def summarize_request_semantics(request_text: str) -> Dict[str, Any]:
    tokens = tokenize_match_text(request_text)
    return {
        "tokens": tokens,
        "modalities": canonical_modalities_from_tokens(tokens),
        "regions": canonical_regions_from_tokens(tokens),
        "acquisitions": canonical_acquisitions_from_tokens(tokens),
        "contrast": canonical_contrast_from_tokens(tokens),
    }


def summarize_exam_semantics(exam: ExamBundle) -> Dict[str, Any]:
    tokens = set(exam.matcher_tokens)
    return {
        "tokens": tokens,
        "modalities": canonical_modalities_from_tokens(tokens),
        "regions": canonical_regions_from_tokens(tokens),
        "acquisitions": canonical_acquisitions_from_tokens(tokens),
        "contrast": canonical_contrast_from_tokens(tokens),
    }


def request_explicitly_asks_prior(request_text: str) -> bool:
    return bool(PRIOR_TEXT_RE.search(request_text or ""))


def request_explicitly_asks_followup(request_text: str) -> bool:
    text = request_text or ""
    return bool(FOLLOWUP_TEXT_RE.search(text)) and not request_explicitly_asks_prior(text)


def score_request_against_exam(request_text: str, exam: ExamBundle) -> float:
    req = summarize_request_semantics(request_text)
    ex = summarize_exam_semantics(exam)

    if req["modalities"] and ex["modalities"] and not (req["modalities"] & ex["modalities"]):
        return -100.0

    score = 0.0

    if req["modalities"] and ex["modalities"]:
        score += 6.0

    if req["acquisitions"]:
        overlap = req["acquisitions"] & ex["acquisitions"]
        score += 3.0 * len(overlap)
        if not overlap:
            score -= 1.0
            if req["acquisitions"] & SEQUENCE_ACQUISITIONS:
                # EuroRad-style protocol-level requests should not match the wrong MRI sequence/view
                # merely because modality and anatomy are correct.
                score -= 3.0

    req_special = req["acquisitions"] & SPECIALIZED_ACQUISITIONS
    ex_special = ex["acquisitions"] & SPECIALIZED_ACQUISITIONS
    if req_special:
        if req_special & ex_special:
            score += 2.0
        else:
            score -= 4.0
    elif ex_special:
        score -= 0.75

    if req["regions"]:
        overlap = req["regions"] & ex["regions"]
        score += 2.0 * len(overlap)
        if not overlap:
            score -= 0.5
        if spine_level_conflict(req["regions"], ex["regions"]):
            score -= 5.0

    if req["contrast"] and ex["contrast"]:
        if req["contrast"] == ex["contrast"] or req["contrast"] == "mixed" or ex["contrast"] == "mixed":
            score += 1.0
        else:
            score -= 1.0

    asks_prior = request_explicitly_asks_prior(request_text)
    exam_prior = is_exam_bundle_prior_timepoint(exam)
    asks_followup = request_explicitly_asks_followup(request_text)
    exam_followup = is_exam_bundle_followup_like(exam)
    if asks_prior and exam_prior:
        score += 1.5
    elif asks_prior and not exam_prior:
        score -= 1.0
    elif asks_followup and exam_followup:
        score += 1.5
    elif asks_followup and not exam_followup:
        score -= 1.0
    elif exam_followup:
        score -= 0.5

    lexical_overlap = len(req["tokens"] & ex["tokens"])
    score += 0.35 * lexical_overlap

    if len(req["tokens"]) <= 2:
        score -= 0.25

    return score


def _scored_exam_candidates(request_text: str, exam_pool: Sequence[ExamBundle]) -> List[Tuple[ExamBundle, float]]:
    candidates: List[Tuple[ExamBundle, float]] = []
    for exam in exam_pool:
        candidates.append((exam, score_request_against_exam(request_text, exam)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _candidate_scores(candidates: Sequence[Tuple[ExamBundle, float]], limit: int = 5, prefix: str = "") -> List[Tuple[str, float]]:
    return [(f"{prefix}{exam.figure}", float(score)) for exam, score in candidates[:limit]]


def patient_provided_timepoint_exams(
    case: CanonicalCase,
    requests: Optional[Sequence[RequestEvent]] = None,
) -> List[ExamBundle]:
    revealed_exam_ids = {
        req.matched_exam_id
        for req in (requests or [])
        if req.outcome == "matched" and req.matched_exam_id
    }
    return [
        exam
        for exam in case.official_exam_pool
        if exam.exam_id not in revealed_exam_ids
        and getattr(exam, "route_evaluable", True)
        and is_exam_bundle_prior_timepoint(exam)
    ]


def describe_patient_provided_exam(exam: ExamBundle) -> str:
    details = [
        exam.modality or "unspecified modality",
        exam.region or "unspecified region",
    ]
    if normalize_space(exam.acquisition):
        details.append(exam.acquisition)
    if normalize_space(exam.contrast):
        details.append(f"contrast: {exam.contrast}")
    details.append(f"timepoint: {exam.time_past}")
    return "; ".join(details)


def format_patient_provided_timepoint_block(
    case: CanonicalCase,
    requests: Optional[Sequence[RequestEvent]] = None,
) -> str:
    exams = patient_provided_timepoint_exams(case, requests)
    if not exams:
        return ""
    lines = [
        "Patient-provided prior/comparison imaging available on request:",
        "The patient brought the following prior imaging from before the current presentation. These images are not shown unless you request the relevant study.",
    ]
    for idx, exam in enumerate(exams, start=1):
        lines.append(f"- Available prior study {idx}: {describe_patient_provided_exam(exam)}")
    lines.append("If one of these studies would help, request it explicitly in natural language.")
    return "\n".join(lines)


def _top_scored_exam(request_text: str, exam_pool: Sequence[ExamBundle]) -> Tuple[Optional[ExamBundle], float, List[Tuple[str, float]]]:
    candidates = _scored_exam_candidates(request_text, exam_pool)
    top_exam = candidates[0][0] if candidates else None
    top_score = float(candidates[0][1]) if candidates else -999.0
    return top_exam, top_score, _candidate_scores(candidates)


def _break_timepoint_tie(request_text: str, candidates: Sequence[Tuple[ExamBundle, float]], ambiguity_margin: float) -> Optional[ExamBundle]:
    if not candidates:
        return None
    top_score = candidates[0][1]
    near = [(exam, score) for exam, score in candidates if top_score - score < ambiguity_margin]
    if len(near) <= 1:
        return near[0][0] if near else None
    asks_prior = request_explicitly_asks_prior(request_text)
    asks_followup = request_explicitly_asks_followup(request_text)
    if asks_prior:
        prior = [exam for exam, _ in near if is_exam_bundle_prior_timepoint(exam)]
        if len(prior) == 1:
            return prior[0]
    if asks_followup:
        delayed = [exam for exam, _ in near if is_exam_bundle_followup_like(exam)]
        if len(delayed) == 1:
            return delayed[0]
    else:
        initial = [exam for exam, _ in near if not is_exam_bundle_followup_like(exam)]
        if len(initial) == 1:
            return initial[0]
    return None


def acquisition_specificity_gain(request_text: str, candidate: Optional[ExamBundle], blocker: Optional[ExamBundle]) -> float:
    if candidate is None or blocker is None:
        return 0.0
    req_acq = summarize_request_semantics(request_text)["acquisitions"]
    if not req_acq:
        return 0.0
    cand_acq = summarize_exam_semantics(candidate)["acquisitions"]
    block_acq = summarize_exam_semantics(blocker)["acquisitions"]
    return float(len(req_acq & cand_acq) - len(req_acq & block_acq))


def compound_region_count(request_text: str) -> int:
    norm = normalize_match_text(request_text)
    tokens = set(norm.split())
    regions = canonical_regions_from_tokens(tokens)
    # Collapse head/brain to one conceptual region; keep explicitly distinct body regions separate.
    if "brain" in regions or "head" in regions:
        regions.discard("brain")
        regions.discard("head")
        regions.add("head_brain")
    spine_parts = {"cspine", "tspine", "lspine", "lssp"} & regions
    if spine_parts:
        for part in spine_parts:
            regions.discard(part)
        regions.add("spine")
    return len(regions & ANATOMIC_COMPOUND_REGIONS | ({"head_brain"} if "head_brain" in regions else set()))


def is_compound_request_text(request_text: str) -> bool:
    norm = normalize_match_text(request_text)
    if compound_region_count(request_text) < 2:
        return False
    return bool(re.search(r"\b(and|plus|,|/|with)\b", norm))


def distinct_bundle_keys(candidates: Sequence[Tuple[ExamBundle, float]]) -> set[Tuple[str, str, str, str]]:
    return {_study_group_key(exam) for exam, _ in candidates}


def generic_spine_request_has_multiple_high_candidates(request_text: str, candidates: Sequence[Tuple[ExamBundle, float]], match_threshold: float) -> bool:
    req_regions = summarize_request_semantics(request_text)["regions"]
    if "spine" not in req_regions or (req_regions & SPINE_LEVELS):
        return False
    high_spine = []
    for exam, score in candidates:
        if score < match_threshold:
            continue
        ex_regions = summarize_exam_semantics(exam)["regions"]
        if "spine" in ex_regions or (ex_regions & SPINE_LEVELS):
            high_spine.append(exam)
    return len(high_spine) > 1


def best_effort_ambiguous_match(
    reason: str,
    best_exam: Optional[ExamBundle],
    best_score: float,
    candidate_scores: List[Tuple[str, float]],
) -> MatchResolution:
    if best_exam is None:
        return MatchResolution(outcome="invalid", reason="no_match_above_threshold", candidate_scores=candidate_scores)
    return MatchResolution(
        outcome="matched",
        reason=reason,
        matched_exam=best_exam,
        match_score=float(best_score),
        candidate_scores=candidate_scores,
    )


def resolve_request_to_exam(
    request_text: str,
    official_exam_pool: Sequence[ExamBundle],
    excluded_exam_pool: Sequence[ExamBundle],
    revealed_exam_ids: set[str],
    attempted_request_texts: set[str],
    *,
    match_threshold: float = 5.0,
    ambiguity_margin: float = 0.75,
) -> MatchResolution:
    normalized_request_text = normalize_match_text(request_text)
    if not normalized_request_text:
        return MatchResolution(outcome="invalid", reason="empty_request")

    if normalized_request_text in attempted_request_texts:
        return MatchResolution(outcome="invalid", reason="duplicate_request_text")

    unrevealed_official = [exam for exam in official_exam_pool if exam.exam_id not in revealed_exam_ids]
    revealed_official = [exam for exam in official_exam_pool if exam.exam_id in revealed_exam_ids]

    scored_unrevealed = _scored_exam_candidates(request_text, unrevealed_official)
    scored_revealed = _scored_exam_candidates(request_text, revealed_official)
    scored_excluded = _scored_exam_candidates(request_text, excluded_exam_pool)

    best_exam = scored_unrevealed[0][0] if scored_unrevealed else None
    best_score = float(scored_unrevealed[0][1]) if scored_unrevealed else -999.0
    best_revealed = scored_revealed[0][0] if scored_revealed else None
    best_revealed_score = float(scored_revealed[0][1]) if scored_revealed else -999.0
    best_excluded = scored_excluded[0][0] if scored_excluded else None
    best_excluded_score = float(scored_excluded[0][1]) if scored_excluded else -999.0

    candidate_scores = _candidate_scores(scored_unrevealed)
    revealed_scores = _candidate_scores(scored_revealed, prefix="REVEALED::")
    excluded_scores = _candidate_scores(scored_excluded, prefix="UNAVAILABLE::")

    if not unrevealed_official:
        reason = "no_official_exam_pool"
        if best_revealed and best_revealed_score >= match_threshold:
            reason = "already_revealed_exam_requested"
        elif best_excluded and best_excluded_score >= match_threshold:
            reason = "unavailable_exam_requested"
        return MatchResolution(
            outcome="invalid",
            reason=reason,
            candidate_scores=candidate_scores + revealed_scores[:2] + excluded_scores[:2],
        )

    second_score = float(scored_unrevealed[1][1]) if len(scored_unrevealed) > 1 else -999.0

    near_unrevealed = [(exam, score) for exam, score in scored_unrevealed if score >= match_threshold and best_score - score < ambiguity_margin]
    strong_unrevealed = [(exam, score) for exam, score in scored_unrevealed if score >= match_threshold]
    if is_compound_request_text(request_text) and len(distinct_bundle_keys(strong_unrevealed)) > 1:
        return best_effort_ambiguous_match("matched_ambiguous_compound_best_effort", best_exam, best_score, candidate_scores)

    if generic_spine_request_has_multiple_high_candidates(request_text, strong_unrevealed, match_threshold):
        return best_effort_ambiguous_match("matched_ambiguous_generic_spine_best_effort", best_exam, best_score, candidate_scores)

    if best_score < match_threshold:
        if best_revealed and best_revealed_score >= match_threshold:
            return MatchResolution(
                outcome="invalid",
                reason="already_revealed_exam_requested",
                candidate_scores=revealed_scores[:3] + candidate_scores[:3],
            )
        if best_excluded and best_excluded_score >= match_threshold:
            return MatchResolution(
                outcome="invalid",
                reason="unavailable_exam_requested",
                candidate_scores=excluded_scores[:3] + candidate_scores[:3],
            )
        return MatchResolution(outcome="invalid", reason="no_match_above_threshold", candidate_scores=candidate_scores)

    if best_excluded and best_excluded_score >= match_threshold and best_excluded_score > best_score + ambiguity_margin:
        return MatchResolution(
            outcome="invalid",
            reason="unavailable_exam_requested",
            candidate_scores=excluded_scores[:3] + candidate_scores[:3],
        )

    if best_revealed and best_revealed_score >= match_threshold:
        gain = acquisition_specificity_gain(request_text, best_exam, best_revealed)
        if best_revealed_score > best_score + ambiguity_margin and gain <= 0:
            return MatchResolution(
                outcome="invalid",
                reason="already_revealed_exam_requested",
                candidate_scores=revealed_scores[:3] + candidate_scores[:3],
            )
        if abs(best_revealed_score - best_score) < ambiguity_margin and gain <= 0:
            return best_effort_ambiguous_match(
                "matched_ambiguous_revealed_tie_best_effort",
                best_exam,
                best_score,
                revealed_scores[:3] + candidate_scores[:3],
            )

    if best_score - second_score < ambiguity_margin:
        tie_break_exam = _break_timepoint_tie(request_text, scored_unrevealed, ambiguity_margin)
        if tie_break_exam is not None:
            return MatchResolution(
                outcome="matched",
                reason="matched_timepoint_tiebreak",
                matched_exam=tie_break_exam,
                match_score=float(score_request_against_exam(request_text, tie_break_exam)),
                candidate_scores=candidate_scores,
            )
        return best_effort_ambiguous_match("matched_ambiguous_score_tie_best_effort", best_exam, best_score, candidate_scores)

    return MatchResolution(
        outcome="matched",
        reason="matched",
        matched_exam=best_exam,
        match_score=float(best_score),
        candidate_scores=candidate_scores,
    )


# =========================
# Prompts and output normalization
# =========================

AGENT_SYSTEM_PROMPT = """You are a rigorous radiology diagnostic agent.
You are participating in a benchmark with a hidden exam inventory.
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
- Every probability must be in [0,1].
- The 4 probabilities must sum to 1.
- If action=request_exam, request exactly one EuroRad-style imaging evidence bundle in free-form natural language; final_location may contain empty strings.
- If action=stop, final_location must describe the lesion location and requested_examination must be empty.
- Each imaging request consumes one step, including unavailable, duplicate, or out-of-scope requests.
- Request imaging only when it is expected to help confirm, exclude, localize, stage, or characterize a relevant diagnosis.
- Request one bundle at a time. Do not combine multiple distinct body regions or modalities in one request.
- Include modality and anatomic region; include sequence/acquisition/contrast or delayed/follow-up timepoint when that distinction matters.
- The hidden inventory may include initial, comparison, follow-up, postoperative, or post-treatment imaging. If you specifically need a delayed/follow-up study, say so explicitly.
- Do not request imaging merely to exhaust the budget. Stop and submit a final answer when you have sufficient evidence.
- Do not mention candidate diagnosis lists or hidden inventory.
- Use only the clinical history, the revealed image evidence, and the revealed minimal exam metadata.
- Expert captions, key findings, and final answers are not provided to you.
"""


def build_initial_turn_prompt(case: CanonicalCase, budget: int) -> str:
    available_bundle_count = len(case.official_exam_pool)
    patient_provided_block = format_patient_provided_timepoint_block(case)
    if patient_provided_block:
        patient_provided_block = f"\n{patient_provided_block}\n"
    return f"""Clinical history:
{case.clinical_history or '[none]'}

Task context:
You are evaluating an unknown diagnostic imaging case. Imaging evidence is hidden until requested. Diagnose and localize the case by actively requesting available EuroRad-style figure/protocol-level imaging bundles when useful.

Official setting reminders:
- Hidden available exam bundles in this case: {available_bundle_count}.
- The hidden exam inventory list is NOT revealed.
- You may request at most {budget} imaging examinations in total.
- No candidate diagnosis list is given.
- Every request consumes one step, even if it is unavailable, duplicate, or out-of-scope.
- Available bundles may include initial, comparison, delayed/follow-up, postoperative, or post-treatment studies.
{patient_provided_block}

This is the first decision turn. No imaging examination has been revealed yet.
Task: output your current four-item differential with probabilities, then either request one next examination bundle or stop if you already have enough evidence.
Respond in STRICT JSON only.
"""


def format_request_history(requests: Sequence[RequestEvent]) -> str:
    if not requests:
        return "- none"
    lines: List[str] = []
    for req in requests:
        if req.outcome == "matched":
            lines.append(
                f"- request #{req.request_index}: \"{req.request_text}\" -> MATCHED {req.matched_figure}"
            )
        else:
            lines.append(
                f"- request #{req.request_index}: \"{req.request_text}\" -> INVALID ({req.invalid_reason})"
            )
    return "\n".join(lines)


def build_update_turn_prompt(
    case: CanonicalCase,
    requests: Sequence[RequestEvent],
    last_resolution: MatchResolution,
    budget: int,
) -> str:
    history_text = format_request_history(requests)
    patient_provided_block = format_patient_provided_timepoint_block(case, requests)
    if patient_provided_block:
        patient_provided_block = f"\n{patient_provided_block}\n"
    if last_resolution.outcome == "matched" and last_resolution.matched_exam is not None:
        exam = last_resolution.matched_exam
        resolution_block = f"""Previous request result:
- MATCHED exam bundle: {exam.figure}
- Source figures: {', '.join(exam.source_figures or [exam.figure])}
- Newly revealed minimal metadata:
  - modality: {exam.modality or 'unspecified'}
  - acquisition: {exam.acquisition or 'unspecified'}
  - region: {exam.region or 'unspecified'}
  - contrast: {exam.contrast or 'unspecified'}
  - time_past: {exam.time_past or 'unspecified'}

New images for this matched exam are attached to this message.
"""
    else:
        resolution_block = f"""Previous request result:
- INVALID / UNMATCHED request
- Reason: {last_resolution.reason}
- No new exam bundle was revealed.
"""

    return f"""Clinical history reminder:
{case.clinical_history or '[none]'}

Request budget used: {len(requests)} / {budget}
Remember: every request consumes one step, including unavailable, duplicate, or out-of-scope requests.

Request history so far:
{history_text}

{resolution_block}
{patient_provided_block}

Task:
Update your current four-item differential using all evidence seen so far, then choose the next action.
- If you still need evidence, set action=request_exam and request exactly one next exam bundle.
- If you are ready to conclude, set action=stop and provide final_location.

Respond in STRICT JSON only.
"""


def build_forced_stop_prompt(
    case: CanonicalCase,
    requests: Sequence[RequestEvent],
    last_resolution: Optional[MatchResolution],
    budget: int,
) -> str:
    history_text = format_request_history(requests)
    if last_resolution and last_resolution.outcome == "matched" and last_resolution.matched_exam is not None:
        exam = last_resolution.matched_exam
        resolution_block = f"""The final request reached the budget limit and matched:
- {exam.figure}
- Source figures: {', '.join(exam.source_figures or [exam.figure])}
- modality: {exam.modality or 'unspecified'}
- acquisition: {exam.acquisition or 'unspecified'}
- region: {exam.region or 'unspecified'}
- contrast: {exam.contrast or 'unspecified'}
- time_past: {exam.time_past or 'unspecified'}

The corresponding images are attached to this message.
"""
    elif last_resolution is not None:
        resolution_block = f"""The final request reached the budget limit but did not match:
- Reason: {last_resolution.reason}
- No new exam bundle was revealed.
"""
    else:
        resolution_block = "No request-resolution update is available."

    return f"""Clinical history reminder:
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


def normalize_action(value: Any) -> str:
    key = normalize_key(str(value or ""))
    if key in {"request_exam", "request", "order_exam", "next_exam", "continue"}:
        return "request_exam"
    if key in {"stop", "final", "finish", "submit", "conclude"}:
        return "stop"
    return ""


def normalize_location(raw: Any) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return {
            "laterality": "",
            "region": normalize_space(raw),
            "substructure": "",
        }
    if isinstance(raw, dict):
        laterality = normalize_space(
            raw.get("laterality")
            or raw.get("Laterality")
            or raw.get("side")
            or ""
        )
        region = normalize_space(
            raw.get("region")
            or raw.get("Region")
            or raw.get("organ")
            or raw.get("organ_region")
            or raw.get("Organ/Region")
            or ""
        )
        substructure = normalize_space(
            raw.get("substructure")
            or raw.get("sub_structure")
            or raw.get("segment")
            or raw.get("Specific Substructure/Segment")
            or ""
        )
        return {
            "laterality": laterality,
            "region": region,
            "substructure": substructure,
        }
    return None


def normalize_differential(raw: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    warning: Optional[str] = None
    entries: List[Dict[str, Any]] = []

    if isinstance(raw, dict):
        for diagnosis, prob in raw.items():
            if isinstance(diagnosis, str):
                entries.append({"diagnosis": diagnosis, "probability": prob})
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                entries.append({"diagnosis": item, "probability": None})
            elif isinstance(item, dict):
                diagnosis = (
                    item.get("diagnosis")
                    or item.get("dx")
                    or item.get("name")
                    or item.get("candidate")
                    or item.get("label")
                )
                prob = item.get("probability")
                if prob is None:
                    prob = item.get("prob")
                if prob is None:
                    prob = item.get("score")
                if diagnosis:
                    entries.append({"diagnosis": diagnosis, "probability": prob})

    dedup: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for entry in entries:
        diagnosis = normalize_space(entry.get("diagnosis"))
        if not diagnosis:
            continue
        key = normalize_key(diagnosis)
        probability = coerce_float(entry.get("probability"), default=0.0)
        if key not in dedup:
            dedup[key] = {"diagnosis": diagnosis, "probability": probability}
            order.append(key)
        else:
            dedup[key]["probability"] += probability

    normalized = [dedup[key] for key in order]
    normalized.sort(key=lambda item: item["probability"], reverse=True)

    if len(normalized) > 4:
        warning = "trimmed_to_top4"
        normalized = normalized[:4]

    if len(normalized) < 4:
        warning = warning or "padded_to_4"
        while len(normalized) < 4:
            normalized.append(
                {
                    "diagnosis": f"Unspecified candidate {len(normalized) + 1}",
                    "probability": 0.0,
                }
            )

    probs = [coerce_float(item.get("probability"), default=0.0) for item in normalized]
    prob_sum = sum(probs)
    if prob_sum <= 0:
        probs = [1.0, 1.0, 1.0, 1.0]
        warning = warning or "uniform_probabilities_used"
    elif abs(prob_sum - 1.0) > 1e-3:
        warning = warning or "probability_renormalized"

    probs = renormalize_probabilities(probs)
    for item, prob in zip(normalized, probs):
        item["probability"] = round(float(prob), 6)

    return normalized, warning


def build_fallback_turn(forced_stop: bool = False) -> Dict[str, Any]:
    action = "stop" if forced_stop else "request_exam"
    payload: Dict[str, Any] = {
        "action": action,
        "current_differential": [
            {"diagnosis": "Unspecified candidate 1", "probability": 0.25},
            {"diagnosis": "Unspecified candidate 2", "probability": 0.25},
            {"diagnosis": "Unspecified candidate 3", "probability": 0.25},
            {"diagnosis": "Unspecified candidate 4", "probability": 0.25},
        ],
    }
    if action == "request_exam":
        payload["requested_examination"] = ""
    else:
        payload["final_location"] = {"laterality": "", "region": "", "substructure": ""}
    return payload


def normalize_turn_output(raw_text: str, *, forced_stop: bool = False) -> Tuple[Dict[str, Any], Optional[str]]:
    warning: Optional[str] = None
    obj = extract_json_from_text(raw_text)
    if not isinstance(obj, dict):
        warning = "json_parse_failed"
        obj = build_fallback_turn(forced_stop=forced_stop)

    action = normalize_action(obj.get("action"))

    requested_examination = normalize_space(
        obj.get("requested_examination")
        or obj.get("request")
        or obj.get("next_exam")
        or obj.get("exam")
        or ""
    )

    differential, diff_warning = normalize_differential(obj.get("current_differential"))
    if diff_warning:
        warning = diff_warning if warning is None else f"{warning};{diff_warning}"

    final_location = normalize_location(obj.get("final_location") or obj.get("location"))

    if forced_stop:
        action = "stop"
    elif not action:
        if requested_examination and final_location is None:
            action = "request_exam"
            warning = "action_inferred_request" if warning is None else f"{warning};action_inferred_request"
        elif final_location is not None and not requested_examination:
            action = "stop"
            warning = "action_inferred_stop" if warning is None else f"{warning};action_inferred_stop"
        else:
            action = "request_exam"
            warning = "action_missing_default_request" if warning is None else f"{warning};action_missing_default_request"

    if action == "stop" and final_location is None:
        final_location = {"laterality": "", "region": "", "substructure": ""}
        warning = "missing_final_location" if warning is None else f"{warning};missing_final_location"

    if action == "request_exam" and not requested_examination:
        if forced_stop:
            action = "stop"
            final_location = final_location or {"laterality": "", "region": "", "substructure": ""}
            warning = "forced_stop_due_to_missing_request" if warning is None else f"{warning};forced_stop_due_to_missing_request"
        else:
            warning = "missing_request_kept_empty" if warning is None else f"{warning};missing_request_kept_empty"

    normalized = {
        "action": action,
        "current_differential": differential,
    }
    if action == "request_exam":
        normalized["requested_examination"] = requested_examination
    else:
        normalized["final_location"] = final_location or {"laterality": "", "region": "", "substructure": ""}

    return normalized, warning


# =========================
# Core pipeline
# =========================

class JudgeProtocol(Protocol):
    def score_case(
        self,
        case: CanonicalCase,
        final_top1_diagnosis: str,
        final_differential: List[Dict[str, Any]],
        final_location: Dict[str, str],
        trajectory_unique_diagnoses: List[str],
    ) -> Dict[str, Any]:
        ...


class MainBenchmarkPipeline:
    def __init__(
        self,
        *,
        target_session_factory: Callable[[str], SessionProtocol],
        judge_runner: JudgeProtocol,
        request_budget: int = 6,
        random_seed: int = 0,
        trajectory_horizon: Optional[int] = None,
        diagnostic_threshold: float = 2.0 / 3.0,
        reveal_unit: str = DEFAULT_REVEAL_UNIT,
    ) -> None:
        self.target_session_factory = target_session_factory
        self.judge_runner = judge_runner
        self.request_budget = int(request_budget)
        self.trajectory_horizon = int(trajectory_horizon) if trajectory_horizon is not None else int(request_budget) + 2
        self.diagnostic_threshold = float(diagnostic_threshold)
        self.reveal_unit = reveal_unit
        self.random = random.Random(random_seed)

    def run_dataset(
        self,
        raw_cases: Sequence[Dict[str, Any]],
        *,
        image_root: Optional[Path],
    ) -> List[CaseResult]:
        results: List[CaseResult] = []
        for raw_case in raw_cases:
            case = canonicalize_case(raw_case, image_root=image_root, reveal_unit=self.reveal_unit)
            results.append(self.run_case(case))
        return results

    def run_case(self, case: CanonicalCase) -> CaseResult:
        session = self.target_session_factory(AGENT_SYSTEM_PROMPT)
        turns: List[TurnRecord] = []
        requests: List[RequestEvent] = []

        attempted_request_texts: set[str] = set()
        revealed_exam_ids: set[str] = set()

        last_resolution: Optional[MatchResolution] = None
        last_resolution_images: List[ImagePayload] = []

        while True:
            if not turns:
                prompt_kind = "initial"
                prompt_text = build_initial_turn_prompt(case, self.request_budget)
                images = []
                forced_stop = False
            elif len(requests) >= self.request_budget:
                prompt_kind = "forced_stop"
                prompt_text = build_forced_stop_prompt(case, requests, last_resolution, self.request_budget)
                images = last_resolution_images
                forced_stop = True
            else:
                prompt_kind = "after_resolution"
                prompt_text = build_update_turn_prompt(case, requests, last_resolution or MatchResolution(outcome="invalid", reason="missing"), self.request_budget)
                images = last_resolution_images
                forced_stop = False

            model_reply = session.send(prompt_text, images)
            normalized_output, parse_warning = normalize_turn_output(model_reply.text, forced_stop=forced_stop)

            turn = TurnRecord(
                turn_index=len(turns) + 1,
                prompt_kind=prompt_kind,
                prompt_text=prompt_text,
                attached_images=[img.to_dict() for img in images],
                raw_model_text=model_reply.text,
                parsed_output=normalized_output,
                action=normalized_output["action"],
                requested_examination=normalized_output.get("requested_examination"),
                current_differential=normalized_output["current_differential"],
                final_location=normalized_output.get("final_location"),
                parse_warning=parse_warning,
                provider_meta=model_reply.provider_meta or {},
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
                last_resolution_images = resolution.matched_exam.image_payloads
            else:
                last_resolution_images = []

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

        final_turn = turns[-1]
        final_differential = final_turn.current_differential
        final_top1_diagnosis = final_differential[0]["diagnosis"] if final_differential else ""
        final_location = final_turn.final_location or {"laterality": "", "region": "", "substructure": ""}

        trajectory_unique_diagnoses = deduplicate_preserve_order(
            diagnosis
            for turn in turns
            for diagnosis in [item.get("diagnosis") for item in turn.current_differential]
            if isinstance(diagnosis, str) and normalize_space(diagnosis)
        )

        judge_result = self.judge_runner.score_case(
            case=case,
            final_top1_diagnosis=final_top1_diagnosis,
            final_differential=final_differential,
            final_location=final_location,
            trajectory_unique_diagnoses=trajectory_unique_diagnoses,
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
            metrics_by_mode[mode_name] = raw_metrics
            metric_status_by_mode[mode_name] = compute_metric_status(raw_metrics)
            metrics_display_by_mode[mode_name] = build_metrics_display(raw_metrics, metric_status_by_mode[mode_name])
        default_metric_mode = judge_result.get("default_mode") or (next(iter(metrics_by_mode)) if metrics_by_mode else "")
        metrics_raw = metrics_by_mode.get(default_metric_mode, next(iter(metrics_by_mode.values())) if metrics_by_mode else {})
        metrics = metrics_display_by_mode.get(default_metric_mode, next(iter(metrics_display_by_mode.values())) if metrics_display_by_mode else {})

        debug = {
            "official_exam_pool": [exam.minimal_metadata() | {"exam_id": exam.exam_id, "label": exam.label, "preferred_order": exam.preferred_order} for exam in case.official_exam_pool],
            "excluded_exam_pool": [exam.minimal_metadata() | {"exam_id": exam.exam_id, "time_past": exam.time_past} for exam in case.excluded_exam_pool],
            "excluded_reasons_by_exam": case.excluded_reasons_by_exam,
        }

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


def deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_key(value)
        if not key or key in seen:
            continue
        out.append(normalize_space(value))
        seen.add(key)
    return out



def compute_turn_confidence_score(turn: TurnRecord, trajectory_labels: Dict[str, str]) -> float:
    """Confidence-weighted semantic alignment over all four current DDx items.

    This is the old belief/confidence trajectory primitive: probability mass on
    E/A diagnoses is rewarded and probability mass on U diagnoses is penalized.
    Its range is [-1, 1]. It is logged separately from the primary fixed-horizon
    diagnosis-rubric trajectory score S_traj.
    """
    score = 0.0
    for item in turn.current_differential:
        diagnosis = normalize_key(item.get("diagnosis"))
        prob = coerce_float(item.get("probability"), default=0.0)
        label = trajectory_labels.get(diagnosis, "U")
        if label in {"E", "A"}:
            score += prob
        else:
            score -= prob
    return float(score)


def first_differential_item(turn: TurnRecord) -> Dict[str, Any]:
    if not turn.current_differential:
        return {"diagnosis": "", "probability": 0.0}
    # normalize_differential sorts by probability descending, so index 0 is the model's top-1 belief.
    return turn.current_differential[0]


def carry_forward_average(values: Sequence[float], horizon: int, *, empty_value: float = 0.0) -> float:
    horizon = max(1, int(horizon))
    if not values:
        return float(empty_value)
    clipped = [float(v) for v in values[:horizon]]
    if len(clipped) < horizon:
        clipped.extend([float(clipped[-1])] * (horizon - len(clipped)))
    return float(sum(clipped) / horizon)


def build_trajectory_score_maps(judge_result: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    trajectory_labels_list = judge_result.get("trajectory_labels") or []
    trajectory_labels: Dict[str, str] = {}
    for entry in trajectory_labels_list:
        diagnosis = normalize_key(entry.get("diagnosis"))
        label = normalize_space(entry.get("label") or "U").upper()[:1]
        trajectory_labels[diagnosis] = label if label in {"E", "A", "U"} else "U"

    trajectory_scores_list = judge_result.get("trajectory_scores") or []
    trajectory_dx_scores: Dict[str, int] = {}
    for entry in trajectory_scores_list:
        diagnosis = normalize_key(entry.get("diagnosis"))
        try:
            score = int(entry.get("score", 0))
        except Exception:
            score = 0
        trajectory_dx_scores[diagnosis] = max(0, min(3, score))

    # Backward compatibility if an older judge result has labels but not scores.
    for diagnosis, label in trajectory_labels.items():
        if diagnosis not in trajectory_dx_scores:
            trajectory_dx_scores[diagnosis] = 3 if label == "E" else (1 if label == "A" else 0)

    return trajectory_labels, trajectory_dx_scores


def essential_recall_before_turn(
    turn_index: int,
    matched_request_events: Sequence[RequestEvent],
    essential_exam_ids: set[str],
) -> Optional[float]:
    if not essential_exam_ids:
        return None
    revealed_before_turn = {
        req.matched_exam_id
        for req in matched_request_events
        if req.matched_exam_id and req.originating_turn_index < turn_index
    }
    return len(revealed_before_turn & essential_exam_ids) / len(essential_exam_ids)


def compute_case_metrics(
    case: CanonicalCase,
    turns: Sequence[TurnRecord],
    requests: Sequence[RequestEvent],
    judge_result: Dict[str, Any],
    *,
    request_budget: int,
    trajectory_horizon: Optional[int] = None,
    diagnostic_threshold: float = 2.0 / 3.0,
) -> Dict[str, Any]:
    judge_final = judge_result.get("final_scores") or {}
    dx_score_raw = int(coerce_float(((judge_final.get("diagnosis") or {}).get("score")), default=0.0))
    loc_score_raw = int(coerce_float(((judge_final.get("localization") or {}).get("score")), default=0.0))
    ddx_score_raw = int(coerce_float(((judge_final.get("differential_list") or {}).get("score")), default=0.0))

    s_dx = dx_score_raw / 3.0
    s_loc = loc_score_raw / 3.0
    s_ddx = ddx_score_raw / 3.0

    matched_request_events = [req for req in requests if req.outcome == "matched" and req.matched_exam_id]
    invalid_request_events = [req for req in requests if req.outcome != "matched"]
    ambiguous_resolved_events = [req for req in matched_request_events if getattr(req, "ambiguity_resolved", False)]

    matched_exam_ids = {req.matched_exam_id for req in matched_request_events if req.matched_exam_id}
    route_exam_pool = [exam for exam in case.official_exam_pool if getattr(exam, "route_evaluable", True)]
    route_exam_ids = {exam.exam_id for exam in route_exam_pool}
    matched_route_exam_ids = matched_exam_ids & route_exam_ids
    nonroute_matched_exam_ids = matched_exam_ids - route_exam_ids
    essential_exam_ids = {exam.exam_id for exam in route_exam_pool if exam.label == "essential"}
    optional_exam_ids = {exam.exam_id for exam in route_exam_pool if exam.label == "optional"}

    num_optional_requests = len(matched_route_exam_ids & optional_exam_ids)
    s_er = safe_div(len(matched_route_exam_ids & essential_exam_ids), len(essential_exam_ids))
    b_opt = num_optional_requests / max(1, len(matched_route_exam_ids))
    b_inv = len(invalid_request_events) / max(1, len(requests))

    optional_request_rate = b_opt
    nonessential_request_rate_proxy = b_opt

    # order concordance
    request_turn_by_exam: Dict[str, int] = {}
    for req in matched_request_events:
        if req.matched_exam_id and req.matched_exam_id not in request_turn_by_exam:
            request_turn_by_exam[req.matched_exam_id] = req.originating_turn_index

    comparable_pairs: List[Tuple[str, str]] = []
    exams_in_order_eval = [exam_id for exam_id in matched_route_exam_ids if exam_id in case.gold_order_by_exam]
    for i, exam_i in enumerate(exams_in_order_eval):
        for exam_j in exams_in_order_eval[i + 1:]:
            ri = case.gold_order_by_exam[exam_i]
            rj = case.gold_order_by_exam[exam_j]
            if ri < rj:
                comparable_pairs.append((exam_i, exam_j))
            elif rj < ri:
                comparable_pairs.append((exam_j, exam_i))
            else:
                continue

    gold_pair_count = 0
    official_exam_ids_in_order = [exam.exam_id for exam in route_exam_pool if exam.exam_id in case.gold_order_by_exam]
    for i, exam_i in enumerate(official_exam_ids_in_order):
        for exam_j in official_exam_ids_in_order[i + 1:]:
            if case.gold_order_by_exam[exam_i] != case.gold_order_by_exam[exam_j]:
                gold_pair_count += 1

    if comparable_pairs:
        concordant = 0
        for earlier, later in comparable_pairs:
            if request_turn_by_exam.get(earlier, 10**9) < request_turn_by_exam.get(later, 10**9):
                concordant += 1
        s_order = concordant / len(comparable_pairs)
    elif gold_pair_count > 0:
        s_order = 1.0 if s_er == 1.0 else 0.0
    else:
        s_order = None

    trajectory_labels, trajectory_dx_scores = build_trajectory_score_maps(judge_result)

    turn_conf_scores = [compute_turn_confidence_score(turn, trajectory_labels) for turn in turns]
    s_traj_conf_actual = sum(turn_conf_scores) / max(1, len(turn_conf_scores))
    s_conf = turn_conf_scores[-1] if turn_conf_scores else -1.0

    t_max = int(trajectory_horizon) if trajectory_horizon is not None else int(request_budget) + 2
    t_max = max(1, t_max)
    tau = clamp(float(diagnostic_threshold), 0.0, 1.0)

    turn_top1_scores_raw: List[int] = []
    turn_top1_scores_norm: List[float] = []
    turn_er_before: List[Optional[float]] = []
    trajectory_turn_details: List[Dict[str, Any]] = []

    for turn in turns:
        top1 = first_differential_item(turn)
        top1_dx = normalize_space(top1.get("diagnosis") or "")
        top1_key = normalize_key(top1_dx)
        raw_score = max(0, min(3, int(trajectory_dx_scores.get(top1_key, 0))))
        norm_score = raw_score / 3.0
        er_before = essential_recall_before_turn(turn.turn_index, matched_request_events, essential_exam_ids)
        turn_top1_scores_raw.append(raw_score)
        turn_top1_scores_norm.append(norm_score)
        turn_er_before.append(er_before)
        trajectory_turn_details.append({
            "turn_index": turn.turn_index,
            "top1_diagnosis": top1_dx,
            "top1_probability": coerce_float(top1.get("probability"), default=0.0),
            "top1_dx_score_raw": raw_score,
            "top1_dx_score_norm": norm_score,
            "essential_recall_before_turn": er_before,
            "confidence_alignment_score": turn_conf_scores[turn.turn_index - 1] if turn.turn_index - 1 < len(turn_conf_scores) else None,
        })

    # Primary trajectory score: fixed-horizon, diagnosis-rubric top-1 trajectory with carry-forward.
    # This is in [0, 1] and rewards reaching a correct/stable diagnosis early.
    s_traj_dx_actual = sum(turn_top1_scores_norm) / max(1, len(turn_top1_scores_norm))
    s_traj_dx_tmax = carry_forward_average(turn_top1_scores_norm, t_max, empty_value=0.0)
    s_traj = s_traj_dx_tmax

    # Also log fixed-horizon version of the older confidence-weighted semantic trajectory.
    s_traj_conf_tmax = carry_forward_average(turn_conf_scores, t_max, empty_value=-1.0)

    time_not_reached_value = t_max + 1
    time_to_diagnostic_guess: int = time_not_reached_value
    for turn, score in zip(turns, turn_top1_scores_norm):
        if score >= tau:
            time_to_diagnostic_guess = turn.turn_index
            break

    time_to_clinically_acceptable_diagnosis: int = time_not_reached_value
    for turn, score, er_before in zip(turns, turn_top1_scores_norm, turn_er_before):
        er_condition = True if er_before is None else (er_before >= 1.0)
        if score >= tau and er_condition:
            time_to_clinically_acceptable_diagnosis = turn.turn_index
            break

    worst_case_time_to_hit4 = int(request_budget) + 2
    time_to_hit4: Optional[int] = None
    for turn in turns:
        labels_this_turn = [
            trajectory_labels.get(normalize_key(item.get("diagnosis")), "U")
            for item in turn.current_differential
        ]
        if "E" in labels_this_turn:
            time_to_hit4 = turn.turn_index
            break
    if time_to_hit4 is None:
        time_to_hit4 = worst_case_time_to_hit4

    final_probabilities = [coerce_float(item.get("probability"), default=0.0) for item in (turns[-1].current_differential if turns else [])]
    p_max = max(final_probabilities) if final_probabilities else 0.0
    brier_top1 = (p_max - s_dx) ** 2

    forced_stop_used = bool(turns and turns[-1].prompt_kind == "forced_stop")
    stopped_by_model = bool(turns and turns[-1].action == "stop" and turns[-1].prompt_kind != "forced_stop")

    return {
        "S_dx": s_dx,
        "S_loc": s_loc,
        "S_ddx": s_ddx,
        "S_ER": s_er,
        "B_opt": b_opt,
        "B_inv": b_inv,
        "S_order": s_order,
        "S_traj": s_traj,
        "S_traj_dx_Tmax": s_traj_dx_tmax,
        "S_traj_dx_actual": s_traj_dx_actual,
        "S_traj_conf_actual": s_traj_conf_actual,
        "S_traj_conf_Tmax": s_traj_conf_tmax,
        "S_conf": s_conf,
        "Brier_top1": brier_top1,
        "time_to_hit4": time_to_hit4,
        "time_to_diagnostic_guess": time_to_diagnostic_guess,
        "time_to_clinically_acceptable_diagnosis": time_to_clinically_acceptable_diagnosis,
        "num_requests": len(requests),
        "num_turns": len(turns),
        "num_matched_requests": len(matched_request_events),
        "num_invalid_requests": len(invalid_request_events),
        "num_ambiguous_resolved_requests": len(ambiguous_resolved_events),
        "num_route_matched_requests": len(matched_route_exam_ids),
        "num_nonroute_matched_requests": len(nonroute_matched_exam_ids),
        "num_optional_requests": num_optional_requests,
        "optional_request_rate": optional_request_rate,
        "nonessential_request_rate_proxy": nonessential_request_rate_proxy,
        "p_max": p_max,
        "order_num_pairs": len(comparable_pairs),
        "order_gold_pair_count": gold_pair_count,
        "essential_exam_count": len(essential_exam_ids),
        "official_exam_count": len(case.official_exam_pool),
        "route_evaluable_exam_count": len(route_exam_pool),
        "nonroute_evidence_count": len(case.official_exam_pool) - len(route_exam_pool),
        "excluded_exam_count": len(case.excluded_exam_pool),
        "trajectory_horizon": t_max,
        "diagnostic_threshold": tau,
        "time_not_reached_value": time_not_reached_value,
        "forced_stop_used": forced_stop_used,
        "stopped_by_model": stopped_by_model,
        "trajectory_turn_details": trajectory_turn_details,
    }


def compute_metric_status(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    non_score_fields = {
        "order_num_pairs", "order_gold_pair_count", "essential_exam_count", "official_exam_count",
        "route_evaluable_exam_count", "nonroute_evidence_count", "excluded_exam_count",
        "num_matched_requests", "num_invalid_requests", "num_ambiguous_resolved_requests", "num_route_matched_requests", "num_nonroute_matched_requests", "num_optional_requests",
        "p_max", "num_requests", "num_turns", "trajectory_horizon", "diagnostic_threshold",
        "time_not_reached_value", "forced_stop_used", "stopped_by_model", "trajectory_turn_details",
    }
    for metric_name, value in metrics.items():
        if metric_name in non_score_fields or isinstance(value, (list, dict, bool)):
            continue
        if value is not None:
            status[metric_name] = {"defined": True, "reason": "defined"}
            continue
        reason = "undefined"
        if metric_name == "S_ER":
            if int(metrics.get("essential_exam_count") or 0) == 0:
                reason = "no_official_essential_exams"
        elif metric_name == "S_order":
            if int(metrics.get("order_gold_pair_count") or 0) == 0:
                reason = "no_gold_order_pairs"
        status[metric_name] = {"defined": False, "reason": reason}
    return status


def build_metrics_display(metrics: Dict[str, Any], metric_status: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if value is not None:
            out[key] = value
        elif key in metric_status:
            out[key] = 0.0
        else:
            out[key] = value
    return out


def build_dataset_preflight(cases: Sequence[CanonicalCase]) -> Dict[str, Any]:
    per_case: List[Dict[str, Any]] = []
    total_official_exams = 0
    total_excluded_exams = 0
    total_missing_image_exams = 0
    total_followup_exams = 0
    total_future_followup_excluded_exams = 0
    total_patient_provided_prior_exams = 0
    for case in cases:
        official_count = len(case.official_exam_pool)
        excluded_count = len(case.excluded_exam_pool)
        route_count = sum(1 for exam in case.official_exam_pool if getattr(exam, "route_evaluable", True))
        nonroute_count = official_count - route_count
        essential_count = sum(1 for exam in case.official_exam_pool if exam.label == "essential" and getattr(exam, "route_evaluable", True))
        missing_image_count = sum(1 for _, reason in case.excluded_reasons_by_exam.items() if reason == "missing_local_images")
        followup_count = sum(1 for exam in case.official_exam_pool if is_exam_bundle_followup_like(exam))
        future_followup_excluded_count = sum(1 for _, reason in case.excluded_reasons_by_exam.items() if reason == "future_followup_time_past")
        patient_provided_prior_count = sum(1 for exam in case.official_exam_pool if is_exam_bundle_prior_timepoint(exam))
        total_official_exams += official_count
        total_excluded_exams += excluded_count
        total_missing_image_exams += missing_image_count
        total_followup_exams += followup_count
        total_future_followup_excluded_exams += future_followup_excluded_count
        total_patient_provided_prior_exams += patient_provided_prior_count
        per_case.append({
            "case_id": case.case_id,
            "official_exam_count": official_count,
            "route_evaluable_exam_count": route_count,
            "nonroute_evidence_count": nonroute_count,
            "excluded_exam_count": excluded_count,
            "essential_exam_count": essential_count,
            "missing_local_image_exams": missing_image_count,
            "followup_or_delayed_available_exams": followup_count,
            "patient_provided_prior_available_exams": patient_provided_prior_count,
            "followup_excluded_exams": future_followup_excluded_count,
        })

    issues: List[Dict[str, Any]] = []
    zero_official_cases = sum(1 for x in per_case if x["official_exam_count"] == 0)
    zero_essential_cases = sum(1 for x in per_case if x["essential_exam_count"] == 0)
    if total_official_exams == 0 and total_missing_image_exams > 0:
        issues.append({
            "severity": "error",
            "code": "all_official_exams_missing_local_images",
            "message": "All official exams were excluded because no local image files could be resolved. Check --image-root and relative image_paths handling.",
        })
    elif zero_official_cases > 0:
        issues.append({
            "severity": "warning",
            "code": "some_cases_have_zero_official_exams",
            "message": "Some cases have zero official exams after exclusions. Route metrics for those cases are structurally unavailable.",
            "n_cases": zero_official_cases,
        })
    if zero_essential_cases > 0:
        issues.append({
            "severity": "warning",
            "code": "some_cases_have_zero_official_essential_exams",
            "message": "Some cases have zero official essential exams after exclusions. S_ER is undefined for those cases and should not be mistaken for model failure.",
            "n_cases": zero_essential_cases,
        })

    return {
        "n_cases": len(cases),
        "total_official_exams": total_official_exams,
        "total_excluded_exams": total_excluded_exams,
        "total_missing_local_image_exams": total_missing_image_exams,
        "total_followup_or_delayed_available_exams": total_followup_exams,
        "total_patient_provided_prior_available_exams": total_patient_provided_prior_exams,
        "total_followup_excluded_exams": total_future_followup_excluded_exams,
        "cases_with_zero_official_exams": zero_official_cases,
        "cases_with_zero_essential_exams": zero_essential_cases,
        "issues": issues,
        "per_case": per_case,
    }


def build_metric_notes() -> Dict[str, Any]:
    return {
        "null_and_worst_case_policy": {
            "scored_reveal_unit": "Default reveal unit is eurorad: each EuroRad imaging_examination / figure-protocol entry is one requestable evidence bundle. No routine T1/T2/FLAIR/STIR/DWI sequence merging is performed.",
            "followup_policy": "Numeric time_past < 0 denotes future follow-up imaging and is excluded from the requestable/evaluable official pool. Numeric time_past > 0 denotes prior patient-provided imaging and is listed as available on request. Null or nonnumeric time_past does not by itself expose or exclude an exam.",
            "route_denominator_policy": "Route metrics S_ER/S_order/Opt_burden are computed over route-evaluable radiology/imaging bundles only. Non-imaging evidence such as histopathology, clinical photo, or illustration is retained in debug/provenance but excluded from imaging-route denominators.",
            "ambiguous_request_policy": "Ambiguous/broad requests are not penalized as invalid when at least one eligible candidate scores above threshold. The resolver reveals the best-scoring eligible EuroRad-style bundle, counts the request as matched, and logs ambiguity_resolved/candidate_scores for audit.",
            "S_order": "If matched comparable pairs exist, use pairwise concordance. If gold comparable pairs exist but no matched comparable pairs exist, assign 0 when S_ER < 1 and 1 when S_ER == 1. If no gold comparable pairs exist, S_order is structurally undefined.",
            "S_traj": "Primary S_traj is fixed-horizon top-1 diagnosis-rubric trajectory: average S_dx^(t) over T_max turns with final answer carried forward after early stopping. Default T_max=B+2 unless --trajectory-horizon is set.",
            "S_traj_conf_actual": "Older confidence-weighted semantic belief trajectory over actual turns only is still logged separately. It uses E/A/U labels and can be negative.",
            "time_to_hit4": "If the exact diagnosis never appears anywhere in the four-way differential, assign the fixed worst-case value B+2.",
            "time_to_diagnostic_guess": "Earliest actual turn where the top-1 diagnosis reaches the normalized diagnosis-rubric threshold tau. If never reached, assign T_max+1.",
            "time_to_clinically_acceptable_diagnosis": "Earliest actual turn where top-1 diagnosis score >= tau and EssentialRecall before that turn equals 1. If no essential exams exist, the route condition is treated as vacuously satisfied for this timing metric only. If never reached, assign T_max+1.",
            "S_ER": "If a case has zero route-evaluable official essential exams, S_ER is intrinsically undefined. The exported display value is 0 with defined=false and reason=no_official_essential_exams.",
            "nonessential_request_rate_proxy": "No annotator used an explicit unnecessary label in the current data. optional_request_rate / nonessential_request_rate_proxy therefore uses optional-labeled matched route-evaluable exams as a conservative proxy; it should not be described as unnecessary imaging.",
            "reliability_bins": "Empty calibration bins are intrinsically undefined. They are exported with count=0, empty_bin=true, defined=false, and avg_p_max/avg_S_dx=null. They should be omitted from calibration plots and ECE-style summaries rather than imputed as 0.",
        },
        "metric_ranges": {
            "S_dx": [0.0, 1.0],
            "S_loc": [0.0, 1.0],
            "S_ddx": [0.0, 1.0],
            "S_ER": [0.0, 1.0],
            "B_opt": [0.0, 1.0],
            "B_inv": [0.0, 1.0],
            "S_order": [0.0, 1.0],
            "S_traj": [0.0, 1.0],
            "S_traj_dx_Tmax": [0.0, 1.0],
            "S_traj_dx_actual": [0.0, 1.0],
            "S_traj_conf_actual": [-1.0, 1.0],
            "S_traj_conf_Tmax": [-1.0, 1.0],
            "S_conf": [-1.0, 1.0],
            "optional_request_rate": [0.0, 1.0],
            "nonessential_request_rate_proxy": [0.0, 1.0],
            "Brier_top1": [0.0, 1.0],
        },
        "calibration_notes": {
            "p_max": "Final-turn top-1 probability after renormalizing the four reported probabilities to sum to 1.",
            "avg_p_max": "Within a reliability bin, the mean of p_max over cases assigned to that bin. Undefined (null) when the bin is empty.",
            "avg_S_dx": "Within a reliability bin, the mean realized normalized diagnosis score S_dx over the same cases. Undefined (null) when the bin is empty.",
            "interpretation": "Calibration compares claimed confidence (avg_p_max) against realized diagnosis quality (avg_S_dx). Overconfidence means avg_p_max > avg_S_dx; underconfidence means avg_p_max < avg_S_dx.",
            "difference_from_S_conf": "S_conf is not a calibration metric. It is the final-turn confidence-weighted semantic alignment score over the full four-item differential, using judge labels E/A/U and allowing negative values.",
        },
        "scoring_modes": {
            "llm": "Single-call LLM judge using case-specific rubrics and a global DDx rubric. By default, it prefers provider-native structured outputs with automatic fallback to prompt-constrained JSON when native structured outputs are unavailable.",
            "rule": "Versioned deterministic scorer using normalization, stricter option matching, rubric-derived alias buckets, MCQ/reference-DDx matching, negation handling, and component-wise localization rules.",
        },
    }


# =========================
# Aggregation for Table 2+
# =========================

def mean_ignore_none(values: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def bootstrap_mean_ci(
    values: Sequence[Optional[float]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    undefined_reason: str = "no_evaluable_cases",
) -> Dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "defined": False,
            "reason": undefined_reason,
        }
    if len(xs) == 1:
        return {
            "mean": xs[0],
            "ci_low": xs[0],
            "ci_high": xs[0],
            "defined": True,
            "reason": "defined",
        }
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [xs[rng.randrange(len(xs))] for _ in xs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": sum(xs) / len(xs),
        "ci_low": percentile(means, 0.025),
        "ci_high": percentile(means, 0.975),
        "defined": True,
        "reason": "defined",
    }


def compute_reliability_bins(case_results: Sequence[CaseResult], num_bins: int = 10) -> List[Dict[str, Any]]:
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(num_bins)]
    for result in case_results:
        p_max = float(result.metrics.get("p_max") or 0.0)
        target = float(result.metrics.get("S_dx") or 0.0)
        idx = min(num_bins - 1, max(0, int(p_max * num_bins)))
        bins[idx].append((p_max, target))

    out: List[Dict[str, Any]] = []
    for idx, values in enumerate(bins):
        lo = idx / num_bins
        hi = (idx + 1) / num_bins
        if values:
            avg_conf = sum(v[0] for v in values) / len(values)
            avg_target = sum(v[1] for v in values) / len(values)
            empty_bin = False
        else:
            avg_conf = None
            avg_target = None
            empty_bin = True
        out.append(
            {
                "bin_index": idx,
                "bin_left": lo,
                "bin_right": hi,
                "count": len(values),
                "avg_p_max": avg_conf,
                "avg_S_dx": avg_target,
                "empty_bin": empty_bin,
                "defined": not empty_bin,
            }
        )
    return out


def aggregate_results(
    case_results: Sequence[CaseResult],
    *,
    model_label: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    metric_names = [
        "S_dx",
        "S_loc",
        "S_ddx",
        "S_ER",
        "B_opt",
        "B_inv",
        "S_order",
        "S_traj",
        "S_traj_dx_Tmax",
        "S_traj_dx_actual",
        "S_traj_conf_actual",
        "S_traj_conf_Tmax",
        "S_conf",
        "Brier_top1",
        "time_to_hit4",
        "time_to_diagnostic_guess",
        "time_to_clinically_acceptable_diagnosis",
        "optional_request_rate",
        "nonessential_request_rate_proxy",
        "num_requests",
        "num_turns",
        "num_ambiguous_resolved_requests",
    ]

    per_metric_values: Dict[str, List[Optional[float]]] = {
        name: [result.metrics.get(name) for result in case_results]
        for name in metric_names
    }

    def _undefined_reason(metric_name: str) -> str:
        reasons: Dict[str, int] = {}
        for result in case_results:
            metric_status = getattr(result, "metric_status", {}) or {}
            entry = metric_status.get(metric_name) or {}
            if entry.get("defined", True):
                continue
            reason = normalize_space(entry.get("reason") or "undefined") or "undefined"
            reasons[reason] = reasons.get(reason, 0) + 1
        if not reasons:
            return "no_evaluable_cases"
        return max(reasons.items(), key=lambda item: item[1])[0]

    means_raw = {name: mean_ignore_none(values) for name, values in per_metric_values.items()}
    means = {name: (means_raw[name] if means_raw[name] is not None else 0.0) for name in metric_names}
    metric_status = {
        name: {
            "defined": means_raw[name] is not None,
            "n_defined": sum(1 for v in per_metric_values[name] if v is not None),
            "n_total": len(case_results),
            "undefined_reason": _undefined_reason(name) if means_raw[name] is None else "defined",
        }
        for name in metric_names
    }
    cis = {
        name: bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
            undefined_reason=metric_status[name]["undefined_reason"],
        )
        for idx, (name, values) in enumerate(per_metric_values.items())
    }

    subgroup_metrics: Dict[str, Any] = {}
    for field_name in ["difficulty", "rarity", "section", "area_of_interest"]:
        buckets: Dict[str, List[CaseResult]] = {}
        for result in case_results:
            key = getattr(result, field_name) or "UNKNOWN"
            buckets.setdefault(key, []).append(result)
        subgroup_metrics[field_name] = {
            key: {
                "n_cases": len(results),
                "S_dx": mean_ignore_none([r.metrics.get("S_dx") for r in results]) or 0.0,
                "S_ER": mean_ignore_none([r.metrics.get("S_ER") for r in results]) or 0.0,
                "S_traj": mean_ignore_none([r.metrics.get("S_traj") for r in results]) or 0.0,
            }
            for key, results in sorted(buckets.items(), key=lambda item: item[0])
        }

    table2_row = {
        "model": model_label,
        "S_dx": means["S_dx"],
        "S_loc": means["S_loc"],
        "S_ddx": means["S_ddx"],
        "S_ER": means["S_ER"],
        "Opt_burden": means["B_opt"],
        "Inv_req_rate": means["B_inv"],
        "S_order": means["S_order"],
        "S_traj": means["S_traj"],
        "S_conf": means["S_conf"],
    }
    table2_row_raw = {
        "model": model_label,
        "S_dx": means_raw["S_dx"],
        "S_loc": means_raw["S_loc"],
        "S_ddx": means_raw["S_ddx"],
        "S_ER": means_raw["S_ER"],
        "Opt_burden": means_raw["B_opt"],
        "Inv_req_rate": means_raw["B_inv"],
        "S_order": means_raw["S_order"],
        "S_traj": means_raw["S_traj"],
        "S_conf": means_raw["S_conf"],
    }
    table2_row_status = {
        "S_dx": metric_status["S_dx"],
        "S_loc": metric_status["S_loc"],
        "S_ddx": metric_status["S_ddx"],
        "S_ER": metric_status["S_ER"],
        "Opt_burden": metric_status["B_opt"],
        "Inv_req_rate": metric_status["B_inv"],
        "S_order": metric_status["S_order"],
        "S_traj": metric_status["S_traj"],
        "S_conf": metric_status["S_conf"],
    }

    extras = {
        "Brier_top1": means["Brier_top1"],
        "time_to_hit4": means["time_to_hit4"],
        "time_to_diagnostic_guess": means["time_to_diagnostic_guess"],
        "time_to_clinically_acceptable_diagnosis": means["time_to_clinically_acceptable_diagnosis"],
        "avg_requests": means["num_requests"],
        "avg_turns": means["num_turns"],
        "S_traj_dx_actual": means["S_traj_dx_actual"],
        "S_traj_conf_actual": means["S_traj_conf_actual"],
        "S_traj_conf_Tmax": means["S_traj_conf_Tmax"],
        "optional_request_rate": means["optional_request_rate"],
        "nonessential_request_rate_proxy": means["nonessential_request_rate_proxy"],
        "n_cases": len(case_results),
        "order_evaluable_cases": sum(1 for r in case_results if r.metrics.get("S_order") is not None),
        "essential_recall_evaluable_cases": sum(1 for r in case_results if r.metrics.get("S_ER") is not None),
    }

    return {
        "table2_row": table2_row,
        "table2_row_raw": table2_row_raw,
        "table2_row_status": table2_row_status,
        "metric_means": means,
        "metric_means_raw": means_raw,
        "metric_status": metric_status,
        "metric_bootstrap_ci": cis,
        "reliability_bins": compute_reliability_bins(case_results),
        "subgroup_metrics": subgroup_metrics,
        "extras": extras,
    }
