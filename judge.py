from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple


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
    return normalize_space(text).lower()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_json_from_text(text: str) -> Optional[Any]:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

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


# =========================
# LLM judge prompt
# =========================

JUDGE_SYSTEM_PROMPT = """You are a strict clinical benchmark judge for multimodal differential diagnosis.
Return STRICT JSON only. Do not use markdown or extra keys.

You will score:
1) final diagnosis quality using the provided case-specific diagnosis rubric;
2) final localization quality using the provided case-specific localization rubric;
3) final four-item differential-list quality using the reference differential set and the global rubric below;
4) exact/acceptable/unmatched labels and 0-3 diagnosis-rubric scores for every diagnosis string in the trajectory.

Global rubric for final differential-list quality (0-3):
- 0: The list is mostly off-target, fails to include the final diagnosis or close equivalent, and has little overlap with the reference differential set.
- 1: The list contains one or more accepted-but-not-gold items, but coverage/ranking is weak and the list does not function as a strong clinical differential.
- 2: The list includes the final diagnosis or a near-gold diagnosis, but coverage or ranking is incomplete/suboptimal.
- 3: The correct diagnosis is prominent, and the remaining items are largely aligned with reference_differential_options or explicit case-rubric examples.

Trajectory labels and scores:
- E (exact): reserved exclusively for the gold final diagnosis concept, including close lexical variants or true near-synonyms of the gold diagnosis. Do NOT label a non-gold reference differential option as E.
- A (acceptable): not exact, but accepted for this case because it matches reference_differential_options, is a close synonym of such an option, or would receive score 1 or 2 under the case-specific diagnosis rubric.
- U (unmatched): not exact and not accepted by the reference differential set or diagnosis rubric. Do not mark a diagnosis A merely because it is generically clinically plausible.
- trajectory_scores.score must use the same case-specific 0-3 diagnosis rubric as the final diagnosis score. It is used to compute diagnostic-trajectory metrics, so score every unique trajectory diagnosis as if it were the model's current top-1 diagnosis.

Be conservative, concise, and consistent.
"""


class JudgeCallerProtocol(Protocol):
    def __call__(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        ...


def build_judge_prompt(
    *,
    case_payload: Dict[str, Any],
    model_payload: Dict[str, Any],
) -> str:
    return f"""Score the model output for this case.

CASE PAYLOAD
{json_dumps(case_payload)}

MODEL PAYLOAD
{json_dumps(model_payload)}

Instructions:
- Use the provided case-specific diagnosis rubric to assign diagnosis.score in {{0,1,2,3}}.
- Use the provided case-specific localization rubric to assign localization.score in {{0,1,2,3}}.
- Use the global differential-list rubric from the system prompt to assign differential_list.score in {{0,1,2,3}}.
- For the differential-list score, treat reference_differential_options as the reference differential set, supplemented only by explicit examples in the case-specific diagnosis rubric. The model list is final_differential.
- Penalize non-reference diagnoses even if they are generically plausible, unless the case-specific diagnosis rubric would clearly award them score 1 or 2.
- For each unique diagnosis string in trajectory_unique_diagnoses, assign exactly one label: E, A, or U. E is only for the gold final diagnosis concept.
- For each unique diagnosis string in trajectory_unique_diagnoses, also assign trajectory_scores.score in {0,1,2,3} using the provided case-specific diagnosis rubric.
- Provide a brief reason (<=30 words) for each final score and each trajectory score.
- Every diagnosis from trajectory_unique_diagnoses must appear exactly once in trajectory_labels and exactly once in trajectory_scores.

Return STRICT JSON only with this schema:
{{
  "final_scores": {{
    "diagnosis": {{"score": 0, "reason": "string"}},
    "localization": {{"score": 0, "reason": "string"}},
    "differential_list": {{"score": 0, "reason": "string"}}
  }},
  "trajectory_labels": [
    {{"diagnosis": "string", "label": "E|A|U", "reason": "string"}}
  ],
  "trajectory_scores": [
    {{"diagnosis": "string", "score": 0, "reason": "string"}}
  ]
}}
"""


def normalize_score_entry(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"score": 0, "reason": ""}
    try:
        score = int(raw.get("score", 0))
    except Exception:
        score = 0
    score = max(0, min(3, score))
    return {
        "score": score,
        "reason": normalize_space(raw.get("reason") or ""),
    }


def normalize_judge_output(raw_obj: Any, trajectory_unique_diagnoses: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(raw_obj, dict):
        raw_obj = {}

    final_scores = raw_obj.get("final_scores") or {}
    normalized = {
        "final_scores": {
            "diagnosis": normalize_score_entry(final_scores.get("diagnosis")),
            "localization": normalize_score_entry(final_scores.get("localization")),
            "differential_list": normalize_score_entry(final_scores.get("differential_list")),
        },
        "trajectory_labels": [],
        "trajectory_scores": [],
    }

    labels_raw = raw_obj.get("trajectory_labels") or []
    label_map: Dict[str, Dict[str, Any]] = {}
    for entry in labels_raw:
        if not isinstance(entry, dict):
            continue
        diagnosis = normalize_space(entry.get("diagnosis") or "")
        if not diagnosis:
            continue
        key = normalize_key(diagnosis)
        label = normalize_space(entry.get("label") or "U").upper()[:1]
        if label not in {"E", "A", "U"}:
            label = "U"
        label_map[key] = {
            "diagnosis": diagnosis,
            "label": label,
            "reason": normalize_space(entry.get("reason") or ""),
        }

    scores_raw = raw_obj.get("trajectory_scores") or []
    score_map: Dict[str, Dict[str, Any]] = {}
    for entry in scores_raw:
        if not isinstance(entry, dict):
            continue
        diagnosis = normalize_space(entry.get("diagnosis") or "")
        if not diagnosis:
            continue
        key = normalize_key(diagnosis)
        try:
            score = int(entry.get("score", 0))
        except Exception:
            score = 0
        score = max(0, min(3, score))
        score_map[key] = {
            "diagnosis": diagnosis,
            "score": score,
            "reason": normalize_space(entry.get("reason") or ""),
        }

    for diagnosis in deduplicate_preserve_order(trajectory_unique_diagnoses):
        key = normalize_key(diagnosis)
        label_entry = label_map.get(
            key,
            {"diagnosis": diagnosis, "label": "U", "reason": ""},
        )
        normalized["trajectory_labels"].append(label_entry)
        if key in score_map:
            normalized["trajectory_scores"].append(score_map[key])
        else:
            # Backward-compatible fallback for prompt/JSON failures: E=3, A=1, U=0.
            fallback_score = 3 if label_entry.get("label") == "E" else (1 if label_entry.get("label") == "A" else 0)
            normalized["trajectory_scores"].append({
                "diagnosis": diagnosis,
                "score": fallback_score,
                "reason": "fallback score inferred from E/A/U label",
            })

    return normalized


# =========================
# LLM judge structured-output schema
# =========================

JUDGE_JSON_SCHEMA_NAME = "benchmark_judge_result"
JUDGE_JSON_SCHEMA_DESCRIPTION = "Structured case-level scoring result with final scores and trajectory diagnosis scores for the EuroRad main benchmark judge."
JUDGE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "final_scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "diagnosis": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                },
                "localization": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                },
                "differential_list": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                },
            },
            "required": ["diagnosis", "localization", "differential_list"],
        },
        "trajectory_labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "diagnosis": {"type": "string"},
                    "label": {"type": "string", "enum": ["E", "A", "U"]},
                    "reason": {"type": "string"},
                },
                "required": ["diagnosis", "label", "reason"],
            },
        },
        "trajectory_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "diagnosis": {"type": "string"},
                    "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "reason": {"type": "string"},
                },
                "required": ["diagnosis", "score", "reason"],
            },
        },
    },
    "required": ["final_scores", "trajectory_labels", "trajectory_scores"],
}


# =========================
# Deterministic scorer
# =========================

BRITISH_AMERICAN = {
    "tumour": "tumor",
    "tumours": "tumors",
    "haemorrhage": "hemorrhage",
    "haemorrhagic": "hemorrhagic",
    "haematoma": "hematoma",
    "oedema": "edema",
    "oedematous": "edematous",
    "ischaemia": "ischemia",
    "ischaemic": "ischemic",
    "leukaemia": "leukemia",
    "paediatric": "pediatric",
    "foetal": "fetal",
    "foetus": "fetus",
    "behaviour": "behavior",
    "centre": "center",
    "grey": "gray",
    "artefact": "artifact",
    "neuroaxis": "neuraxis",
    "haemangioblastoma": "hemangioblastoma",
    "haemangioma": "hemangioma",
    "haemangiomas": "hemangiomas",
    "haemangioendothelioma": "hemangioendothelioma",
}

TERM_CANONICAL_REPLACEMENTS = [
    (r"\bgas embolism\b", "air embolism"),
    (r"\bair emboli\b", "air embolism"),
    (r"\bgas emboli\b", "air embolism"),
    (r"\bdecompression sickness\b", "decompression illness"),
    (r"barotrauma/decompression illness", "barotrauma decompression illness"),
    (r"\bcaa[\s\-]*ri\b", "cerebral amyloid angiopathy related inflammation"),
    (r"\bposterior reversible encephalopathy\b", "posterior reversible encephalopathy syndrome"),
    (r"\bpml\b", "progressive multifocal leukoencephalopathy"),
]

FILLER_PHRASES = [
    "most likely",
    "likely",
    "possible",
    "probably",
    "probable",
    "suggestive of",
    "consistent with",
    "compatible with",
    "concerning for",
    "suspicious for",
    "favor",
    "favours",
    "favored",
    "favoured",
    "diagnosis of",
    "evidence of",
    "features of",
    "appearance of",
    "representing",
    "reflecting",
]

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bunlikely\b",
    r"\brule(?:s)? out\b",
    r"\bexclude(?:s|d|ing)?\b",
    r"\bnegative for\b",
    r"\babsence of\b",
]

MULTI_DX_SPLIT_RE = re.compile(r"\b(?:vs\.?|versus|or|and/or)\b|/|;|\|", flags=re.I)

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "with",
    "without",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "due",
    "related",
    "associated",
    "secondary",
    "primary",
    "most",
    "likely",
    "possible",
    "probable",
    "disease",
    "disorder",
    "syndrome",
    "process",
    "lesion",
}

QUALIFIER_TOKENS = {
    "acute",
    "subacute",
    "chronic",
    "bilateral",
    "unilateral",
    "left",
    "right",
    "anterior",
    "posterior",
    "medial",
    "lateral",
    "rostral",
    "caudal",
    "upper",
    "lower",
    "distal",
    "proximal",
    "adult",
    "pediatric",
    "paediatric",
    "mild",
    "severe",
    "marked",
    "focal",
    "diffuse",
    "isolated",
    "central",
    "peripheral",
    "segmental",
    "lobar",
    "subcortical",
    "cortical",
}

DISEASE_FAMILY_MAP = {
    "infarct": "stroke",
    "infarction": "stroke",
    "stroke": "stroke",
    "ischemia": "stroke",
    "ischemic": "stroke",
    "bleed": "hemorrhage",
    "hemorrhage": "hemorrhage",
    "hemorrhagic": "hemorrhage",
    "hematoma": "hemorrhage",
    "tumor": "neoplasm",
    "tumors": "neoplasm",
    "neoplasm": "neoplasm",
    "neoplasia": "neoplasm",
    "mass": "neoplasm",
    "metastasis": "neoplasm",
    "metastatic": "neoplasm",
    "lymphoma": "neoplasm",
    "glioma": "neoplasm",
    "glioblastoma": "neoplasm",
    "medulloblastoma": "neoplasm",
    "meningioma": "neoplasm",
    "infection": "infection",
    "infectious": "infection",
    "encephalitis": "infection",
    "cerebritis": "infection",
    "abscess": "infection",
    "vasculitis": "vasculitis",
    "demyelinating": "demyelination",
    "demyelination": "demyelination",
    "myelitis": "demyelination",
    "encephalopathy": "encephalopathy",
}

GENERIC_REFERENCE_OPTION_TOKENS = {
    "diagnosis",
    "finding",
    "findings",
    "abnormality",
    "abnormalities",
    "stroke",
    "infarct",
    "infarction",
    "ischemia",
    "ischemic",
    "hemorrhage",
    "hemorrhagic",
    "hematoma",
    "tumor",
    "tumors",
    "neoplasm",
    "lesion",
    "mass",
    "embolism",
    "injury",
    "change",
    "changes",
    "syndrome",
    "process",
    "disease",
    "disorder",
}

REGION_STOPWORDS = {
    "organ",
    "region",
    "specific",
    "segment",
    "substructure",
    "substructures",
    "segmental",
    "area",
    "level",
    "side",
}

GENERIC_SUBSTRUCTURE_TOKENS = {
    "brain",
    "brainstem",
    "cerebrum",
    "cerebral",
    "cerebellum",
    "cerebellar",
    "posterior",
    "fossa",
    "posteriorfossa",
    "intracranial",
    "parenchyma",
    "hemisphere",
    "hemispheric",
    "lobar",
    "lobes",
    "lobe",
    "white",
    "matter",
    "whitematter",
    "cortex",
    "cortical",
    "interface",
    "organ",
    "region",
    "segment",
    "specific",
    "lesion",
    "mass",
    "effect",
    "ventricle",
    "ventricular",
}

LATERALITY_ALIAS_TO_CANONICAL = {
    "bilateral": "bilateral",
    "both": "bilateral",
    "bothsides": "bilateral",
    "bothside": "bilateral",
    "symmetric": "bilateral",
    "symmetrical": "bilateral",
    "unilateral": "unilateral",
    "left": "left",
    "right": "right",
    "midline": "midline",
    "central": "midline",
}

ANATOMY_EQUIVALENCE = {
    "brainstem": {"brainstem", "medulla", "medullary", "pons", "pontine", "midbrain", "midbrainstem"},
    "medulla": {"medulla", "medullary", "medullaoblongata", "bulbar"},
    "cerebellum": {"cerebellum", "cerebellar", "posteriorfossa", "vermis", "paravermis"},
    "posteriorfossa": {"posteriorfossa", "cerebellum", "brainstem"},
    "white matter": {"whitematter", "subcortical", "leukoencephalopathy"},
    "cortex": {"cortex", "cortical", "corticalsubcortical", "corticosubcortical"},
    "ventricle": {"ventricle", "ventricular", "ivventricle", "fourthventricle"},
    "spinal cord": {"spinalcord", "cord", "myelon"},
}


@dataclass
class RuleBuckets:
    score3_terms: List[str] = field(default_factory=list)
    score2_terms: List[str] = field(default_factory=list)
    score1_terms: List[str] = field(default_factory=list)
    score3_aliases: Set[str] = field(default_factory=set)
    score2_aliases: Set[str] = field(default_factory=set)
    score1_aliases: Set[str] = field(default_factory=set)
    option_alias_to_option: Dict[str, str] = field(default_factory=dict)
    gold_reference: str = ""
    gold_relaxed_aliases: Set[str] = field(default_factory=set)


@dataclass
class RuleScoreResult:
    payload: Dict[str, Any]
    diagnostics: Dict[str, Any]


class RuleScorer:
    def __init__(self, *, version: str = "rule_v8_rubric_extracted_terms") -> None:
        self.version = version

    def score_case(
        self,
        case: Any,
        final_top1_diagnosis: str,
        final_differential: List[Dict[str, Any]],
        final_location: Dict[str, str],
        trajectory_unique_diagnoses: List[str],
    ) -> RuleScoreResult:
        buckets = self._build_rule_buckets(case)

        diagnosis_score, diagnosis_reason, diagnosis_meta = self._score_top1(final_top1_diagnosis, buckets)
        localization_score, localization_reason, localization_meta = self._score_location(case, final_location)
        ddx_score, ddx_reason, ddx_meta = self._score_final_ddx(final_differential, buckets)

        trajectory_labels: List[Dict[str, Any]] = []
        trajectory_scores: List[Dict[str, Any]] = []
        for diagnosis in deduplicate_preserve_order(trajectory_unique_diagnoses):
            label, reason, _ = self._label_diagnosis(diagnosis, buckets)
            score, score_reason, _ = self._score_top1(diagnosis, buckets)
            trajectory_labels.append({
                "diagnosis": diagnosis,
                "label": label,
                "reason": reason,
            })
            trajectory_scores.append({
                "diagnosis": diagnosis,
                "score": score,
                "reason": score_reason,
            })

        payload = {
            "final_scores": {
                "diagnosis": {"score": diagnosis_score, "reason": diagnosis_reason},
                "localization": {"score": localization_score, "reason": localization_reason},
                "differential_list": {"score": ddx_score, "reason": ddx_reason},
            },
            "trajectory_labels": trajectory_labels,
            "trajectory_scores": trajectory_scores,
            "prompt_version": self.version,
            "provider_meta": {"mode": "rule_based"},
            "raw_text": "",
        }
        diagnostics = {
            "diagnosis_meta": diagnosis_meta,
            "localization_meta": localization_meta,
            "ddx_meta": ddx_meta,
            "rule_buckets": {
                "score_3": buckets.score3_terms,
                "score_2": buckets.score2_terms,
                "score_1": buckets.score1_terms,
                "gold_reference": buckets.gold_reference,
            },
        }
        return RuleScoreResult(payload=payload, diagnostics=diagnostics)

    def _build_rule_buckets(self, case: Any) -> RuleBuckets:
        diagnosis_rubric = getattr(case, "diagnosis_rubric", {}) or {}
        gold_reference = normalize_space(
            diagnosis_rubric.get("reference_answer")
            or getattr(case, "final_diagnosis", "")
            or ""
        )
        reference_options = list(getattr(case, "reference_ddx_options", []) or [])

        score3_terms = deduplicate_preserve_order(
            [gold_reference]
            + self._extract_rubric_terms(diagnosis_rubric.get("3") or "")
        )
        if not score3_terms and gold_reference:
            score3_terms = [gold_reference]

        relaxed_gold_aliases: Set[str] = set()
        for term in score3_terms or [gold_reference]:
            relaxed_gold_aliases.update(self._relaxed_aliases_for_term(term))

        score2_terms = deduplicate_preserve_order(
            self._extract_rubric_terms(diagnosis_rubric.get("2") or "")
        )
        if not score2_terms:
            score2_terms = deduplicate_preserve_order(sorted(relaxed_gold_aliases))

        score1_terms = deduplicate_preserve_order(
            [opt for opt in reference_options if normalize_key(opt) != normalize_key(gold_reference)]
            + self._extract_rubric_terms(diagnosis_rubric.get("1") or "")
        )

        buckets = RuleBuckets(
            score3_terms=score3_terms,
            score2_terms=score2_terms,
            score1_terms=score1_terms,
            gold_reference=gold_reference,
            gold_relaxed_aliases=relaxed_gold_aliases,
        )

        for term in score3_terms:
            # Exact aliases must stay specific. Relaxed/subset variants belong to score-2/acceptable buckets.
            buckets.score3_aliases.update(self._generate_aliases(term, include_relaxed=False))
        for term in score2_terms:
            buckets.score2_aliases.update(self._generate_aliases(term, include_relaxed=True))
        buckets.score2_aliases.update(relaxed_gold_aliases)
        for term in score1_terms:
            aliases = self._generate_aliases(term)
            buckets.score1_aliases.update(aliases)
            for alias in aliases:
                buckets.option_alias_to_option.setdefault(alias, term)

        if gold_reference:
            for alias in self._generate_aliases(gold_reference):
                buckets.option_alias_to_option.setdefault(alias, gold_reference)

        return buckets

    def _extract_quoted_terms(self, text: str) -> List[str]:
        raw = normalize_space(text)
        if not raw:
            return []
        candidates: List[str] = []
        quote_patterns = [
            r'"([^"\n]+)"',
            r"'([^'\n]+)'",
            r'“([^”\n]+)”',
            r'‘([^’\n]+)’',
        ]
        for pattern in quote_patterns:
            for match in re.findall(pattern, raw):
                term = normalize_space(match)
                if term and len(term) <= 120:
                    candidates.append(term)
        return deduplicate_preserve_order(candidates)

    def _split_rubric_candidate_terms(self, text: str) -> List[str]:
        raw = normalize_space(text)
        if not raw:
            return []
        raw = re.sub(r"\([^)]{0,40}\)", " ", raw)
        parts = re.split(r",|;|\b(?:or|and/or)\b", raw, flags=re.I)
        out: List[str] = []
        for part in parts:
            term = normalize_space(part)
            term = re.sub(
                r"^(?:the|a|an|diagnosis|answer|term|terms|phrase|phrasing|mentions?|states?|includes?|include|such as|e\.g\.?|for example)\s+",
                "",
                term,
                flags=re.I,
            )
            term = normalize_space(term.strip(" .:-–—()[]{}"))
            if 3 <= len(term) <= 120 and re.search(r"[A-Za-z]", term):
                if normalize_key(term) not in {"diagnosis", "condition", "disease", "syndrome", "entity", "finding", "findings"}:
                    out.append(term)
        return deduplicate_preserve_order(out)

    def _extract_rubric_terms(self, text: str) -> List[str]:
        raw = normalize_space(text)
        if not raw:
            return []
        candidates: List[str] = []
        candidates.extend(self._extract_quoted_terms(raw))
        marker_patterns = [
            r"(?:acceptable(?:\s+(?:phrasing|terms?|answers?))?\s+(?:includes?|are)|accept(?:s|able)?(?:\s+minor\s+wording\s+variations)?(?:\s+such\s+as)?|equivalent(?:\s+terms?)?(?:\s+(?:includes?|are))?|near[- ]gold(?:\s+answers?)?(?:\s+includes?)?|such\s+as|e\.g\.?|for\s+example|including)\s*[:：]?\s*([^.;]+)",
            r"\bor\s+equivalent\s*[:：]?\s*([^.;)]+)",
        ]
        for pattern in marker_patterns:
            for match in re.findall(pattern, raw, flags=re.I):
                candidates.extend(self._split_rubric_candidate_terms(match))
        return deduplicate_preserve_order(candidates)

    def _apply_spelling_map(self, text: str) -> str:
        out = f" {text.lower()} "
        for src, dst in BRITISH_AMERICAN.items():
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
        return normalize_space(out)

    def _apply_term_canonicalization(self, text: str) -> str:
        out = f" {normalize_space(text).lower()} "
        for pattern, replacement in TERM_CANONICAL_REPLACEMENTS:
            out = re.sub(pattern, replacement, out, flags=re.I)
        return normalize_space(out)

    def _clean_surface_text(self, text: str) -> str:
        text = self._apply_term_canonicalization(self._apply_spelling_map(normalize_space(text)))
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\b(?:dx|ddx)\b", " diagnosis ", text)
        for phrase in FILLER_PHRASES:
            text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
        text = re.sub(r"[\[\]{}]", " ", text)
        text = re.sub(r"[,:]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_alias(self, text: str) -> str:
        text = self._clean_surface_text(text)
        text = re.sub(r"[()]+", " ", text)
        text = re.sub(r"[^a-z0-9+\-/ ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _contains_negation(self, text: str) -> bool:
        head = self._clean_surface_text(text)[:80]
        return any(re.search(pat, head, flags=re.I) for pat in NEGATION_PATTERNS)

    def _extract_primary_concept(self, text: str) -> str:
        cleaned = self._clean_surface_text(text)
        if not cleaned or self._contains_negation(cleaned):
            return ""
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        parts = [normalize_space(p) for p in MULTI_DX_SPLIT_RE.split(cleaned) if normalize_space(p)]
        primary = parts[0] if parts else cleaned
        primary = re.sub(r"^[-–—]+", "", primary).strip()
        return primary

    def _tokenize(self, text: str, *, remove_stopwords: bool = True, remove_qualifiers: bool = False) -> List[str]:
        norm = self._normalize_alias(text)
        norm = norm.replace("/", " ").replace("-", " ")
        tokens = [tok for tok in re.split(r"\s+", norm) if tok]
        out: List[str] = []
        for token in tokens:
            if remove_stopwords and token in STOPWORDS:
                continue
            if remove_qualifiers and token in QUALIFIER_TOKENS:
                continue
            out.append(token)
        return out

    def _canonical_family_tokens(self, tokens: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for token in tokens:
            if token in STOPWORDS:
                continue
            out.add(DISEASE_FAMILY_MAP.get(token, token))
        return out

    def _expanded_anatomy_tokens(self, tokens: Iterable[str]) -> Set[str]:
        token_set = set(tokens)
        out: Set[str] = set()
        for canonical, aliases in ANATOMY_EQUIVALENCE.items():
            canon_norm = canonical.replace(" ", "")
            if token_set & aliases or canon_norm in token_set or canonical in token_set:
                out.add(canon_norm)
                out.update(tok for tok in token_set if tok in aliases)
        return out

    def _generate_aliases(self, term: str, *, include_relaxed: bool = True) -> Set[str]:
        if not normalize_space(term):
            return set()
        term = normalize_space(term)
        aliases: Set[str] = set()
        aliases.add(self._normalize_alias(term))

        no_paren = normalize_space(re.sub(r"\([^)]*\)", " ", term))
        if no_paren:
            aliases.add(self._normalize_alias(no_paren))

        for acronym in re.findall(r"\(([A-Z][A-Z0-9\-]{1,})\)", term):
            aliases.add(self._normalize_alias(acronym))
            aliases.add(self._normalize_alias(acronym.replace("-", "")))

        tokens = self._tokenize(no_paren, remove_stopwords=True, remove_qualifiers=False)
        sig_tokens = [tok for tok in tokens if len(tok) > 2]
        full_tokens = [tok for tok in self._tokenize(no_paren, remove_stopwords=False, remove_qualifiers=False) if len(tok) > 2]
        initialisms = set()
        if sig_tokens:
            initialisms.add("".join(tok[0] for tok in sig_tokens))
        if full_tokens:
            initialisms.add("".join(tok[0] for tok in full_tokens))
        for initialism in initialisms:
            if len(initialism) >= 3:
                aliases.add(initialism)
                if len(initialism) >= 4:
                    aliases.add(f"{initialism[:-2]}-{initialism[-2:]}")
                    aliases.add(f"{initialism[:-2]} {initialism[-2:]}")

        if include_relaxed:
            relaxed = self._relaxed_aliases_for_term(term)
            aliases.update(relaxed)

        expanded: Set[str] = set()
        for alias in aliases:
            alias = self._normalize_alias(alias)
            if not alias:
                continue
            expanded.add(alias)
            expanded.add(alias.replace("-", " "))
            expanded.add(alias.replace("-", ""))
            expanded.add(alias.replace(" ", ""))
        return {alias for alias in expanded if alias}

    def _relaxed_aliases_for_term(self, term: str) -> Set[str]:
        tokens = self._tokenize(term, remove_stopwords=True, remove_qualifiers=False)
        core_tokens = [tok for tok in tokens if tok not in QUALIFIER_TOKENS]
        aliases: Set[str] = set()
        if core_tokens:
            aliases.add(" ".join(core_tokens))
        if len(core_tokens) >= 2:
            aliases.add(" ".join(core_tokens[-2:]))
        if tokens and len(tokens) >= 2:
            aliases.add(" ".join(tokens[-2:]))
        return {self._normalize_alias(alias) for alias in aliases if alias}

    def _match_exact_gold(self, pred_alias: str, buckets: RuleBuckets) -> bool:
        return pred_alias in buckets.score3_aliases

    def _match_near_gold(self, pred_text: str, pred_alias: str, buckets: RuleBuckets) -> bool:
        if pred_alias in buckets.score2_aliases or pred_alias in buckets.gold_relaxed_aliases:
            return True
        gold_alias = self._normalize_alias(buckets.gold_reference)
        if (("air embolism" in pred_alias or "airembolism" in pred_alias) and ("air embolism" in gold_alias or "airembolism" in gold_alias)
                and ("decompression illness" in pred_alias or "decompressionillness" in pred_alias)
                and ("decompression illness" in gold_alias or "decompressionillness" in gold_alias)):
            return True
        pred_core = self._tokenize(pred_text, remove_stopwords=True, remove_qualifiers=True)
        gold_core = self._tokenize(buckets.gold_reference, remove_stopwords=True, remove_qualifiers=True)
        if pred_core and gold_core and set(pred_core) == set(gold_core):
            return True
        if pred_core and gold_core:
            pred_set = set(pred_core)
            gold_set = set(gold_core)
            if (pred_set <= gold_set or gold_set <= pred_set) and min(len(pred_set), len(gold_set)) >= 2:
                return True
            fam_pred = self._canonical_family_tokens(pred_core)
            fam_gold = self._canonical_family_tokens(gold_core)
            if fam_pred == fam_gold and len(fam_pred) >= 1 and SequenceMatcher(None, pred_alias, gold_alias).ratio() >= 0.72:
                return True
        return False

    def _match_reference_option(self, pred_text: str, pred_alias: str, buckets: RuleBuckets) -> Tuple[bool, Optional[str]]:
        if pred_alias in buckets.option_alias_to_option:
            return True, buckets.option_alias_to_option[pred_alias]

        pred_tokens = set(self._tokenize(pred_text, remove_stopwords=True, remove_qualifiers=False))
        if not pred_tokens:
            return False, None
        pred_family = self._canonical_family_tokens(pred_tokens)
        pred_informative = pred_tokens - GENERIC_REFERENCE_OPTION_TOKENS - STOPWORDS - QUALIFIER_TOKENS
        pred_anatomy = self._expanded_anatomy_tokens(pred_tokens)

        best_option: Optional[str] = None
        best_score = 0.0
        unique_options = deduplicate_preserve_order(buckets.option_alias_to_option.values())
        for option in unique_options:
            option_alias = self._normalize_alias(option)
            if option_alias and SequenceMatcher(None, pred_alias, option_alias).ratio() >= 0.9:
                return True, option

            option_tokens = set(self._tokenize(option, remove_stopwords=True, remove_qualifiers=False))
            if not option_tokens:
                continue
            option_family = self._canonical_family_tokens(option_tokens)
            option_informative = option_tokens - GENERIC_REFERENCE_OPTION_TOKENS - STOPWORDS - QUALIFIER_TOKENS
            option_anatomy = self._expanded_anatomy_tokens(option_tokens)
            direct_overlap = pred_tokens & option_tokens
            informative_overlap = pred_informative & option_informative
            family_overlap = pred_family & option_family
            anatomy_overlap = pred_anatomy & option_anatomy

            accept = False
            if len(option_tokens) == 1 and next(iter(option_tokens)) in pred_tokens:
                # Generic one-token options such as "infection" or "vasculitis" may legitimately appear with modifiers.
                accept = True
            elif informative_overlap and (family_overlap or len(direct_overlap) >= 2):
                accept = True
            elif len(direct_overlap) >= 2 and (informative_overlap or SequenceMatcher(None, pred_alias, option_alias).ratio() >= 0.78):
                accept = True
            elif option_alias and option_alias in pred_alias and len(option_informative) >= 1:
                accept = True
            elif family_overlap and anatomy_overlap and (informative_overlap or len(direct_overlap) >= 1):
                accept = True

            score = (
                3.0 * len(informative_overlap)
                + 1.5 * len(direct_overlap)
                + 0.5 * len(family_overlap)
                + 0.5 * len(anatomy_overlap)
                + (1.0 if accept else 0.0)
            )
            if accept and score > best_score:
                best_score = score
                best_option = option

        if best_option:
            return True, best_option
        return False, None

    def _label_diagnosis(self, diagnosis: str, buckets: RuleBuckets) -> Tuple[str, str, Dict[str, Any]]:
        primary = self._extract_primary_concept(diagnosis)
        pred_alias = self._normalize_alias(primary)
        meta = {
            "input": diagnosis,
            "primary": primary,
            "alias": pred_alias,
        }
        if not primary:
            return "U", "negated/empty/invalid diagnosis string", meta
        if self._match_exact_gold(pred_alias, buckets):
            return "E", "exact or very close lexical variant of gold diagnosis", meta
        if self._match_near_gold(primary, pred_alias, buckets):
            return "A", "near-gold diagnosis missing a key qualifier or specificity", meta
        acceptable, matched_option = self._match_reference_option(primary, pred_alias, buckets)
        if acceptable:
            meta["matched_option"] = matched_option
            return "A", "matches a reference differential option or close equivalent", meta
        return "U", "not matched to gold diagnosis or reference differential set", meta

    def _score_top1(self, top1_diagnosis: str, buckets: RuleBuckets) -> Tuple[int, str, Dict[str, Any]]:
        primary = self._extract_primary_concept(top1_diagnosis)
        pred_alias = self._normalize_alias(primary)
        meta = {
            "input": top1_diagnosis,
            "primary": primary,
            "alias": pred_alias,
        }
        if not primary:
            return 0, "empty, negated, or invalid top-1 diagnosis", meta
        if self._match_exact_gold(pred_alias, buckets):
            return 3, "exact match to the gold diagnosis concept", meta
        if self._match_near_gold(primary, pred_alias, buckets):
            return 2, "near-gold diagnosis but missing specificity/qualifier", meta
        acceptable, matched_option = self._match_reference_option(primary, pred_alias, buckets)
        if acceptable:
            meta["matched_option"] = matched_option
            return 1, "acceptable alternative differential but not the gold diagnosis", meta
        return 0, "off-target top-1 diagnosis", meta

    def _normalize_laterality(self, text: str) -> Set[str]:
        norm = self._normalize_alias(text).replace(" ", "")
        out: Set[str] = set()
        for alias, canonical in LATERALITY_ALIAS_TO_CANONICAL.items():
            if alias in norm:
                out.add(canonical)
        return out

    def _laterality_mode(self, text: str) -> str:
        lat_set = self._normalize_laterality(text)
        if not lat_set:
            return "none"
        if "bilateral" in lat_set or ({"left", "right"} <= lat_set):
            return "bilateral"
        if "midline" in lat_set:
            return "midline"
        if "left" in lat_set and "right" not in lat_set:
            return "left"
        if "right" in lat_set and "left" not in lat_set:
            return "right"
        if "unilateral" in lat_set:
            return "unilateral"
        return "other"

    def _laterality_relation(self, gold_text: str, pred_text: str) -> str:
        gold_mode = self._laterality_mode(gold_text)
        pred_mode = self._laterality_mode(pred_text)
        if gold_mode == "none":
            return "neutral"
        if pred_mode == gold_mode:
            return "full"
        if gold_mode == "bilateral" and pred_mode in {"left", "right", "unilateral"}:
            return "partial"
        if gold_mode in {"left", "right"} and pred_mode in {"bilateral", "unilateral"}:
            return "partial"
        if gold_mode == "midline" and pred_mode in {"bilateral", "left", "right", "unilateral"}:
            return "partial"
        return "none"

    def _normalize_region_tokens(self, text: str) -> Set[str]:
        raw_tokens = self._tokenize(text, remove_stopwords=True, remove_qualifiers=False)
        tokens = [tok for tok in raw_tokens if tok not in REGION_STOPWORDS]
        merged = "".join(tokens)
        out: Set[str] = set(tokens)
        if merged:
            out.add(merged)
        for canonical, aliases in ANATOMY_EQUIVALENCE.items():
            if out & aliases:
                out.add(canonical.replace(" ", ""))
        return out

    def _primary_location_phrase(self, text: str) -> str:
        raw = normalize_space(text)
        if not raw:
            return ""
        raw = re.sub(r"\([^)]*\)", " ", raw)
        raw = re.split(r"[;|]", raw, maxsplit=1)[0]
        return normalize_space(raw)

    def _strict_substructure_tokens(self, text: str) -> Set[str]:
        raw_tokens = self._tokenize(text, remove_stopwords=True, remove_qualifiers=False)
        tokens = [tok for tok in raw_tokens if tok not in REGION_STOPWORDS and tok not in GENERIC_SUBSTRUCTURE_TOKENS]
        merged = "".join(tokens)
        out: Set[str] = set(tokens)
        if merged and merged not in GENERIC_SUBSTRUCTURE_TOKENS:
            out.add(merged)
        return out

    def _score_location(self, case: Any, final_location: Dict[str, str]) -> Tuple[int, str, Dict[str, Any]]:
        loc_rubric = getattr(case, "localization_rubric", {}) or {}
        gold_ref = loc_rubric.get("reference_answer") or {}
        if not isinstance(gold_ref, dict):
            gold_ref = {}

        gold_laterality = normalize_space(gold_ref.get("Laterality") or gold_ref.get("laterality") or "")
        gold_region = normalize_space(gold_ref.get("Organ/Region") or gold_ref.get("region") or "")
        gold_sub = normalize_space(gold_ref.get("Specific Substructure/Segment") or gold_ref.get("substructure") or "")

        pred_laterality = normalize_space((final_location or {}).get("laterality") or "")
        pred_region = normalize_space((final_location or {}).get("region") or "")
        pred_sub = normalize_space((final_location or {}).get("substructure") or "")

        gold_region_tokens = self._normalize_region_tokens(gold_region)
        gold_sub_tokens = self._normalize_region_tokens(gold_sub)
        pred_region_tokens = self._normalize_region_tokens(pred_region)
        pred_sub_tokens = self._normalize_region_tokens(pred_sub)

        gold_primary_sub = self._primary_location_phrase(gold_sub)
        pred_primary_sub = self._primary_location_phrase(pred_sub)
        gold_sub_specific = self._strict_substructure_tokens(gold_primary_sub)
        pred_specific = self._strict_substructure_tokens(pred_primary_sub) | self._strict_substructure_tokens(pred_region)

        region_match = bool(pred_region_tokens & gold_region_tokens) or bool(pred_region_tokens & gold_sub_tokens)
        broad_overlap = bool((pred_region_tokens | pred_sub_tokens) & (gold_region_tokens | gold_sub_tokens))
        laterality_relation = self._laterality_relation(gold_laterality, pred_laterality)

        specific_overlap = pred_specific & gold_sub_specific
        gold_specific_count = len(gold_sub_specific)
        sub_match_full = False
        sub_match_partial = False
        if gold_specific_count == 0:
            sub_match_partial = broad_overlap
        else:
            if specific_overlap and (len(specific_overlap) >= 2 or gold_specific_count == 1):
                sub_match_full = True
            elif specific_overlap:
                sub_match_partial = True

        specific_conflict = bool(gold_sub_specific) and bool(pred_specific) and not bool(specific_overlap)

        meta = {
            "gold": {
                "laterality": gold_laterality,
                "region": gold_region,
                "substructure": gold_sub,
            },
            "pred": {
                "laterality": pred_laterality,
                "region": pred_region,
                "substructure": pred_sub,
            },
            "region_match": region_match,
            "laterality_relation": laterality_relation,
            "broad_overlap": broad_overlap,
            "gold_primary_substructure": gold_primary_sub,
            "gold_sub_specific": sorted(gold_sub_specific),
            "pred_specific": sorted(pred_specific),
            "specific_overlap": sorted(specific_overlap),
            "substructure_match_full": sub_match_full,
            "substructure_match_partial": sub_match_partial,
            "specific_conflict": specific_conflict,
        }

        if region_match and sub_match_full and laterality_relation in {"full", "neutral"}:
            return 3, "region, laterality, and specific substructure are aligned with the reference", meta
        if region_match and not specific_conflict and laterality_relation in {"full", "partial", "neutral"} and (sub_match_partial or laterality_relation in {"full", "partial"}):
            return 2, "correct region with partial but incomplete localization specificity", meta
        if region_match or broad_overlap:
            return 1, "broad but incomplete localization overlap with the reference", meta
        return 0, "wrong or absent localization", meta

    def _score_final_ddx(self, final_differential: List[Dict[str, Any]], buckets: RuleBuckets) -> Tuple[int, str, Dict[str, Any]]:
        labels: List[str] = []
        matched_options: List[str] = []
        gold_exact_rank: Optional[int] = None
        gold_near_rank: Optional[int] = None
        per_item: List[Dict[str, Any]] = []

        for idx, item in enumerate(final_differential[:4], start=1):
            diagnosis = normalize_space(item.get("diagnosis") or "")
            label, reason, meta = self._label_diagnosis(diagnosis, buckets)
            labels.append(label)
            primary = meta.get("primary") or diagnosis
            pred_alias = meta.get("alias") or self._normalize_alias(primary)
            if self._match_exact_gold(pred_alias, buckets) and gold_exact_rank is None:
                gold_exact_rank = idx
            elif self._match_near_gold(primary, pred_alias, buckets) and gold_near_rank is None:
                gold_near_rank = idx
            acceptable, matched_option = self._match_reference_option(primary, pred_alias, buckets)
            if acceptable and matched_option:
                matched_options.append(matched_option)
            per_item.append({
                "rank": idx,
                "diagnosis": diagnosis,
                "label": label,
                "reason": reason,
                "meta": meta,
            })

        unique_acceptables = len(deduplicate_preserve_order(matched_options))
        if gold_exact_rank == 1 and unique_acceptables >= 3:
            score, reason = 3, "gold diagnosis is prominent and most items align with the reference differential set"
        elif (gold_exact_rank is not None or gold_near_rank in {1, 2}) and unique_acceptables >= 2:
            score, reason = 2, "gold diagnosis is present or nearly present but coverage/ranking is incomplete"
        elif unique_acceptables >= 1:
            score, reason = 1, "list contains at least some acceptable reference differential overlap"
        else:
            score, reason = 0, "final differential list is mostly off-target"

        meta = {
            "per_item": per_item,
            "unique_acceptable_matches": unique_acceptables,
            "gold_exact_rank": gold_exact_rank,
            "gold_near_rank": gold_near_rank,
        }
        return score, reason, meta


# =========================
# Dual-mode judge orchestrator
# =========================

class JudgeRunner:
    def __init__(
        self,
        *,
        model_call: Optional[JudgeCallerProtocol],
        enable_llm: bool = True,
        enable_rule: bool = True,
        prompt_version: str = "judge_v5_schema_aligned_trajectory_scores",
        rule_version: str = "rule_v8_rubric_extracted_terms",
    ) -> None:
        if not enable_llm and not enable_rule:
            raise ValueError("At least one judge mode must be enabled.")
        self.model_call = model_call
        self.enable_llm = enable_llm
        self.enable_rule = enable_rule
        self.prompt_version = prompt_version
        self.rule_scorer = RuleScorer(version=rule_version)

    def score_case(
        self,
        case: Any,
        final_top1_diagnosis: str,
        final_differential: List[Dict[str, Any]],
        final_location: Dict[str, str],
        trajectory_unique_diagnoses: List[str],
    ) -> Dict[str, Any]:
        case_payload = {
            "case_id": getattr(case, "case_id", ""),
            "clinical_history": getattr(case, "clinical_history", ""),
            "gold_final_diagnosis": getattr(case, "final_diagnosis", ""),
            "reference_differential_options": list(getattr(case, "reference_ddx_options", []) or []),
            "diagnosis_rubric": getattr(case, "diagnosis_rubric", {}) or {},
            "localization_rubric": getattr(case, "localization_rubric", {}) or {},
        }
        model_payload = {
            "final_top1_diagnosis": final_top1_diagnosis,
            "final_differential": final_differential,
            "final_location": final_location,
            "trajectory_unique_diagnoses": deduplicate_preserve_order(trajectory_unique_diagnoses),
        }

        enabled_modes: List[str] = []
        by_mode: Dict[str, Any] = {}
        diagnostics: Dict[str, Any] = {}

        if self.enable_rule:
            rule_result = self.rule_scorer.score_case(
                case=case,
                final_top1_diagnosis=final_top1_diagnosis,
                final_differential=final_differential,
                final_location=final_location,
                trajectory_unique_diagnoses=trajectory_unique_diagnoses,
            )
            rule_payload = dict(rule_result.payload)
            rule_payload["case_payload"] = case_payload
            rule_payload["model_payload"] = model_payload
            by_mode["rule"] = rule_payload
            diagnostics["rule"] = rule_result.diagnostics
            enabled_modes.append("rule")

        errors: Dict[str, str] = {}
        if self.enable_llm:
            if self.model_call is None:
                if not self.enable_rule:
                    raise ValueError("LLM judge mode requested but no model_call was provided.")
                errors["llm"] = "LLM judge mode requested but no model_call was provided."
            else:
                try:
                    user_prompt = build_judge_prompt(case_payload=case_payload, model_payload=model_payload)
                    response = self.model_call(system_prompt=JUDGE_SYSTEM_PROMPT, user_prompt=user_prompt)
                    raw_text = response.get("text") if isinstance(response, dict) else str(response)
                    parsed = extract_json_from_text(raw_text or "")
                    normalized = normalize_judge_output(parsed, trajectory_unique_diagnoses)
                    normalized["prompt_version"] = self.prompt_version
                    normalized["raw_text"] = raw_text
                    normalized["provider_meta"] = response.get("provider_meta", {}) if isinstance(response, dict) else {}
                    normalized["case_payload"] = case_payload
                    normalized["model_payload"] = model_payload
                    by_mode["llm"] = normalized
                    enabled_modes.append("llm")
                except Exception as exc:
                    if not self.enable_rule:
                        raise
                    errors["llm"] = repr(exc)

        default_mode = "llm" if "llm" in by_mode else "rule"
        agreement = self._compute_mode_agreement(by_mode)

        return {
            "default_mode": default_mode,
            "enabled_modes": enabled_modes,
            "by_mode": by_mode,
            "agreement": agreement,
            "diagnostics": diagnostics,
            "errors": errors,
        }

    def _compute_mode_agreement(self, by_mode: Dict[str, Any]) -> Dict[str, Any]:
        if not ({"llm", "rule"} <= set(by_mode.keys())):
            return {}
        llm = by_mode["llm"]
        rule = by_mode["rule"]
        llm_scores = llm.get("final_scores") or {}
        rule_scores = rule.get("final_scores") or {}

        out = {
            "final_score_exact_match": {},
            "trajectory_label_agreement": None,
            "trajectory_score_mae": None,
        }
        for key in ["diagnosis", "localization", "differential_list"]:
            out["final_score_exact_match"][key] = int(
                ((llm_scores.get(key) or {}).get("score", 0)) == ((rule_scores.get(key) or {}).get("score", 0))
            )

        llm_labels = {normalize_key(x.get("diagnosis")): normalize_space(x.get("label") or "U").upper()[:1] for x in (llm.get("trajectory_labels") or [])}
        rule_labels = {normalize_key(x.get("diagnosis")): normalize_space(x.get("label") or "U").upper()[:1] for x in (rule.get("trajectory_labels") or [])}
        keys = sorted(set(llm_labels) | set(rule_labels))
        if keys:
            agree = sum(1 for k in keys if llm_labels.get(k, "U") == rule_labels.get(k, "U"))
            out["trajectory_label_agreement"] = agree / len(keys)

        llm_scores = {normalize_key(x.get("diagnosis")): int(x.get("score", 0)) for x in (llm.get("trajectory_scores") or [])}
        rule_scores = {normalize_key(x.get("diagnosis")): int(x.get("score", 0)) for x in (rule.get("trajectory_scores") or [])}
        score_keys = sorted(set(llm_scores) | set(rule_scores))
        if score_keys:
            out["trajectory_score_mae"] = sum(abs(llm_scores.get(k, 0) - rule_scores.get(k, 0)) for k in score_keys) / len(score_keys)
        return out
