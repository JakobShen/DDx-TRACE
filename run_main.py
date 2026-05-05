
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from pipeline_main import (
    AGENT_SYSTEM_PROMPT,
    CanonicalCase,
    ImagePayload,
    MainBenchmarkPipeline,
    ModelReply,
    aggregate_results,
    build_dataset_preflight,
    build_metric_notes,
    canonicalize_case,
    ensure_dir,
    json_dumps,
    normalize_key,
    normalize_space,
)
from judge import (
    JUDGE_JSON_SCHEMA,
    JUDGE_JSON_SCHEMA_DESCRIPTION,
    JUDGE_JSON_SCHEMA_NAME,
    JudgeRunner,
)


# =========================
# Generic utilities
# =========================

def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp_path.replace(path)


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def slugify(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "-" for ch in (text or "").strip().lower())
    out = "-".join(part for part in out.split("-") if part)
    return out or "run"


def read_binary(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def detected_image_mime_type(data: bytes, fallback: str) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def image_to_data_url(image: ImagePayload) -> str:
    if image.path:
        data = read_binary(image.path)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{detected_image_mime_type(data, image.mime_type)};base64,{encoded}"
    if image.url:
        return image.url
    raise FileNotFoundError(f"No accessible path/url for image payload: {image}")


def openai_response_output_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload.get("output_text"):
        return payload["output_text"]

    texts: List[str] = []
    for item in payload.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content") or []:
                if block.get("type") in {"output_text", "text"} and isinstance(block.get("text"), str):
                    texts.append(block["text"])
    return "\n".join(texts).strip()


def anthropic_output_text(payload: Dict[str, Any]) -> str:
    texts: List[str] = []
    for block in payload.get("content") or []:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    return "\n".join(texts).strip()


def anthropic_sdk_payload(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(response, "dict"):
        return response.dict()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "model_dump_json"):
        return json.loads(response.model_dump_json())
    raise ProviderError(f"Unexpected Anthropic SDK response type: {type(response)}")


def gemini_output_text(payload: Dict[str, Any]) -> str:
    texts: List[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
        if texts:
            break
    return "\n".join(texts).strip()


def openai_chat_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        texts: List[str] = []
        for part in message_content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts).strip()
    return normalize_space(message_content)


class ProviderError(RuntimeError):
    pass


class GoogleRetryExhaustedError(ProviderError):
    pass


@dataclass
class ProviderSettings:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str]
    timeout: float
    max_output_tokens: int
    image_detail: str = "auto"
    reasoning_effort: Optional[str] = None
    agent_structured_output: str = "auto"
    gemini_thinking_level: Optional[str] = None
    qwen_transport: str = "auto"
    gemini_api_version: str = "v1beta"
    vertex_project: Optional[str] = None
    vertex_region: Optional[str] = None


AGENT_TURN_JSON_SCHEMA_NAME = "eurorad_agent_turn_v1"
AGENT_TURN_JSON_SCHEMA_DESCRIPTION = "Official EuroRad active-diagnosis agent turn output."
AGENT_TURN_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["request_exam", "stop"]},
        "requested_examination": {"type": "string"},
        "current_differential": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "diagnosis": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["diagnosis", "probability"],
            },
        },
        "final_location": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "laterality": {"type": "string"},
                "region": {"type": "string"},
                "substructure": {"type": "string"},
            },
            "required": ["laterality", "region", "substructure"],
        },
    },
    "required": ["action", "requested_examination", "current_differential", "final_location"],
}


def wants_agent_schema(settings: ProviderSettings) -> bool:
    return (settings.agent_structured_output or "auto").lower() != "never"


def requires_agent_schema(settings: ProviderSettings) -> bool:
    return (settings.agent_structured_output or "auto").lower() == "always"


def add_openai_responses_json_schema(body: Dict[str, Any], *, schema_name: str, schema: Dict[str, Any], schema_description: str) -> None:
    body["text"] = {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "description": schema_description,
            "strict": True,
            "schema": schema,
        }
    }


def add_chat_json_schema(body: Dict[str, Any], *, schema_name: str, schema: Dict[str, Any], schema_description: str) -> None:
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "description": schema_description,
            "schema": schema,
        },
    }


def add_anthropic_json_schema(body: Dict[str, Any], *, schema: Dict[str, Any]) -> None:
    body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}


def add_gemini_rest_thinking_config(generation_config: Dict[str, Any], settings: ProviderSettings) -> None:
    if settings.gemini_thinking_level:
        generation_config.setdefault("thinkingConfig", {})["thinkingLevel"] = settings.gemini_thinking_level


def add_vertex_thinking_config(config: Dict[str, Any], settings: ProviderSettings) -> None:
    if settings.gemini_thinking_level:
        config.setdefault("thinking_config", {})["thinking_level"] = settings.gemini_thinking_level


GOOGLE_RETRY_STATUS_CODES = {429, 500, 503}
GOOGLE_RETRY_TEXT_MARKERS = (
    "resource_exhausted",
    "unavailable",
    "internal",
    "rate limit",
    "quota",
    "too many requests",
    "timeout",
    "timed out",
)
GOOGLE_RETRY_MAX_RETRIES = 2
GOOGLE_RETRY_BASE_DELAY_SECONDS = 1.0
GOOGLE_RETRY_MAX_DELAY_SECONDS = 2.5
GOOGLE_RETRY_JITTER_SECONDS = 0.25


def google_error_status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value

    match = re.search(r"\b(429|500|503)\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def is_retryable_google_error(exc: BaseException) -> bool:
    status = google_error_status(exc)
    if status in GOOGLE_RETRY_STATUS_CODES:
        return True
    text = normalize_key(str(exc))
    return any(marker in text for marker in GOOGLE_RETRY_TEXT_MARKERS)


def google_retry_delay_seconds(retry_number: int) -> float:
    base_delay = GOOGLE_RETRY_BASE_DELAY_SECONDS * (2 ** max(retry_number - 1, 0))
    return min(base_delay + random.uniform(0.0, GOOGLE_RETRY_JITTER_SECONDS), GOOGLE_RETRY_MAX_DELAY_SECONDS)


def google_retry_meta(retry_count: int, sleep_seconds: float, last_error: Optional[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "google_retry_count": retry_count,
        "google_retry_sleep_seconds": round(float(sleep_seconds), 3),
    }
    if last_error:
        meta["google_retry_last_error"] = last_error
    return meta


def call_google_with_retry(
    call: Callable[[], Any],
    *,
    settings: ProviderSettings,
    operation: str,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[Any, Dict[str, Any]]:
    retry_count = 0
    total_sleep = 0.0
    last_error: Optional[str] = None

    for attempt_index in range(GOOGLE_RETRY_MAX_RETRIES + 1):
        try:
            result = call()
            return result, google_retry_meta(retry_count, total_sleep, last_error)
        except Exception as exc:
            if not is_retryable_google_error(exc):
                raise

            last_error = normalize_space(repr(exc))
            if attempt_index >= GOOGLE_RETRY_MAX_RETRIES:
                raise GoogleRetryExhaustedError(
                    f"{operation} exhausted Google retry budget after {retry_count} retries: {last_error}"
                ) from exc

            retry_number = attempt_index + 1
            delay = google_retry_delay_seconds(retry_number)
            status = google_error_status(exc)
            print(
                "[RETRY] "
                f"provider={settings.provider} model={settings.model} operation={operation} "
                f"status={status or 'unknown'} attempt={retry_number}/{GOOGLE_RETRY_MAX_RETRIES} "
                f"sleep={delay:.2f}s error={last_error}",
                file=sys.stderr,
                flush=True,
            )
            retry_count += 1
            total_sleep += delay
            sleep_fn(delay)

    raise AssertionError("unreachable Google retry loop exit")


def post_google_json_with_retry(
    *,
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    settings: ProviderSettings,
    operation: str,
    error_prefix: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _post() -> Dict[str, Any]:
        response = client.post(url, headers=headers, params=params, json=body)
        if response.status_code >= 400:
            raise ProviderError(f"{error_prefix} {response.status_code}: {response.text}")
        return response.json()

    return call_google_with_retry(_post, settings=settings, operation=operation)


def post_json_with_agent_schema_fallback(
    *,
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    settings: ProviderSettings,
    structured_method: str,
    remove_schema: Callable[[Dict[str, Any]], None],
    error_prefix: str,
    params: Optional[Dict[str, Any]] = None,
    retry_google: bool = False,
    retry_operation: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    provider_meta: Dict[str, Any] = {}
    try:
        if retry_google:
            payload, retry_meta = post_google_json_with_retry(
                client=client,
                url=url,
                headers=headers,
                params=params,
                body=body,
                settings=settings,
                operation=retry_operation or structured_method,
                error_prefix=error_prefix,
            )
        else:
            response = client.post(url, headers=headers, params=params, json=body)
            if response.status_code >= 400:
                raise ProviderError(f"{error_prefix} {response.status_code}: {response.text}")
            payload = response.json()
            retry_meta = {}
        provider_meta.update({
            "agent_structured_output_used": wants_agent_schema(settings),
            "agent_structured_output_method": structured_method if wants_agent_schema(settings) else "prompt_json_only",
        })
        provider_meta.update(retry_meta)
        return payload, provider_meta
    except GoogleRetryExhaustedError:
        raise
    except Exception as exc:
        structured_error = exc

    if wants_agent_schema(settings) and not requires_agent_schema(settings):
        fallback_error = repr(structured_error)
        remove_schema(body)
        if retry_google:
            payload, retry_meta = post_google_json_with_retry(
                client=client,
                url=url,
                headers=headers,
                params=params,
                body=body,
                settings=settings,
                operation=f"{retry_operation or structured_method}.prompt_json_fallback",
                error_prefix=error_prefix,
            )
        else:
            response = client.post(url, headers=headers, params=params, json=body)
            if response.status_code >= 400:
                raise ProviderError(f"{error_prefix} {response.status_code}: {response.text}")
            payload = response.json()
            retry_meta = {}
        provider_meta.update({
            "agent_structured_output_used": False,
            "agent_structured_output_method": "prompt_json_fallback",
            "agent_structured_output_fallback_error": fallback_error,
        })
        provider_meta.update(retry_meta)
        return payload, provider_meta

    raise structured_error


def create_anthropic_vertex_message_with_retry(
    *,
    provider: "AnthropicVertexMessagesProvider",
    body: Dict[str, Any],
    operation: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _create() -> Dict[str, Any]:
        return anthropic_sdk_payload(provider.client.messages.create(**body))

    return call_google_with_retry(_create, settings=provider.settings, operation=operation)


def anthropic_vertex_message_with_agent_schema_fallback(
    *,
    provider: "AnthropicVertexMessagesProvider",
    body: Dict[str, Any],
    structured_method: str,
    remove_schema: Callable[[Dict[str, Any]], None],
    retry_operation: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    provider_meta: Dict[str, Any] = {}
    try:
        payload, retry_meta = create_anthropic_vertex_message_with_retry(
            provider=provider,
            body=body,
            operation=retry_operation or structured_method,
        )
        provider_meta.update({
            "agent_structured_output_used": wants_agent_schema(provider.settings),
            "agent_structured_output_method": structured_method if wants_agent_schema(provider.settings) else "prompt_json_only",
        })
        provider_meta.update(retry_meta)
        return payload, provider_meta
    except GoogleRetryExhaustedError:
        raise
    except Exception as exc:
        structured_error = exc

    if wants_agent_schema(provider.settings) and not requires_agent_schema(provider.settings):
        fallback_error = repr(structured_error)
        remove_schema(body)
        payload, retry_meta = create_anthropic_vertex_message_with_retry(
            provider=provider,
            body=body,
            operation=f"{retry_operation or structured_method}.prompt_json_fallback",
        )
        provider_meta.update({
            "agent_structured_output_used": False,
            "agent_structured_output_method": "prompt_json_fallback",
            "agent_structured_output_fallback_error": fallback_error,
        })
        provider_meta.update(retry_meta)
        return payload, provider_meta

    raise structured_error


def provider_native_temperature_policy() -> str:
    return "provider_default_no_temperature_parameter_sent"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_clinical_history_redaction(raw_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    redacted = [
        str(case.get("case_id"))
        for case in raw_cases
        if case.get("clinical_history_redacted") or case.get("clinical_history_sanitized") or case.get("original_clinical_history")
    ]
    return {
        "enabled": bool(redacted),
        "n_redacted_cases": len(redacted),
        "redacted_case_ids": redacted,
    }


def build_code_version() -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    files = ["run_main.py", "pipeline_main.py", "judge.py"]
    return {
        "schema_version": "main_benchmark_v4_eurorad_best_effort_ambiguity",
        "agent_schema_version": AGENT_TURN_JSON_SCHEMA_NAME,
        "judge_schema_version": "judge_v5_schema_aligned_trajectory_scores",
        "rule_scorer_version": "rule_v8_rubric_extracted_terms",
        "evidence_unit_version": "eurorad_figure_protocol_v1",
        "followup_policy_version": "available_followup_v1",
        "route_denominator_policy_version": "route_evaluable_imaging_only_v1",
        "ambiguous_request_policy_version": "best_effort_best_score_not_penalized_v1",
        "temperature_policy": provider_native_temperature_policy(),
        "file_sha256": {name: file_sha256(here / name) for name in files if (here / name).exists()},
    }


class SessionProtocol:
    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        raise NotImplementedError


class ProviderBase:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.client = httpx.Client(timeout=settings.timeout)

    def create_session(self, system_prompt: str) -> SessionProtocol:
        raise NotImplementedError

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        session = self.create_session(system_prompt)
        reply = session.send(user_prompt, [])
        return {"text": reply.text, "raw": reply.raw, "provider_meta": reply.provider_meta}

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise ProviderError(f"Structured outputs are not implemented for provider adapter {self.__class__.__name__}")


# =========================
# OpenAI Responses API
# =========================

class OpenAIResponsesSession(SessionProtocol):
    def __init__(self, provider: "OpenAIResponsesProvider", system_prompt: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.previous_response_id: Optional[str] = None
        self._is_first_turn = True

    def _build_user_message(self, user_prompt: str, images: Sequence[ImagePayload]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for image in images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(image),
                    "detail": self.provider.settings.image_detail,
                }
            )
        return {"role": "user", "content": content}

    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        input_items: List[Dict[str, Any]] = []
        if self._is_first_turn:
            input_items.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                }
            )
        input_items.append(self._build_user_message(user_prompt, images))

        body: Dict[str, Any] = {
            "model": self.provider.settings.model,
            "input": input_items,
            "max_output_tokens": self.provider.settings.max_output_tokens,
            "store": True,
        }
        if self.provider.settings.reasoning_effort:
            body["reasoning"] = {"effort": self.provider.settings.reasoning_effort}
        if wants_agent_schema(self.provider.settings):
            add_openai_responses_json_schema(
                body,
                schema_name=AGENT_TURN_JSON_SCHEMA_NAME,
                schema=AGENT_TURN_JSON_SCHEMA,
                schema_description=AGENT_TURN_JSON_SCHEMA_DESCRIPTION,
            )
        if self.previous_response_id:
            body["previous_response_id"] = self.previous_response_id

        url = f"{self.provider.base_url}/responses"
        payload, schema_meta = post_json_with_agent_schema_fallback(
            client=self.provider.client,
            url=url,
            headers={
                "Authorization": f"Bearer {self.provider.settings.api_key}",
                "Content-Type": "application/json",
            },
            body=body,
            settings=self.provider.settings,
            structured_method="responses.text.format",
            remove_schema=lambda b: b.pop("text", None),
            error_prefix="OpenAI Responses API error",
        )
        text = openai_response_output_text(payload)
        self.previous_response_id = payload.get("id") or self.previous_response_id
        self._is_first_turn = False

        usage = payload.get("usage") or {}
        provider_meta = {"response_id": payload.get("id"), "usage": usage}
        provider_meta.update(schema_meta)
        return ModelReply(text=text, raw=payload, provider_meta=provider_meta)



class OpenAIResponsesProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        self.base_url = (settings.base_url or "https://api.openai.com/v1").rstrip("/")

    def create_session(self, system_prompt: str) -> SessionProtocol:
        return OpenAIResponsesSession(self, system_prompt)

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "max_output_tokens": self.settings.max_output_tokens,
            "store": False,
        }
        if self.settings.reasoning_effort:
            body["reasoning"] = {"effort": self.settings.reasoning_effort}
        response = self.client.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"OpenAI Responses API error {response.status_code}: {response.text}")
        payload = response.json()
        text = openai_response_output_text(payload)
        return {
            "text": text,
            "raw": payload,
            "provider_meta": {
                "response_id": payload.get("id"),
                "usage": payload.get("usage") or {},
            },
        }

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "max_output_tokens": self.settings.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "description": schema_description or schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if self.settings.reasoning_effort:
            body["reasoning"] = {"effort": self.settings.reasoning_effort}
        response = self.client.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"OpenAI Responses structured-output error {response.status_code}: {response.text}")
        payload = response.json()
        text = openai_response_output_text(payload)
        return {
            "text": text,
            "raw": payload,
            "provider_meta": {
                "response_id": payload.get("id"),
                "usage": payload.get("usage") or {},
                "structured_output_used": True,
                "structured_output_method": "responses.text.format",
                "structured_output_schema_name": schema_name,
            },
        }


# =========================
# OpenAI-compatible chat (Qwen fallback / local gateways)
# =========================

class OpenAICompatibleChatSession(SessionProtocol):
    def __init__(self, provider: "OpenAICompatibleChatProvider", system_prompt: str) -> None:
        self.provider = provider
        self.messages: List[Dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _build_user_content(self, user_prompt: str, images: Sequence[ImagePayload]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image)},
                }
            )
        return content

    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        self.messages.append(
            {
                "role": "user",
                "content": self._build_user_content(user_prompt, images),
            }
        )
        body: Dict[str, Any] = {
            "model": self.provider.settings.model,
            "messages": self.messages,
            "max_tokens": self.provider.settings.max_output_tokens,
        }
        if wants_agent_schema(self.provider.settings):
            add_chat_json_schema(
                body,
                schema_name=AGENT_TURN_JSON_SCHEMA_NAME,
                schema=AGENT_TURN_JSON_SCHEMA,
                schema_description=AGENT_TURN_JSON_SCHEMA_DESCRIPTION,
            )

        payload, schema_meta = post_json_with_agent_schema_fallback(
            client=self.provider.client,
            url=f"{self.provider.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.provider.settings.api_key}",
                "Content-Type": "application/json",
            },
            body=body,
            settings=self.provider.settings,
            structured_method="chat.response_format.json_schema",
            remove_schema=lambda b: b.pop("response_format", None),
            error_prefix="OpenAI-compatible chat error",
        )
        try:
            message = payload["choices"][0]["message"]
            text = openai_chat_message_text(message.get("content"))
        except Exception as exc:
            raise ProviderError(f"Unexpected chat payload: {payload}") from exc

        self.messages.append({"role": "assistant", "content": text})
        provider_meta = {"response_id": payload.get("id"), "usage": payload.get("usage") or {}}
        provider_meta.update(schema_meta)
        return ModelReply(text=text, raw=payload, provider_meta=provider_meta)



class OpenAICompatibleChatProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        self.base_url = (settings.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")

    def create_session(self, system_prompt: str) -> SessionProtocol:
        return OpenAICompatibleChatSession(self, system_prompt)

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.settings.max_output_tokens,
        }
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"OpenAI-compatible chat error {response.status_code}: {response.text}")
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
            text = openai_chat_message_text(message.get("content"))
        except Exception as exc:
            raise ProviderError(f"Unexpected chat payload: {payload}") from exc
        return {"text": text, "raw": payload, "provider_meta": {"response_id": payload.get("id"), "usage": payload.get("usage") or {}}}

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.settings.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "description": schema_description or schema_name,
                    "schema": schema,
                },
            },
        }
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"OpenAI-compatible structured-output error {response.status_code}: {response.text}")
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
            text = openai_chat_message_text(message.get("content"))
        except Exception as exc:
            raise ProviderError(f"Unexpected structured chat payload: {payload}") from exc
        return {
            "text": text,
            "raw": payload,
            "provider_meta": {
                "response_id": payload.get("id"),
                "usage": payload.get("usage") or {},
                "structured_output_used": True,
                "structured_output_method": "chat.response_format.json_schema",
                "structured_output_schema_name": schema_name,
            },
        }


# =========================
# Anthropic Messages API
# =========================

class AnthropicMessagesSession(SessionProtocol):
    def __init__(self, provider: "AnthropicMessagesProvider", system_prompt: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []

    def _image_block(self, image: ImagePayload) -> Dict[str, Any]:
        if image.path:
            data = read_binary(image.path)
            encoded = base64.b64encode(data).decode("ascii")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": detected_image_mime_type(data, image.mime_type),
                    "data": encoded,
                },
            }
        if image.url:
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image.url,
                },
            }
        raise FileNotFoundError(f"No accessible image source for {image}")

    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append(self._image_block(image))
        self.messages.append({"role": "user", "content": content})

        body: Dict[str, Any] = {
            "model": self.provider.settings.model,
            "system": self.system_prompt,
            "messages": self.messages,
            "max_tokens": self.provider.settings.max_output_tokens,
        }
        if wants_agent_schema(self.provider.settings):
            add_anthropic_json_schema(body, schema=AGENT_TURN_JSON_SCHEMA)

        payload, schema_meta = post_json_with_agent_schema_fallback(
            client=self.provider.client,
            url=f"{self.provider.base_url}/v1/messages",
            headers={
                "x-api-key": self.provider.settings.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body=body,
            settings=self.provider.settings,
            structured_method="messages.output_config.format",
            remove_schema=lambda b: b.pop("output_config", None),
            error_prefix="Anthropic Messages API error",
        )
        text = anthropic_output_text(payload)
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

        provider_meta = {"response_id": payload.get("id"), "usage": payload.get("usage") or {}}
        provider_meta.update(schema_meta)
        return ModelReply(text=text, raw=payload, provider_meta=provider_meta)



class AnthropicMessagesProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        self.base_url = (settings.base_url or "https://api.anthropic.com").rstrip("/")

    def create_session(self, system_prompt: str) -> SessionProtocol:
        return AnthropicMessagesSession(self, system_prompt)

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "max_tokens": self.settings.max_output_tokens,
        }
        response = self.client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.settings.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"Anthropic Messages API error {response.status_code}: {response.text}")
        payload = response.json()
        text = anthropic_output_text(payload)
        return {"text": text, "raw": payload, "provider_meta": {"response_id": payload.get("id"), "usage": payload.get("usage") or {}}}

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        del schema_description
        body = {
            "model": self.settings.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "max_tokens": self.settings.max_output_tokens,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        }
        response = self.client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.settings.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"Anthropic structured-output error {response.status_code}: {response.text}")
        payload = response.json()
        text = anthropic_output_text(payload)
        return {
            "text": text,
            "raw": payload,
            "provider_meta": {
                "response_id": payload.get("id"),
                "usage": payload.get("usage") or {},
                "structured_output_used": True,
                "structured_output_method": "messages.output_config.format",
                "structured_output_schema_name": schema_name,
            },
        }


class AnthropicVertexMessagesSession(SessionProtocol):
    def __init__(self, provider: "AnthropicVertexMessagesProvider", system_prompt: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []

    def _image_block(self, image: ImagePayload) -> Dict[str, Any]:
        if image.path:
            data = read_binary(image.path)
            encoded = base64.b64encode(data).decode("ascii")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": detected_image_mime_type(data, image.mime_type),
                    "data": encoded,
                },
            }
        if image.url:
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image.url,
                },
            }
        raise FileNotFoundError(f"No accessible image source for {image}")

    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append(self._image_block(image))
        self.messages.append({"role": "user", "content": content})

        body: Dict[str, Any] = {
            "model": self.provider.settings.model,
            "system": self.system_prompt,
            "messages": self.messages,
            "max_tokens": self.provider.settings.max_output_tokens,
        }
        if wants_agent_schema(self.provider.settings):
            add_anthropic_json_schema(body, schema=AGENT_TURN_JSON_SCHEMA)

        payload, schema_meta = anthropic_vertex_message_with_agent_schema_fallback(
            provider=self.provider,
            body=body,
            structured_method="messages.output_config.format",
            remove_schema=lambda b: b.pop("output_config", None),
            retry_operation="anthropic_vertex.messages.create",
        )
        text = anthropic_output_text(payload)
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

        provider_meta = {"response_id": payload.get("id"), "usage": payload.get("usage") or {}}
        provider_meta.update(schema_meta)
        return ModelReply(text=text, raw=payload, provider_meta=provider_meta)


class AnthropicVertexMessagesProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        if not settings.vertex_project or not settings.vertex_region:
            raise ProviderError(
                "provider=vertex with a Claude model requires --vertex-project and --vertex-region "
                "or VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT and VERTEX_REGION/VERTEX_LOCATION/GOOGLE_CLOUD_LOCATION."
            )
        try:
            from anthropic import AnthropicVertex
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderError("The `anthropic[vertex]` package is required for Claude on Vertex.") from exc

        access_token = os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
        kwargs: Dict[str, Any] = {
            "project_id": settings.vertex_project,
            "region": settings.vertex_region,
            "timeout": settings.timeout,
            "max_retries": 0,
        }
        if access_token:
            kwargs["access_token"] = access_token
        self.client = AnthropicVertex(**kwargs)

    def create_session(self, system_prompt: str) -> SessionProtocol:
        return AnthropicVertexMessagesSession(self, system_prompt)

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "max_tokens": self.settings.max_output_tokens,
        }
        payload, retry_meta = create_anthropic_vertex_message_with_retry(
            provider=self,
            body=body,
            operation="anthropic_vertex.messages.create",
        )
        text = anthropic_output_text(payload)
        provider_meta = {"response_id": payload.get("id"), "usage": payload.get("usage") or {}}
        provider_meta.update(retry_meta)
        return {"text": text, "raw": payload, "provider_meta": provider_meta}

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        del schema_description
        body = {
            "model": self.settings.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "max_tokens": self.settings.max_output_tokens,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        }
        payload, retry_meta = create_anthropic_vertex_message_with_retry(
            provider=self,
            body=body,
            operation="anthropic_vertex.messages.create_structured",
        )
        text = anthropic_output_text(payload)
        provider_meta = {
            "response_id": payload.get("id"),
            "usage": payload.get("usage") or {},
            "structured_output_used": True,
            "structured_output_method": "messages.output_config.format",
            "structured_output_schema_name": schema_name,
        }
        provider_meta.update(retry_meta)
        return {"text": text, "raw": payload, "provider_meta": provider_meta}


# =========================
# Gemini generateContent
# =========================

class GeminiGenerateContentSession(SessionProtocol):
    def __init__(self, provider: "GeminiGenerateContentProvider", system_prompt: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.contents: List[Dict[str, Any]] = []

    def _image_part(self, image: ImagePayload) -> Dict[str, Any]:
        if image.path:
            encoded = base64.b64encode(read_binary(image.path)).decode("ascii")
            return {
                "inline_data": {
                    "mime_type": image.mime_type,
                    "data": encoded,
                }
            }
        if image.url:
            # generateContent can also work with file references, but for this benchmark
            # local files are expected; URL fallback is provided for convenience only.
            return {
                "file_data": {
                    "mime_type": image.mime_type,
                    "file_uri": image.url,
                }
            }
        raise FileNotFoundError(f"No accessible image source for {image}")

    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        user_parts: List[Dict[str, Any]] = [{"text": user_prompt}]
        for image in images:
            user_parts.append(self._image_part(image))
        self.contents.append({"role": "user", "parts": user_parts})

        generation_config: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "maxOutputTokens": self.provider.settings.max_output_tokens,
        }
        add_gemini_rest_thinking_config(generation_config, self.provider.settings)
        if wants_agent_schema(self.provider.settings):
            generation_config["responseJsonSchema"] = AGENT_TURN_JSON_SCHEMA

        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": self.contents,
            "generationConfig": generation_config,
        }

        url = (
            f"{self.provider.base_url}/{self.provider.settings.gemini_api_version}"
            f"/models/{self.provider.settings.model}:generateContent"
        )
        payload, schema_meta = post_json_with_agent_schema_fallback(
            client=self.provider.client,
            url=url,
            headers={"Content-Type": "application/json"},
            params={"key": self.provider.settings.api_key},
            body=body,
            settings=self.provider.settings,
            structured_method="generateContent.responseJsonSchema",
            remove_schema=lambda b: b.get("generationConfig", {}).pop("responseJsonSchema", None),
            error_prefix="Gemini generateContent error",
            retry_google=True,
            retry_operation="gemini.agent.generateContent",
        )
        text = gemini_output_text(payload)
        self.contents.append({"role": "model", "parts": [{"text": text}]})

        provider_meta = {"usage": payload.get("usageMetadata") or {}}
        provider_meta.update(schema_meta)
        return ModelReply(text=text, raw=payload, provider_meta=provider_meta)



class GeminiGenerateContentProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        self.base_url = (settings.base_url or "https://generativelanguage.googleapis.com").rstrip("/")

    def create_session(self, system_prompt: str) -> SessionProtocol:
        return GeminiGenerateContentSession(self, system_prompt)

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        generation_config: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "maxOutputTokens": self.settings.max_output_tokens,
        }
        add_gemini_rest_thinking_config(generation_config, self.settings)
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        url = f"{self.base_url}/{self.settings.gemini_api_version}/models/{self.settings.model}:generateContent"
        payload, retry_meta = post_google_json_with_retry(
            client=self.client,
            url=url,
            params={"key": self.settings.api_key},
            headers={"Content-Type": "application/json"},
            body=body,
            settings=self.settings,
            operation="gemini.complete.generateContent",
            error_prefix="Gemini generateContent error",
        )
        text = gemini_output_text(payload)
        provider_meta = {"usage": payload.get("usageMetadata") or {}}
        provider_meta.update(retry_meta)
        return {"text": text, "raw": payload, "provider_meta": provider_meta}

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        del schema_description
        generation_config: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "maxOutputTokens": self.settings.max_output_tokens,
        }
        add_gemini_rest_thinking_config(generation_config, self.settings)
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        url = (
            f"{self.base_url}/{self.settings.gemini_api_version}"
            f"/models/{self.settings.model}:generateContent"
        )
        payload, retry_meta = post_google_json_with_retry(
            client=self.client,
            url=url,
            params={"key": self.settings.api_key},
            headers={"Content-Type": "application/json"},
            body=body,
            settings=self.settings,
            operation="gemini.structured.generateContent",
            error_prefix="Gemini structured-output error",
        )
        text = gemini_output_text(payload)
        provider_meta = {
            "usage": payload.get("usageMetadata") or {},
            "structured_output_used": True,
            "structured_output_method": "generateContent.responseJsonSchema",
            "structured_output_schema_name": schema_name,
        }
        provider_meta.update(retry_meta)
        return {
            "text": text,
            "raw": payload,
            "provider_meta": provider_meta,
        }



class VertexGenerateContentSession(SessionProtocol):
    def __init__(self, provider: "VertexGenerateContentProvider", system_prompt: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.contents: List[Any] = []

    def _image_part(self, image: ImagePayload) -> Any:
        try:
            from google.genai import types
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderError("The `google-genai` package is required for provider=vertex.") from exc

        if image.path:
            return types.Part.from_bytes(data=read_binary(image.path), mime_type=image.mime_type)
        if image.url:
            return types.Part.from_uri(file_uri=image.url, mime_type=image.mime_type)
        raise FileNotFoundError(f"No accessible image source for {image}")

    def send(self, user_prompt: str, images: Sequence[ImagePayload]) -> ModelReply:
        try:
            from google.genai import types
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderError("The `google-genai` package is required for provider=vertex.") from exc

        user_parts: List[Any] = [types.Part.from_text(text=user_prompt)]
        for image in images:
            user_parts.append(self._image_part(image))
        self.contents.append(types.Content(role="user", parts=user_parts))

        config: Dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "response_mime_type": "application/json",
            "max_output_tokens": self.provider.settings.max_output_tokens,
        }
        add_vertex_thinking_config(config, self.provider.settings)
        if wants_agent_schema(self.provider.settings):
            config["response_json_schema"] = AGENT_TURN_JSON_SCHEMA

        schema_meta: Dict[str, Any] = {
            "agent_structured_output_used": wants_agent_schema(self.provider.settings),
            "agent_structured_output_method": "vertex.generateContent.responseJsonSchema" if wants_agent_schema(self.provider.settings) else "prompt_json_only",
        }
        try:
            response, retry_meta = self.provider.generate_content_with_retry(
                contents=self.contents,
                config=config,
                operation="vertex.agent.generateContent",
            )
        except GoogleRetryExhaustedError:
            raise
        except Exception as exc:
            if wants_agent_schema(self.provider.settings) and not requires_agent_schema(self.provider.settings):
                fallback_error = repr(exc)
                config.pop("response_json_schema", None)
                response, retry_meta = self.provider.generate_content_with_retry(
                    contents=self.contents,
                    config=config,
                    operation="vertex.agent.generateContent.prompt_json_fallback",
                )
                schema_meta = {
                    "agent_structured_output_used": False,
                    "agent_structured_output_method": "prompt_json_fallback",
                    "agent_structured_output_fallback_error": fallback_error,
                }
            else:
                raise

        text = normalize_space(getattr(response, "text", ""))
        self.contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

        payload = self.provider.response_payload(response)
        provider_meta = {"usage": payload.get("usage_metadata") or payload.get("usageMetadata") or {}}
        provider_meta.update(schema_meta)
        provider_meta.update(retry_meta)
        return ModelReply(text=text, raw=payload, provider_meta=provider_meta)



class VertexGenerateContentProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderError("The `google-genai` package is required for provider=vertex.") from exc
        timeout_ms = int(max(settings.timeout, 1.0) * 1000)
        self.client = genai.Client(
            vertexai=True,
            api_key=settings.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    @staticmethod
    def response_payload(response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", by_alias=True, exclude_none=True)
        if hasattr(response, "dict"):
            return response.dict()
        return {"text": normalize_space(getattr(response, "text", ""))}

    def create_session(self, system_prompt: str) -> SessionProtocol:
        return VertexGenerateContentSession(self, system_prompt)

    def generate_content_with_retry(
        self,
        *,
        contents: Any,
        config: Dict[str, Any],
        operation: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        return call_google_with_retry(
            lambda: self.client.models.generate_content(
                model=self.settings.model,
                contents=contents,
                config=config,
            ),
            settings=self.settings,
            operation=operation,
        )

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "max_output_tokens": self.settings.max_output_tokens,
        }
        add_vertex_thinking_config(config, self.settings)
        response, retry_meta = self.generate_content_with_retry(
            contents=user_prompt,
            config=config,
            operation="vertex.complete.generateContent",
        )
        text = normalize_space(getattr(response, "text", ""))
        payload = self.response_payload(response)
        provider_meta = {"usage": payload.get("usage_metadata") or payload.get("usageMetadata") or {}}
        provider_meta.update(retry_meta)
        return {"text": text, "raw": payload, "provider_meta": provider_meta}

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        del schema_description
        config: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        add_vertex_thinking_config(config, self.settings)
        response, retry_meta = self.generate_content_with_retry(
            contents=user_prompt,
            config=config,
            operation="vertex.structured.generateContent",
        )
        text = normalize_space(getattr(response, "text", ""))
        payload = self.response_payload(response)
        provider_meta = {
            "usage": payload.get("usage_metadata") or payload.get("usageMetadata") or {},
            "structured_output_used": True,
            "structured_output_method": "vertex.generateContent.responseJsonSchema",
            "structured_output_schema_name": schema_name,
        }
        provider_meta.update(retry_meta)
        return {
            "text": text,
            "raw": payload,
            "provider_meta": provider_meta,
        }



# =========================
# Qwen provider router
# =========================

class QwenProvider(ProviderBase):
    def __init__(self, settings: ProviderSettings) -> None:
        super().__init__(settings)
        self.responses_provider = OpenAIResponsesProvider(
            ProviderSettings(
                provider="qwen-responses",
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=settings.timeout,
                max_output_tokens=settings.max_output_tokens,
                image_detail=settings.image_detail,
                reasoning_effort=settings.reasoning_effort,
                agent_structured_output=settings.agent_structured_output,
                gemini_thinking_level=settings.gemini_thinking_level,
                qwen_transport=settings.qwen_transport,
                gemini_api_version=settings.gemini_api_version,
            )
        )
        self.chat_provider = OpenAICompatibleChatProvider(
            ProviderSettings(
                provider="qwen-chat",
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=settings.timeout,
                max_output_tokens=settings.max_output_tokens,
                image_detail=settings.image_detail,
                reasoning_effort=settings.reasoning_effort,
                agent_structured_output=settings.agent_structured_output,
                gemini_thinking_level=settings.gemini_thinking_level,
                qwen_transport=settings.qwen_transport,
                gemini_api_version=settings.gemini_api_version,
            )
        )

    def create_session(self, system_prompt: str) -> SessionProtocol:
        transport = self.settings.qwen_transport.lower()
        if transport == "responses":
            return self.responses_provider.create_session(system_prompt)
        if transport == "chat":
            return self.chat_provider.create_session(system_prompt)

        # auto: local image roots are expected in this benchmark, so default to chat-compatible
        # transport unless the user explicitly forces Responses.
        return self.chat_provider.create_session(system_prompt)

    def complete_once(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # Text-only judge fallback uses chat without the agent schema.
        return self.chat_provider.complete_once(system_prompt, user_prompt)

    def complete_once_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        schema_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Use DashScope's OpenAI-compatible chat endpoint for judge-side structured JSON.
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "description": schema_description or schema_name,
                    "schema": schema,
                },
            },
        }
        response = self.chat_provider.client.post(
            f"{self.chat_provider.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"Qwen structured-output error {response.status_code}: {response.text}")
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
            text = openai_chat_message_text(message.get("content"))
        except Exception as exc:
            raise ProviderError(f"Unexpected Qwen structured payload: {payload}") from exc
        return {
            "text": text,
            "raw": payload,
            "provider_meta": {
                "response_id": payload.get("id"),
                "usage": payload.get("usage") or {},
                "structured_output_used": True,
                "structured_output_method": "dashscope.chat.response_format.json_schema",
                "structured_output_schema_name": schema_name,
            },
        }


# =========================
# Provider factory
# =========================

def env_default(provider: str) -> tuple[str, Optional[str]]:
    provider = provider.lower()
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", ""), os.getenv("OPENAI_BASE_URL")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("ANTHROPIC_BASE_URL")
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""), os.getenv("GEMINI_BASE_URL")
    if provider == "vertex":
        return os.getenv("VERTEX_API_KEY", ""), None
    if provider == "qwen":
        return os.getenv("DASHSCOPE_API_KEY", ""), os.getenv("DASHSCOPE_BASE_URL")
    raise ValueError(f"Unsupported provider: {provider}")


def build_provider(
    *,
    provider_name: str,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float,
    max_output_tokens: int,
    image_detail: str,
    reasoning_effort: Optional[str],
    agent_structured_output: str,
    gemini_thinking_level: Optional[str],
    qwen_transport: str,
    gemini_api_version: str,
    vertex_project: Optional[str],
    vertex_region: Optional[str],
) -> ProviderBase:
    default_key, default_base = env_default(provider_name)
    resolved_api_key = api_key or default_key
    resolved_base = base_url or default_base
    provider_name = provider_name.lower()
    resolved_vertex_project = vertex_project or os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    resolved_vertex_region = vertex_region or os.getenv("VERTEX_REGION") or os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION")

    if not resolved_api_key and not (provider_name == "vertex" and model_name.lower().startswith("claude")):
        raise ProviderError(f"Missing API key for provider={provider_name}")

    settings = ProviderSettings(
        provider=provider_name,
        model=model_name,
        api_key=resolved_api_key,
        base_url=resolved_base,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        image_detail=image_detail,
        reasoning_effort=reasoning_effort,
        agent_structured_output=agent_structured_output,
        gemini_thinking_level=gemini_thinking_level,
        qwen_transport=qwen_transport,
        gemini_api_version=gemini_api_version,
        vertex_project=resolved_vertex_project,
        vertex_region=resolved_vertex_region,
    )

    if provider_name == "openai":
        return OpenAIResponsesProvider(settings)
    if provider_name == "anthropic":
        return AnthropicMessagesProvider(settings)
    if provider_name == "gemini":
        return GeminiGenerateContentProvider(settings)
    if provider_name == "vertex":
        if model_name.lower().startswith("claude"):
            return AnthropicVertexMessagesProvider(settings)
        return VertexGenerateContentProvider(settings)
    if provider_name == "qwen":
        return QwenProvider(settings)
    raise ValueError(f"Unsupported provider: {provider_name}")


def make_judge_model_call(provider: ProviderBase, structured_policy: str) -> callable:
    structured_policy = (structured_policy or "auto").lower()
    if structured_policy not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported judge structured-output policy: {structured_policy}")

    def _call(*, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if structured_policy != "never":
            try:
                response = provider.complete_once_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema_name=JUDGE_JSON_SCHEMA_NAME,
                    schema=JUDGE_JSON_SCHEMA,
                    schema_description=JUDGE_JSON_SCHEMA_DESCRIPTION,
                )
                provider_meta = response.setdefault("provider_meta", {})
                provider_meta.setdefault("structured_output_used", True)
                provider_meta.setdefault("structured_output_mode", "native_structured_output")
                provider_meta.setdefault("structured_output_schema_name", JUDGE_JSON_SCHEMA_NAME)
                return response
            except GoogleRetryExhaustedError:
                raise
            except Exception as exc:
                if structured_policy == "always":
                    raise
                fallback = provider.complete_once(system_prompt, user_prompt)
                provider_meta = fallback.setdefault("provider_meta", {})
                provider_meta["structured_output_used"] = False
                provider_meta["structured_output_mode"] = "prompt_json_fallback"
                provider_meta["structured_output_schema_name"] = JUDGE_JSON_SCHEMA_NAME
                provider_meta["structured_output_fallback_error"] = repr(exc)
                return fallback

        fallback = provider.complete_once(system_prompt, user_prompt)
        provider_meta = fallback.setdefault("provider_meta", {})
        provider_meta["structured_output_used"] = False
        provider_meta["structured_output_mode"] = "prompt_json_only"
        provider_meta["structured_output_schema_name"] = JUDGE_JSON_SCHEMA_NAME
        return fallback

    return _call


def summarize_judge_transport(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_llm_attempted_cases = 0
    total_llm_success_cases = 0
    structured_success = 0
    prompt_json_fallback = 0
    prompt_json_only = 0
    llm_error_cases = 0
    fallback_errors: Dict[str, int] = {}
    llm_errors: Dict[str, int] = {}
    methods: Dict[str, int] = {}

    for record in case_results:
        judge = record.get("judge") or {}
        llm_payload = ((judge.get("by_mode") or {}).get("llm")) or {}
        llm_error = normalize_space(((judge.get("errors") or {}).get("llm")) or "")
        llm_attempted = bool(llm_payload) or bool(llm_error)
        if not llm_attempted:
            continue

        total_llm_attempted_cases += 1
        if llm_payload:
            total_llm_success_cases += 1
            provider_meta = llm_payload.get("provider_meta") or {}
            method = normalize_space(provider_meta.get("structured_output_method") or provider_meta.get("structured_output_mode") or "unknown")
            methods[method] = methods.get(method, 0) + 1
            if provider_meta.get("structured_output_used") is True:
                structured_success += 1
            elif provider_meta.get("structured_output_mode") == "prompt_json_fallback":
                prompt_json_fallback += 1
            else:
                prompt_json_only += 1
            fallback_error = normalize_space(provider_meta.get("structured_output_fallback_error") or "")
            if fallback_error:
                fallback_errors[fallback_error] = fallback_errors.get(fallback_error, 0) + 1
        if llm_error:
            llm_error_cases += 1
            llm_errors[llm_error] = llm_errors.get(llm_error, 0) + 1

    return {
        "n_llm_attempted_cases": total_llm_attempted_cases,
        "n_llm_success_cases": total_llm_success_cases,
        "n_llm_error_cases": llm_error_cases,
        "structured_success_cases": structured_success,
        "prompt_json_fallback_cases": prompt_json_fallback,
        "prompt_json_only_cases": prompt_json_only,
        "structured_success_rate_over_attempted": (structured_success / total_llm_attempted_cases) if total_llm_attempted_cases else 0.0,
        "structured_success_rate_over_successful": (structured_success / total_llm_success_cases) if total_llm_success_cases else 0.0,
        "fallback_error_counts": dict(sorted(fallback_errors.items(), key=lambda item: (-item[1], item[0]))),
        "llm_error_counts": dict(sorted(llm_errors.items(), key=lambda item: (-item[1], item[0]))),
        "method_counts": dict(sorted(methods.items(), key=lambda item: (-item[1], item[0]))),
    }


def split_warning_flags(value: Any) -> List[str]:
    text = normalize_space(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def summarize_target_output_health(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_cases = len(case_results)
    n_turns = 0
    parse_warning_turns = 0
    warning_counts: Dict[str, int] = {}
    structured_used_turns = 0
    structured_fallback_turns = 0
    prompt_json_only_turns = 0
    structured_method_counts: Dict[str, int] = {}
    fallback_error_counts: Dict[str, int] = {}

    for record in case_results:
        for turn in record.get("turns") or []:
            n_turns += 1
            flags = split_warning_flags(turn.get("parse_warning"))
            if flags:
                parse_warning_turns += 1
            for flag in flags:
                warning_counts[flag] = warning_counts.get(flag, 0) + 1

            meta = turn.get("provider_meta") or {}
            method = normalize_space(meta.get("agent_structured_output_method") or "unknown")
            if meta.get("agent_structured_output_used") is True:
                structured_used_turns += 1
                structured_method_counts[method] = structured_method_counts.get(method, 0) + 1
            elif method == "prompt_json_fallback":
                structured_fallback_turns += 1
                structured_method_counts[method] = structured_method_counts.get(method, 0) + 1
                err = normalize_space(meta.get("agent_structured_output_fallback_error") or "")
                if err:
                    fallback_error_counts[err] = fallback_error_counts.get(err, 0) + 1
            elif method == "prompt_json_only":
                prompt_json_only_turns += 1
                structured_method_counts[method] = structured_method_counts.get(method, 0) + 1

    return {
        "n_cases": n_cases,
        "n_turns": n_turns,
        "parse_warning_turns": parse_warning_turns,
        "parse_warning_rate": (parse_warning_turns / n_turns) if n_turns else 0.0,
        "warning_counts": dict(sorted(warning_counts.items(), key=lambda item: (-item[1], item[0]))),
        "json_parse_failed_turns": warning_counts.get("json_parse_failed", 0),
        "padded_to_4_turns": warning_counts.get("padded_to_4", 0),
        "trimmed_to_top4_turns": warning_counts.get("trimmed_to_top4", 0),
        "missing_action_turns": warning_counts.get("action_missing_default_request", 0),
        "missing_request_turns": warning_counts.get("missing_request_kept_empty", 0),
        "probability_renormalized_turns": warning_counts.get("probability_renormalized", 0),
        "uniform_probabilities_used_turns": warning_counts.get("uniform_probabilities_used", 0),
        "missing_final_location_turns": warning_counts.get("missing_final_location", 0),
        "agent_structured_output_used_turns": structured_used_turns,
        "agent_structured_output_fallback_turns": structured_fallback_turns,
        "prompt_json_only_turns": prompt_json_only_turns,
        "agent_structured_output_method_counts": dict(sorted(structured_method_counts.items(), key=lambda item: (-item[1], item[0]))),
        "agent_structured_output_fallback_error_counts": dict(sorted(fallback_error_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def summarize_request_outcomes(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    matched_resolution_reason_counts: Dict[str, int] = {}
    total_requests = 0
    ambiguous_resolved_total = 0
    per_case: List[Dict[str, Any]] = []
    for record in case_results:
        case_counts: Dict[str, int] = {}
        case_matched_reason_counts: Dict[str, int] = {}
        case_total = 0
        case_ambiguous_resolved = 0
        for req in record.get("requests") or []:
            total_requests += 1
            case_total += 1
            if req.get("outcome") == "matched":
                key = "matched"
                reason = normalize_space(req.get("resolution_reason") or "matched") or "matched"
                matched_resolution_reason_counts[reason] = matched_resolution_reason_counts.get(reason, 0) + 1
                case_matched_reason_counts[reason] = case_matched_reason_counts.get(reason, 0) + 1
                if req.get("ambiguity_resolved") or "ambiguous" in normalize_key(reason):
                    ambiguous_resolved_total += 1
                    case_ambiguous_resolved += 1
            else:
                key = normalize_space(req.get("invalid_reason") or "invalid_unknown") or "invalid_unknown"
            counts[key] = counts.get(key, 0) + 1
            case_counts[key] = case_counts.get(key, 0) + 1
        if case_total:
            case_invalid = case_total - case_counts.get("matched", 0)
            per_case.append({
                "case_id": record.get("case_id"),
                "total_requests": case_total,
                "matched_requests": case_counts.get("matched", 0),
                "invalid_requests": case_invalid,
                "invalid_request_rate": case_invalid / case_total,
                "ambiguous_resolved_requests": case_ambiguous_resolved,
                "counts": dict(sorted(case_counts.items(), key=lambda item: (-item[1], item[0]))),
                "matched_resolution_reason_counts": dict(sorted(case_matched_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
            })
    invalid_total = total_requests - counts.get("matched", 0)
    return {
        "total_requests": total_requests,
        "matched_requests": counts.get("matched", 0),
        "invalid_requests": invalid_total,
        "invalid_request_rate": (invalid_total / total_requests) if total_requests else 0.0,
        "ambiguous_resolved_requests": ambiguous_resolved_total,
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "matched_resolution_reason_counts": dict(sorted(matched_resolution_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "per_case": per_case,
    }


# =========================
# Run orchestration
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official main benchmark pipeline for EuroRad active DDx.")
    parser.add_argument("--data-path", required=True, help="Path to eurorad_neuro_01 JSON.")
    parser.add_argument("--image-root", required=True, help="Root directory for local images referenced by image_paths.")
    parser.add_argument("--out-dir", required=True, help="Directory for run outputs.")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini", "vertex", "qwen"])
    parser.add_argument("--target-model", required=True, help="Model name for the target agent.")
    parser.add_argument("--judge-provider", default="vertex", choices=["openai", "anthropic", "gemini", "vertex", "qwen"])
    parser.add_argument("--judge-model", default="gemini-3-flash-preview", help="Model name for the judge. Default: gemini-3-flash-preview on Vertex.")
    parser.add_argument("--judge-modes", default="both", choices=["both", "llm", "rule"], help="Enable LLM judge, rule-based judge, or both (default).")
    parser.add_argument("--judge-structured-output", default="auto", choices=["auto", "always", "never"], help="Judge-only structured-output policy: auto tries provider-native JSON schema first and falls back to prompt-JSON; always requires native structured output; never disables it.")
    parser.add_argument("--api-key", default=None, help="Optional explicit API key for the target provider.")
    parser.add_argument("--judge-api-key", default=None, help="Optional explicit API key for the judge provider.")
    parser.add_argument("--base-url", default=None, help="Optional explicit base URL for the target provider.")
    parser.add_argument("--judge-base-url", default=None, help="Optional explicit base URL for the judge provider.")
    parser.add_argument("--budget", type=int, default=6, help="Request budget B for the official sequential setting.")
    parser.add_argument("--reveal-unit", default="eurorad", choices=["eurorad", "figure"], help="Evidence reveal unit. Default eurorad/figure keeps raw EuroRad imaging_examination figure-protocol units without sequence merging.")
    parser.add_argument("--trajectory-horizon", type=int, default=None, help="Fixed T_max for S_traj. Defaults to budget+2 when omitted.")
    parser.add_argument("--diagnostic-threshold", type=float, default=2.0/3.0, help="Tau for time-to-diagnostic-guess and time-to-clinically-acceptable-diagnosis. Default 2/3.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of cases to run.")
    parser.add_argument("--case-id", action="append", default=None, help="Optional case_id filter; can be passed multiple times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None, help="Deprecated; accepted for compatibility but ignored. Provider defaults are always used.")
    parser.add_argument("--judge-temperature", type=float, default=None, help="Deprecated; accepted for compatibility but ignored. Provider defaults are always used.")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--image-detail", default="auto", choices=["low", "high", "auto", "original"])
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"], help="Optional reasoning effort for providers that support it. Omitted by default, so provider/model defaults apply.")
    parser.add_argument("--agent-structured-output", default="auto", choices=["auto", "always", "never"], help="Target-agent structured-output policy. auto uses provider-native JSON schema when available and falls back to prompt JSON.")
    parser.add_argument("--gemini-thinking-level", default=None, choices=["minimal", "low", "medium", "high"], help="Optional Gemini/Vertex thinking level. Omitted by default, so Gemini defaults apply.")
    parser.add_argument("--qwen-transport", default="auto", choices=["auto", "responses", "chat"], help="DashScope transport mode for Qwen.")
    parser.add_argument("--gemini-api-version", default="v1beta", help="Gemini REST API version, default v1beta.")
    parser.add_argument("--vertex-project", default=None, help="Optional Google Cloud project ID for Claude on Vertex. Defaults to VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT.")
    parser.add_argument("--vertex-region", default=None, help="Optional Google Cloud region for Claude on Vertex, for example us-east5. Defaults to VERTEX_REGION, VERTEX_LOCATION, or GOOGLE_CLOUD_LOCATION.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running remaining cases when a case fails.")
    parser.add_argument("--allow-preflight-errors", action="store_true", help="Run anyway when dataset preflight finds blocking errors such as missing local images. Default is to abort before API calls.")
    parser.add_argument("--model-label", default=None, help="Optional label used in Table 2 row outputs.")
    return parser.parse_args()


def filter_raw_cases(raw_cases: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = raw_cases
    if args.case_id:
        wanted = {str(case_id) for case_id in args.case_id}
        selected = [case for case in selected if str(case.get("case_id")) in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def run() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    image_root = Path(args.image_root)
    out_root = Path(args.out_dir)

    raw_cases = read_json(data_path)
    if not isinstance(raw_cases, list):
        raise RuntimeError(f"Expected a list of cases in {data_path}, got {type(raw_cases)}")
    raw_cases = filter_raw_cases(raw_cases, args)

    run_name = f"{slugify(args.provider)}-{slugify(args.target_model)}-{now_timestamp()}"
    run_dir = out_root / run_name
    ensure_dir(run_dir)

    canonical_cases: List[CanonicalCase] = [canonicalize_case(raw_case, image_root=image_root, reveal_unit=args.reveal_unit) for raw_case in raw_cases]
    dataset_preflight = build_dataset_preflight(canonical_cases)
    preflight_issues = list(dataset_preflight.get("issues") or [])
    run_health = {
        "has_blocking_preflight_error": any(issue.get("severity") == "error" for issue in preflight_issues),
        "preflight_errors": [issue for issue in preflight_issues if issue.get("severity") == "error"],
        "preflight_warnings": [issue for issue in preflight_issues if issue.get("severity") == "warning"],
    }

    config_preview = {
        "data_path": str(data_path),
        "data_path_sha256": file_sha256(data_path),
        "input_data_name": data_path.name,
        "image_root": str(image_root),
        "reveal_unit": args.reveal_unit,
        "clinical_history_redaction": summarize_clinical_history_redaction(raw_cases),
        "provider": args.provider,
        "target_model": args.target_model,
        "judge_provider": (args.judge_provider or args.provider) if args.judge_modes in {"both", "llm"} else None,
        "judge_model": (args.judge_model or args.target_model) if args.judge_modes in {"both", "llm"} else None,
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
        "deprecated_temperature_argument_ignored": args.temperature,
        "deprecated_judge_temperature_argument_ignored": args.judge_temperature if args.judge_modes in {"both", "llm"} else None,
        "max_output_tokens": args.max_output_tokens,
        "judge_max_output_tokens": args.judge_max_output_tokens if args.judge_modes in {"both", "llm"} else None,
        "timeout": args.timeout,
        "image_detail": args.image_detail,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_effort_policy": "provider_default_when_none",
        "agent_structured_output": args.agent_structured_output,
        "gemini_thinking_level": args.gemini_thinking_level,
        "gemini_thinking_policy": "provider_default_when_none",
        "qwen_transport": args.qwen_transport,
        "gemini_api_version": args.gemini_api_version,
        "vertex_project": args.vertex_project or os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"),
        "vertex_region": args.vertex_region or os.getenv("VERTEX_REGION") or os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION"),
        "allow_preflight_errors": bool(args.allow_preflight_errors),
    }

    if run_health["has_blocking_preflight_error"] and not args.allow_preflight_errors:
        summary_payload = {
            "run_config": config_preview,
            "run_health": run_health,
            "dataset_preflight": dataset_preflight,
            "failures": [],
            "enabled_scoring_modes": [],
            "default_scoring_mode": None,
            "table2_rows": {},
            "aggregate_by_mode": {},
            "judge_mode_agreement": {},
            "judge_transport": {},
            "metric_notes": build_metric_notes(),
            "code_version": build_code_version(),
            "target_output_health": {},
            "request_outcome_counts": {},
            "aborted_before_api_calls": True,
            "abort_reason": "blocking_dataset_preflight_error",
        }
        write_json(run_dir / "benchmark_summary.json", summary_payload)
        write_json(run_dir / "benchmark_full.json", {**summary_payload, "case_results": []})
        print(f"[ABORT] Blocking dataset preflight error. No API calls were made. Saved preflight summary to: {run_dir}", file=sys.stderr)
        return 2

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

    judge_provider_name = args.judge_provider or args.provider
    judge_model_name = args.judge_model or args.target_model
    judge_provider = None
    if enable_llm_judge:
        judge_provider = build_provider(
            provider_name=judge_provider_name,
            model_name=judge_model_name,
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

    pipeline = MainBenchmarkPipeline(
        target_session_factory=lambda system_prompt: target_provider.create_session(system_prompt),
        judge_runner=judge_runner,
        request_budget=args.budget,
        random_seed=args.seed,
        trajectory_horizon=args.trajectory_horizon,
        diagnostic_threshold=args.diagnostic_threshold,
        reveal_unit=args.reveal_unit,
    )

    config = {
        "data_path": str(data_path),
        "data_path_sha256": file_sha256(data_path),
        "input_data_name": data_path.name,
        "image_root": str(image_root),
        "reveal_unit": args.reveal_unit,
        "clinical_history_redaction": summarize_clinical_history_redaction(raw_cases),
        "provider": args.provider,
        "target_model": args.target_model,
        "judge_provider": judge_provider_name if enable_llm_judge else None,
        "judge_model": judge_model_name if enable_llm_judge else None,
        "judge_modes": args.judge_modes,
        "judge_structured_output": args.judge_structured_output if enable_llm_judge else None,
        "budget": args.budget,
        "trajectory_horizon": args.trajectory_horizon if args.trajectory_horizon is not None else args.budget + 2,
        "trajectory_horizon_policy": "explicit" if args.trajectory_horizon is not None else "default_budget_plus_2",
        "diagnostic_threshold": args.diagnostic_threshold,
        "limit": args.limit,
        "case_id": args.case_id,
        "seed": args.seed,
        "temperature_policy": provider_native_temperature_policy(),
        "deprecated_temperature_argument_ignored": args.temperature,
        "deprecated_judge_temperature_argument_ignored": args.judge_temperature if args.judge_modes in {"both", "llm"} else None,
        "max_output_tokens": args.max_output_tokens,
        "judge_max_output_tokens": args.judge_max_output_tokens if enable_llm_judge else None,
        "timeout": args.timeout,
        "image_detail": args.image_detail,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_effort_policy": "provider_default_when_none",
        "agent_structured_output": args.agent_structured_output,
        "gemini_thinking_level": args.gemini_thinking_level,
        "gemini_thinking_policy": "provider_default_when_none",
        "qwen_transport": args.qwen_transport,
        "gemini_api_version": args.gemini_api_version,
        "vertex_project": args.vertex_project or os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"),
        "vertex_region": args.vertex_region or os.getenv("VERTEX_REGION") or os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION"),
        "allow_preflight_errors": bool(args.allow_preflight_errors),
    }

    case_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    class _Obj:
        def __init__(self, payload: Dict[str, Any], metric_mode: str) -> None:
            self.__dict__.update(payload)
            self.metrics = (payload.get("metrics_by_mode_raw") or {}).get(metric_mode) or {}
            self.metric_status = (payload.get("metric_status_by_mode") or {}).get(metric_mode) or {}
            self.metrics_display = (payload.get("metrics_by_mode") or {}).get(metric_mode) or {}

    enabled_modes: List[str] = []
    if enable_llm_judge:
        enabled_modes.append("llm")
    if enable_rule_judge:
        enabled_modes.append("rule")

    def build_run_payloads(*, is_final: bool) -> tuple[Dict[str, Any], Dict[str, Any]]:
        aggregate_by_mode: Dict[str, Any] = {}
        table2_rows: Dict[str, Any] = {}
        for mode in enabled_modes:
            aggregate = aggregate_results(
                [_Obj(record, mode) for record in case_results],
                model_label=args.model_label or args.target_model,
                n_bootstrap=1000,
                seed=args.seed,
            )
            aggregate_by_mode[mode] = aggregate
            table2_rows[mode] = aggregate["table2_row"]

        mode_agreement: Dict[str, Any] = {}
        if {"llm", "rule"} <= set(enabled_modes):
            per_case_scores = {
                key: []
                for key in ["S_dx", "S_loc", "S_ddx", "S_ER", "B_opt", "B_inv", "S_order", "S_traj", "S_traj_dx_actual", "S_traj_conf_actual", "S_conf"]
            }
            trajectory_label_agreements: List[float] = []
            trajectory_score_maes: List[float] = []
            for record in case_results:
                llm_metrics = ((record.get("metrics_by_mode_raw") or {}).get("llm")) or {}
                rule_metrics = ((record.get("metrics_by_mode_raw") or {}).get("rule")) or {}
                for key in per_case_scores:
                    lv = llm_metrics.get(key)
                    rv = rule_metrics.get(key)
                    if lv is not None and rv is not None:
                        per_case_scores[key].append(abs(float(lv) - float(rv)))
                agreement_payload = ((record.get("judge") or {}).get("agreement") or {})
                agreement = agreement_payload.get("trajectory_label_agreement")
                if agreement is not None:
                    trajectory_label_agreements.append(float(agreement))
                score_mae = agreement_payload.get("trajectory_score_mae")
                if score_mae is not None:
                    trajectory_score_maes.append(float(score_mae))
            mode_agreement = {
                "mean_absolute_metric_difference": {
                    key: {
                        "value": (sum(vals) / len(vals) if vals else 0.0),
                        "defined": bool(vals),
                        "n_compared": len(vals),
                        "reason": ("defined" if vals else "no_cases_with_metric_in_both_modes"),
                    }
                    for key, vals in per_case_scores.items()
                },
                "avg_trajectory_label_agreement": {
                    "value": (sum(trajectory_label_agreements) / len(trajectory_label_agreements) if trajectory_label_agreements else 0.0),
                    "defined": bool(trajectory_label_agreements),
                    "n_compared": len(trajectory_label_agreements),
                    "reason": ("defined" if trajectory_label_agreements else "no_cases_with_trajectory_labels_in_both_modes"),
                },
                "avg_trajectory_score_mae_raw_0_to_3": {
                    "value": (sum(trajectory_score_maes) / len(trajectory_score_maes) if trajectory_score_maes else 0.0),
                    "defined": bool(trajectory_score_maes),
                    "n_compared": len(trajectory_score_maes),
                    "reason": ("defined" if trajectory_score_maes else "no_cases_with_trajectory_scores_in_both_modes"),
                },
            }

        metric_notes = build_metric_notes()
        judge_transport = summarize_judge_transport(case_results)
        target_output_health = summarize_target_output_health(case_results)
        request_outcome_counts = summarize_request_outcomes(case_results)
        code_version = build_code_version()
        preflight_issues = list(dataset_preflight.get("issues") or [])
        current_run_health = {
            "has_blocking_preflight_error": any(issue.get("severity") == "error" for issue in preflight_issues),
            "preflight_errors": [issue for issue in preflight_issues if issue.get("severity") == "error"],
            "preflight_warnings": [issue for issue in preflight_issues if issue.get("severity") == "warning"],
        }

        summary_payload = {
            "run_progress": {
                "is_final": is_final,
                "completed_cases": len(case_results),
                "failed_cases": len(failures),
                "attempted_cases": len(case_results) + len(failures),
                "total_cases": len(canonical_cases),
            },
            "run_config": config,
            "run_health": current_run_health,
            "dataset_preflight": dataset_preflight,
            "failures": failures,
            "enabled_scoring_modes": enabled_modes,
            "default_scoring_mode": "llm" if "llm" in enabled_modes else "rule",
            "table2_rows": table2_rows,
            "aggregate_by_mode": aggregate_by_mode,
            "judge_mode_agreement": mode_agreement,
            "judge_transport": judge_transport,
            "target_output_health": target_output_health,
            "request_outcome_counts": request_outcome_counts,
            "metric_notes": metric_notes,
            "code_version": code_version,
        }
        full_payload = dict(summary_payload)
        full_payload["case_results"] = case_results
        return summary_payload, full_payload

    def write_run_outputs(*, is_final: bool) -> None:
        summary_payload, full_payload = build_run_payloads(is_final=is_final)
        write_json(run_dir / "benchmark_summary.json", summary_payload)
        write_json(run_dir / "benchmark_full.json", full_payload)

    for case in canonical_cases:
        case_id = str(case.case_id or "unknown_case")
        try:
            result = pipeline.run_case(case)
            case_results.append(result.to_dict())
            write_run_outputs(is_final=False)
            print(f"[OK] case_id={case_id}", flush=True)
        except Exception as exc:
            failure = {"case_id": case_id, "error": repr(exc)}
            failures.append(failure)
            write_run_outputs(is_final=False)
            print(f"[FAIL] case_id={case_id} error={exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    write_run_outputs(is_final=True)

    print(f"Saved run outputs to: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
