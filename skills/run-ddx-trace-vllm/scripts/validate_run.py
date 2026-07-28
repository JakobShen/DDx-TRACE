#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def latest_full(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.rglob("benchmark_full.json"))
    if not candidates:
        raise FileNotFoundError(f"No benchmark_full.json found under {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--expected-cases", type=int, default=3)
    parser.add_argument("--expected-provider", default="vllm")
    parser.add_argument("--expected-model", required=True)
    args = parser.parse_args()

    full_path = latest_full(args.run_path)
    payload: dict[str, Any] = json.loads(full_path.read_text())
    progress = payload.get("run_progress") or {}
    config = payload.get("run_config") or {}
    health = payload.get("run_health") or {}
    output_health = payload.get("target_output_health") or {}

    checks = {
        "is_final": progress.get("is_final") is True,
        "completed_cases": progress.get("completed_cases") == args.expected_cases,
        "failed_cases": progress.get("failed_cases") == 0,
        "attempted_cases": progress.get("attempted_cases") == args.expected_cases,
        "provider": config.get("provider") == args.expected_provider,
        "target_model": config.get("target_model") == args.expected_model,
        "no_blocking_preflight_error": health.get("has_blocking_preflight_error") is False,
        "json_parse_failed_turns": output_health.get("json_parse_failed_turns") == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    summary = {
        "benchmark_full": str(full_path),
        "checks": checks,
        "run_progress": progress,
        "target_output_health": output_health,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        print(f"Validation failed: {', '.join(failures)}")
        return 1
    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
