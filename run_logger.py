"""Module 5: run logger.

Writes one JSON line per model attempt (the D8 run record) so every result can be
reproduced and audited. The log is append-only and flushed after every write, so a
crash or a shutdown never loses finished work, and a later run can RESUME.

A record holds: run id, alert id, condition, model, prompt format, sampling
settings, seed, repetition, attempt number, token counts, latency, RAM, raw output,
parsed output, validity flags and safety flags. Alerts themselves live in the
experiment files (exp_d2.jsonl); records point to them by alert_id.

Also writes a manifest (environment, versions, model details) next to the log.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path


def meminfo_used_mb() -> float | None:
    """System memory in use (total - available), in MB. Linux only; None elsewhere."""
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, val = line.split(":", 1)
                info[key] = int(val.split()[0])
        return round((info["MemTotal"] - info["MemAvailable"]) / 1024, 1)
    except Exception:
        return None


def meminfo_total_mb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            return round(int(f.readline().split()[1]) / 1024, 1)
    except Exception:
        return None


def _http_json(url: str, payload: dict | None = None, timeout: int = 5):
    try:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                     method="POST" if payload is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def model_details(host: str, model: str) -> dict:
    """Best effort: parameter size and quantisation of a pulled model."""
    show = _http_json(host.rstrip("/") + "/api/show", {"model": model}) or {}
    d = show.get("details") or {}
    return {"model": model, "parameter_size": d.get("parameter_size"),
            "quantization_level": d.get("quantization_level"), "family": d.get("family")}


def environment_info(host: str, models: list[str]) -> dict:
    version = _http_json(host.rstrip("/") + "/api/version") or {}
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": meminfo_total_mb(),
        "ollama_version": version.get("version"),
        "models": [model_details(host, m) for m in models],
    }


def make_condition(model: str, prompt_format: str, temperature: float, mode: str = "A",
                   top_p: float = 0.9, output_format: str = "schema") -> str:
    """One readable key per experimental condition."""
    return f"{model}|{prompt_format}|T{temperature:g}|p{top_p:g}|{output_format}|mode{mode}"


class RunLogger:
    def __init__(self, path: str, experiment_id: str):
        self.path = Path(path)
        self.experiment_id = experiment_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def log_attempt(self, *, alert_id: str, condition: str, repetition: int, attempt: dict,
                    final: bool, mode: str = "A", quantisation: str | None = None) -> dict:
        """Log one model attempt. `attempt` is an item of assess_validated()['attempts']."""
        s, v = attempt["settings"], attempt["validation"]
        rec = {
            "run_id": f"{self.experiment_id}|{condition}|{alert_id}|r{repetition:02d}|a{attempt['attempt']}",
            "experiment_id": self.experiment_id, "alert_id": alert_id, "condition": condition, "mode": mode,
            "model": s["model"], "quantisation": quantisation, "prompt_format": attempt["prompt_format"],
            "temperature": s["temperature"], "top_p": s["top_p"], "max_tokens": s["max_tokens"],
            "seed": s["seed"], "num_ctx": s["num_ctx"], "output_format": s["output_format"],
            "repetition": repetition, "attempt": attempt["attempt"], "is_retry": attempt["is_retry"],
            "final": final, "started_at": attempt["started_at"],
            "input_tokens": attempt["input_tokens"], "output_tokens": attempt["output_tokens"],
            "latency_s": attempt["latency_s"], "load_s": attempt["load_s"],
            "tokens_per_s": attempt["tokens_per_s"], "ram_used_mb": meminfo_used_mb(),
            "raw_output": attempt["raw_output"], "parsed": v["parsed"], "valid": v["valid"],
            "json_strict_ok": v["json_strict_ok"], "needed_extraction": v["needed_extraction"],
            "errors": v["errors"], "grounding_rate": v["grounding_rate"],
            "invented_ips": v["invented_ips"], "disruptive_commands": v["disruptive_commands"],
            "error": attempt["error"],
        }
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        return rec

    def write_manifest(self, manifest: dict) -> None:
        path = self.path.with_suffix(".manifest.json")
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sessions": []}
        old["sessions"].append(manifest)
        path.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def read(path: str) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def completed(path: str) -> set[tuple[str, str, int]]:
        """(condition, alert_id, repetition) triples that already have a final record, for resuming."""
        return {(r["condition"], r["alert_id"], r["repetition"]) for r in RunLogger.read(path) if r["final"]}
