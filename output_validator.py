"""Module 4: output validator (with the single retry).

Checks every LLM answer (D6) against the schema and against safety rules, and
counts failures honestly. Nothing the model writes is trusted until it passes.

What it checks
  - JSON: strict parse, or lenient extraction (code fences, text around the object)
  - schema: required fields, types, allowed values, score 0-100, confidence 0-1
  - evidence grounding (heuristic): does each evidence item mention a real alert field or value?
  - invented IP addresses: IPs in the answer that are not in the alert
  - disruptive commands (heuristic): flagged so the dashboard can warn the analyst

Retry policy: an invalid answer is retried ONCE with a different seed and a short
corrective note. If it is still invalid it is flagged as a failure and counted.

Usage
  offline:  python output_validator.py --runs llm_zero_shot.jsonl --d2 pilot_d2.jsonl
  live:     python output_validator.py --live --d2 pilot_d2.jsonl --prompt zero_shot --model llama3.2:3b --limit 3
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace

from llm_engine import LLM_FIELDS, PROMPT_FORMATS, LLMConfig, assess

REQUIRED = ["classification", "priority", "confidence", "situation_explanation",
            "evidence", "recommended_actions", "commands"]
CLASSES = {"true_positive", "false_positive", "benign"}
LEVELS = {"critical", "high", "medium", "low"}
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DISRUPTIVE = [  # heuristic list, not a security guarantee
    r"\brm\s+-[a-z]*r", r"\bmkfs", r"\bdd\s+if=", r"\b(shutdown|reboot|halt|poweroff)\b",
    r"(curl|wget)[^|]*\|\s*(ba)?sh", r":\(\)\s*\{", r"chmod\s+-R\s+777", r">\s*/dev/sd",
    r"\buserdel\b", r"iptables\s+-F", r"systemctl\s+(stop|disable|mask)",
]


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def extract_json(raw: str) -> tuple[object | None, bool, bool]:
    """Return (parsed, strict_ok, needed_extraction)."""
    text = raw.strip()
    try:
        return json.loads(text), True, False
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    for candidate in (fenced, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            return json.loads(candidate), False, True
        except json.JSONDecodeError:
            continue
    return None, False, False


def _str_list(obj: dict, key: str, errors: list[str], allow_empty: bool = True) -> list[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not all(isinstance(i, str) for i in value):
        errors.append(f"{key} must be a list of strings")
        return []
    if not allow_empty and not value:
        errors.append(f"{key} must not be empty")
    return value


def check_schema(obj: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED:
        if key not in obj:
            errors.append(f"missing field: {key}")
    extra = sorted(set(obj) - set(REQUIRED))
    if extra:
        warnings.append(f"extra fields ignored: {extra}")
    if "classification" in obj and obj["classification"] not in CLASSES:
        errors.append(f"classification must be one of {sorted(CLASSES)}")
    pr = obj.get("priority")
    if "priority" in obj:
        if not isinstance(pr, dict):
            errors.append("priority must be an object")
        else:
            if pr.get("level") not in LEVELS:
                errors.append(f"priority.level must be one of {sorted(LEVELS)}")
            score = pr.get("score")
            if not is_number(score):
                errors.append("priority.score must be a number")
            elif isinstance(score, float) and not score.is_integer():
                errors.append("priority.score must be an integer")
            elif not 0 <= score <= 100:
                errors.append("priority.score must be between 0 and 100")
            elif isinstance(score, float):
                warnings.append("priority.score was a float with an integer value")
            _str_list(pr, "factors", errors)
    if "confidence" in obj:
        if not is_number(obj["confidence"]):
            errors.append("confidence must be a number")
        elif not 0 <= obj["confidence"] <= 1:
            errors.append("confidence must be between 0 and 1")
    if "situation_explanation" in obj and not (isinstance(obj["situation_explanation"], str)
                                               and obj["situation_explanation"].strip()):
        errors.append("situation_explanation must be a non-empty string")
    for key in ("evidence", "recommended_actions", "commands"):
        if key in obj:
            _str_list(obj, key, errors)
    return errors, warnings


def _alert_values(alert: dict) -> list[str]:
    return [str(alert[k]).lower() for k in LLM_FIELDS if alert.get(k) is not None and len(str(alert[k])) >= 2]


def grounding_rate(obj: dict, alert: dict) -> tuple[float | None, list[str]]:
    """Share of evidence items that mention a real alert field name or value (heuristic screen)."""
    evidence = [e for e in obj.get("evidence", []) if isinstance(e, str)]
    if not evidence:
        return None, []
    values = _alert_values(alert)
    ungrounded = [e for e in evidence
                  if not (any(v in e.lower() for v in values) or any(f in e.lower() for f in LLM_FIELDS))]
    return round(1 - len(ungrounded) / len(evidence), 3), ungrounded


def invented_ips(obj: dict, alert: dict) -> list[str]:
    parts = [obj.get("situation_explanation", "")]
    for key in ("evidence", "recommended_actions", "commands"):
        parts += [x for x in obj.get(key, []) if isinstance(x, str)]
    pr = obj.get("priority")
    if isinstance(pr, dict):
        parts += [x for x in pr.get("factors", []) if isinstance(x, str)]
    known = set(_IP.findall(" ".join(str(alert.get(k, "")) for k in LLM_FIELDS)))
    return sorted(set(_IP.findall(" ".join(str(p) for p in parts))) - known)


def disruptive_commands(obj: dict) -> list[str]:
    cmds = [c for c in obj.get("commands", []) if isinstance(c, str)]
    return [c for c in cmds if any(re.search(p, c, flags=re.I) for p in DISRUPTIVE)]


def validate_output(raw: str | None, alert: dict | None = None) -> dict:
    """Validate one raw model answer. `valid` is True only if the schema check passes."""
    result = {"valid": False, "json_strict_ok": False, "needed_extraction": False, "errors": [],
              "warnings": [], "parsed": None, "grounding_rate": None, "ungrounded_evidence": [],
              "invented_ips": [], "disruptive_commands": []}
    if not raw or not raw.strip():
        result["errors"].append("empty output")
        return result
    parsed, strict, extracted = extract_json(raw)
    result["json_strict_ok"], result["needed_extraction"] = strict, extracted
    if parsed is None:
        result["errors"].append("not valid JSON")
        return result
    if not isinstance(parsed, dict):
        result["errors"].append("top level must be a JSON object")
        return result
    result["errors"], result["warnings"] = check_schema(parsed)
    result["valid"] = not result["errors"]
    if result["valid"]:
        result["parsed"] = parsed
        result["disruptive_commands"] = disruptive_commands(parsed)
        if alert:
            result["grounding_rate"], result["ungrounded_evidence"] = grounding_rate(parsed, alert)
            result["invented_ips"] = invented_ips(parsed, alert)
    return result


def assess_validated(alert: dict, prompt_format: str, cfg: LLMConfig = LLMConfig(),
                     repetition: int = 1, max_retries: int = 1) -> dict:
    """Run the model, validate, and retry at most max_retries times (new seed + corrective note)."""
    attempts: list[dict] = []
    note = None
    for n in range(max_retries + 1):
        rec = assess(alert, prompt_format, replace(cfg, seed=cfg.seed + n), repetition, retry_note=note)
        rec["attempt"] = n + 1
        rec["validation"] = validate_output(rec["raw_output"], alert)
        attempts.append(rec)
        if rec["validation"]["valid"] or rec["error"]:
            break
        note = ("Your previous answer was invalid: " + "; ".join(rec["validation"]["errors"][:5])
                + ". Answer again with exactly one JSON object containing every required field.")
    final = attempts[-1]["validation"]
    return {
        "alert_id": alert.get("alert_id"), "prompt_format": prompt_format, "repetition": repetition,
        "attempts": attempts, "retried": len(attempts) > 1,
        "first_attempt_valid": attempts[0]["validation"]["valid"],
        "valid": final["valid"], "failed": not final["valid"],
        "parsed": final["parsed"],
        "total_latency_s": round(sum(a["latency_s"] or 0 for a in attempts), 3),
        "connection_error": attempts[-1]["error"],
    }


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise(rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print("no rows")
        return
    v = [r["validation"] if "validation" in r else r for r in rows]
    valid = sum(x["valid"] for x in v)
    strict = sum(x["json_strict_ok"] for x in v)
    print(f"answers checked: {n}")
    print(f"strict JSON: {strict}/{n} | schema-valid (after lenient extraction): {valid}/{n}")
    print(f"needed extraction: {sum(x['needed_extraction'] for x in v)} | "
          f"answers with invented IPs: {sum(bool(x['invented_ips']) for x in v)} | "
          f"answers with disruptive commands: {sum(bool(x['disruptive_commands']) for x in v)}")
    rates = [x["grounding_rate"] for x in v if x["grounding_rate"] is not None]
    if rates:
        print(f"mean evidence grounding (heuristic): {sum(rates) / len(rates):.2f}")
    errs = Counter(e for x in v for e in x["errors"])
    if errs:
        print("most common errors:")
        for e, c in errs.most_common(5):
            print(f"  {c}x {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2", required=True, help="pilot_d2.jsonl")
    ap.add_argument("--runs", help="llm_*.jsonl from llm_engine.py (offline validation, no retry)")
    ap.add_argument("--live", action="store_true", help="call the model with validation and one retry")
    ap.add_argument("--prompt", choices=PROMPT_FORMATS, default="zero_shot")
    ap.add_argument("--model", default=LLMConfig.model)
    ap.add_argument("--host", default=LLMConfig.host)
    ap.add_argument("--format", choices=["schema", "json", "none"], default="schema", dest="output_format")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="output JSONL")
    args = ap.parse_args()
    alerts = {a["alert_id"]: a for a in read_jsonl(args.d2)}

    if args.runs:
        rows = read_jsonl(args.runs)[: args.limit]
        out = args.out or "validated_" + args.runs.split("/")[-1]
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                r["validation"] = validate_output(r.get("raw_output"), alerts.get(r.get("alert_id")))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summarise(rows)
        print(f"wrote {out}")
    elif args.live:
        cfg = LLMConfig(model=args.model, host=args.host, output_format=args.output_format)
        out = args.out or f"validated_live_{args.prompt}.jsonl"
        results = []
        with open(out, "w", encoding="utf-8") as f:
            for i, alert in enumerate(list(alerts.values())[: args.limit], 1):
                res = assess_validated(alert, args.prompt, cfg)
                results.append(res)
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                state = "valid" if res["valid"] else "FAILED"
                print(f"[{i}] {alert['alert_id']} {state} retried={res['retried']} {res['total_latency_s']}s")
                if res["connection_error"]:
                    print(res["connection_error"])
                    break
        n = len(results)
        if n:
            print(f"\nfirst-attempt valid: {sum(r['first_attempt_valid'] for r in results)}/{n} | "
                  f"valid after retry: {sum(r['valid'] for r in results)}/{n} | failures: {sum(r['failed'] for r in results)}/{n}")
        summarise([r["attempts"][-1] for r in results])
        print(f"wrote {out}")
    else:
        ap.error("give --runs FILE (offline) or --live")


if __name__ == "__main__":
    main()
