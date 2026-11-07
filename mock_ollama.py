"""TEST TOOL ONLY: a fake Ollama server for rehearsing the pipeline.

It imitates the parts of the Ollama API that the pipeline uses (/api/chat,
/api/version, /api/show) and answers with simple rule-of-thumb verdicts, some random
mistakes and occasional invalid output (to exercise the retry). The numbers it
produces are MEANINGLESS. Never report results from it.

Usage:
  terminal 1:  python mock_ollama.py --port 11435
  terminal 2:  python orchestrator.py run --host http://localhost:11435 --models fake-model --limit 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

ATTACK_WORDS = ("synthetic", "scan", "brute", "failed", "attack", "exploit", "malware", "denied", "injection")


def fake_answer(alert: dict, rng: random.Random) -> dict:
    desc = alert.get("rule_description", "")
    attack = any(w in desc.lower() for w in ATTACK_WORDS)
    if rng.random() < 0.15:  # random mistake
        attack = not attack
    return {
        "classification": "true_positive" if attack else "false_positive",
        "priority": {"level": "high" if attack else "low", "score": rng.randint(65, 92) if attack else rng.randint(3, 30),
                     "factors": [f"repeat_count {alert.get('repeat_count')}"]},
        "confidence": round(rng.uniform(0.55, 0.95), 2),
        "situation_explanation": f"Fake explanation for: {desc}",
        "evidence": [f"rule_description: {desc}", f"repeat_count: {alert.get('repeat_count')}"],
        "recommended_actions": ["This is fake model output."],
        "commands": [],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({"version": "mock-0.0"} if self.path.startswith("/api/version") else {})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/api/show"):
            return self._send({"details": {"parameter_size": "fake", "quantization_level": "none", "family": "mock"}})
        last = body["messages"][-1]["content"]
        m = re.search(r"<alert_data>\n(.*?)\n</alert_data>", last, flags=re.S)
        alert = json.loads(m.group(1)) if m else {}
        retry = "previous answer was invalid" in last
        seed = body.get("options", {}).get("seed", 0)
        rng = random.Random(int(hashlib.md5(f"{alert}{seed}".encode()).hexdigest(), 16))
        if not retry and rng.random() < 0.10:
            content = "I believe this alert is probably harmless, but I would check it."  # invalid on purpose
        else:
            content = json.dumps(fake_answer(alert, rng))
        n_in = sum(len(x["content"]) for x in body["messages"]) // 4
        self._send({"model": body.get("model"), "message": {"role": "assistant", "content": content}, "done": True,
                    "total_duration": 1_500_000_000, "load_duration": 100_000_000, "prompt_eval_count": n_in,
                    "eval_count": 90, "eval_duration": 900_000_000})

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11435)
    args = ap.parse_args()
    print(f"FAKE Ollama listening on http://localhost:{args.port} (results are meaningless). Ctrl-C to stop.")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
