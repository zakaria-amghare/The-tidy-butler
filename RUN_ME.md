# Running the pipeline (read this when your PC is on)

All files go in ONE folder, together with your alert files (for example `russellmitchell_alerts.txt`).

| File | What it is |
|---|---|
| `alerts_loader.py` | Module 1: loads alerts, collapses repeats, samples a pilot set |
| `baseline_engine.py` | Module 2: baseline (risk score, known false positives) |
| `llm_engine.py` | Module 3: talks to the local model (Ollama), 4 prompt formats |
| `output_validator.py` | Module 4: checks the model's JSON, one retry, safety screens |
| `run_logger.py` | Module 5: logs every attempt, supports resuming |
| `orchestrator.py` | Module 6: prepare / run / status |
| `evaluate.py` | Module 7: metrics, statistics, report |
| `mock_ollama.py` | TEST TOOL: a fake model server. Its numbers are meaningless. |

## 0. One-time setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy
```
(scipy is optional; it only adds a latency test.) For real models install Ollama and keep `ollama serve` running.

## 1. Check Module 1 on your real file (5 minutes)

```
python alerts_loader.py russellmitchell_alerts.txt --n-attack 10 --n-other 10
```
Read the printed `time_label` and `event_label` counts. The code assumes that any `time_label` other than
`false_positive` means attack. If the real values say otherwise, tell me before going further.

## 2. Rehearse the whole pipeline with the FAKE model (10 minutes, no Ollama needed)

Terminal 1:
```
python mock_ollama.py --port 11435
```
Terminal 2:
```
python orchestrator.py prepare russellmitchell_alerts.txt --n 60 --dir rehearsal
python orchestrator.py run --dir rehearsal --host http://localhost:11435 --models fake-model --prompts zero_shot,few_shot --limit 20
python orchestrator.py status --dir rehearsal --host http://localhost:11435 --models fake-model --prompts zero_shot,few_shot --limit 20
python evaluate.py --dir rehearsal
```
If this works, the plumbing works. Do NOT report any number from it.

## 3. Try the real model on 3 alerts

```
ollama pull llama3.2:3b
python llm_engine.py --d2 pilot_d2.jsonl --prompt few_shot --dry-run
python llm_engine.py --d2 pilot_d2.jsonl --model llama3.2:3b --prompt zero_shot --limit 3
python output_validator.py --live --d2 pilot_d2.jsonl --prompt zero_shot --model llama3.2:3b --limit 3
```
(`pilot_d2.jsonl` comes from step 1. Check `ollama list` for models you can use. On a CPU-only laptop start with a 3B model.)
Note the seconds per alert. That number decides how big your experiment can be.

## 4. First real experiment (small)

```
python orchestrator.py prepare russellmitchell_alerts.txt --n 200 --dir experiment
python orchestrator.py run --dir experiment --models llama3.2:3b --prompts zero_shot --limit 20
python evaluate.py --dir experiment
```
- You can stop with Ctrl-C or switch the PC off. Run the SAME command again and it resumes where it stopped.
- `prepare` prints warnings if the test set is not balanced. Read them.
- Evaluation only uses alerts that EVERY condition has finished, and it says how many.

## 5. Grow it step by step

Add prompt formats (`--prompts zero_shot,zero_shot_guidelines,one_shot,few_shot`), then a second model
(`--models a,b`), then repetitions with temperature above 0 (`--temperatures 0.7 --reps 3`).
At temperature 0, repetitions are identical, so the code runs them once.
Remove `--limit` only when you know the time per alert.

## Known gaps (on purpose, for now)

- Your CSV has no `rule.level`, so the severity-only baseline is not available yet.
- The few-shot examples are made-up placeholders. Replace them with real development-set examples. Never use test alerts.
- The class and priority definitions in `zero_shot_guidelines` must match what you use when labelling.
- Context enrichment (modes B and C), the dashboard and the notifier are Modules 8-10, not built yet.
- Explanation quality still needs the human rubric. The automatic screens are only heuristics.
- Schema mode forces valid JSON. Also run some tests with `--format json` so the validity rate is really measured.

## If something fails

Send me: which step, the full command, and the first error line. Please do not edit `runs.jsonl` by hand.
