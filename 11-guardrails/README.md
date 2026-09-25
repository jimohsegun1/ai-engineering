# Guardrails

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

[`02-prompt-engineering/05_structured_output_prompting.py`](../02-prompt-engineering/README.md)
shows the problem: a model often doesn't follow a requested JSON schema, and that demo just
catches the parsing error and stops there. A guardrail does one more thing: when validation
fails, it feeds the parser's error back to the LLM and asks it to fix its own output, instead
of surfacing a broken result to whatever code called it. `OutputFixingParser` wraps a normal
`PydanticOutputParser` with exactly this retry loop.

Uses Ollama (`llama3.2:3b`) rather than `flan-t5-base` — flan-t5-base barely attempted the
schema at all here (empty or one-word raw output), leaving the retry nothing recoverable to
work with.

`01_output_validation_retry.py` runs the same retry-guarded parser against **two prompts**,
on purpose — see Notes / gotchas below for why the comparison is the actual lesson here, not
just the final working version.

## Running it

```powershell
python 11-guardrails/01_output_validation_retry.py
```
or `cd 11-guardrails` and run it by its bare name. Needs the same Ollama setup as
[`06-agents/`](../06-agents/README.md) — no API key.

## Notes / gotchas

- **A retry guardrail is a safety net for near-miss formatting errors, not a fix for a model
  that has misunderstood the task.** Testing this turned up something worth keeping rather
  than prompting away. With only `parser.get_format_instructions()` (the bare JSON schema, no
  filled example) in the prompt, `llama3.2:3b` consistently echoed the *schema itself* back —
  the literal field descriptions and types — instead of a filled instance, on every single
  review (naive: 0/5). `OutputFixingParser`'s automatic retry did not recover a single one of
  those failures (retry: still 0/5), because its repair prompt shows the model that same
  confusing schema again, so it fails the same way. Adding one concrete filled-in example to
  the prompt fixed the problem outright (naive: 5/5, no retry needed at all). The file runs
  both prompts side by side so you see the failure and the fix, not just the working version.
- **Good prompting prevents more failures than automated repair recovers.** The guardrail
  still has real value — it catches genuine near-misses (an extra trailing comma, a stray
  code fence) that a slightly-off but fundamentally-correct attempt might produce — but it
  isn't a substitute for giving the model a concrete example of the output you want.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
