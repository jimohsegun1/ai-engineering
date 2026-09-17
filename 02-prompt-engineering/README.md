# Prompt Engineering

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

`02-prompt-engineering/` has five standalone files, each demonstrating a different prompting
technique on the same local `google/flan-t5-base` model:

| File | Technique | What it shows |
| --- | --- | --- |
| `01_zero_shot_prompting.py` | Zero-shot | Just an instruction, no examples of the desired output |
| `02_few_shot_prompting.py` | Few-shot (`FewShotPromptTemplate`) | A handful of labeled examples before the real question — compare with the file above |
| `03_chain_of_thought_prompting.py` | Chain-of-thought | Asking the model to reason step by step, compared against a direct-answer prompt on the same question |
| `04_role_based_prompting.py` | Role-based | The same question answered twice, once per assigned persona |
| `05_structured_output_prompting.py` | Structured output (`PydanticOutputParser`) | Asks for JSON matching a schema, and handles the (likely) case where a small model doesn't follow it |

No API key needed — every file runs on the same free, local `flan-t5-base` model.

## Running it

```powershell
python 02-prompt-engineering/01_zero_shot_prompting.py
python 02-prompt-engineering/02_few_shot_prompting.py
python 02-prompt-engineering/03_chain_of_thought_prompting.py
python 02-prompt-engineering/04_role_based_prompting.py
python 02-prompt-engineering/05_structured_output_prompting.py
```
or `cd 02-prompt-engineering` and run each file by its bare name. Either way, the venv needs
to be active and `data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **`flan-t5-base` is an unreliable narrator for some of these.** It's the same small
  (~250M parameter) model used in `rag_pipeline_huggingface.py`, chosen for being free and
  CPU-friendly, not for quality. Chain-of-thought answers can loop or drift, and
  `05_structured_output_prompting.py`'s JSON parsing often fails outright. That's noted in
  each file and is expected; the point is to see the prompting mechanics work, not to get
  perfect answers out of a ~250M parameter model.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
