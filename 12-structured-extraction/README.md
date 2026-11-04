# Structured Extraction

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

`12-structured-extraction/` has five standalone files, each building on the previous one
to show a more complete picture of structured extraction with LangChain and Pydantic.
Structured extraction is the technique of asking an LLM to fill a typed schema from
free-form text, rather than returning a free-text answer — turning unstructured documents
into validated, machine-readable objects your code can use directly.

| File | What it shows |
| --- | --- |
| `01_basic_extraction.py` | `PydanticOutputParser` on a flat schema — the simplest possible extraction: one object, three fields |
| `02_nested_models.py` | A schema that contains another schema (a `Company` with an embedded `Address`) — introduces JSON nesting |
| `03_list_extraction.py` | A container model wrapping a `list[LineItem]` — one pass extracts multiple entities from the same text |
| `04_validation.py` | `field_validator` and `model_validator` — catches semantic errors (negative salary, inverted range) that JSON parsing alone misses |
| `05_rag_extraction.py` | RAG + extraction in one chain — retrieves the relevant passage first, then extracts a typed object from it |

No API key needed — every file runs on the same free, local `google/flan-t5-base` model used
across this project.

## Running it

Every file resolves its paths relative to its own location so you can run from the project
root:
```powershell
python 12-structured-extraction/01_basic_extraction.py
python 12-structured-extraction/02_nested_models.py
python 12-structured-extraction/03_list_extraction.py
python 12-structured-extraction/04_validation.py
python 12-structured-extraction/05_rag_extraction.py
```
or `cd 12-structured-extraction` and run each file by its bare name. Either way, the venv
needs to be active and `data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **`flan-t5-base` frequently mis-formats JSON.** Structured extraction depends on the model
  producing valid JSON that matches the schema exactly. Small models like `flan-t5-base`
  are unreliable at this — you will often see `Parsing failed` in the output. That failure
  is part of the lesson: every file handles it explicitly and explains why. A larger
  instruction-tuned model (Ollama `llama3.2:3b`, Qorebit) produces valid JSON nearly every
  time. To swap the model, change the `CHAT_MODEL` constant at the top of any file.
- **Pydantic validators run after parsing.** `04_validation.py` shows this explicitly: the
  JSON can parse successfully but still raise a `ValidationError` if a field constraint or
  cross-field rule fires. This is intentional — validators are the right place to enforce
  business rules like "salary_min must not exceed salary_max", not somewhere in your
  application code that might be bypassed.
- **`05_rag_extraction.py` wipes its Chroma folder on every run.** Same behaviour as every
  other RAG demo in this project — the folder at `db/extraction_rag/` is deleted and rebuilt
  each time so re-running never duplicates chunks.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store reset
behaviour, dependency version pins) that apply across the whole project.
