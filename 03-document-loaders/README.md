# Document Loaders

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

`03-document-loaders/` has six standalone files, each demonstrating a different LangChain
document loader (step 1 of the RAG pipeline) — everything else about the pipeline (or, for
the smaller formats, what's left of it) stays the same. Step 6 (the Qorebit call) is
commented out in every file, same as `01-chunking-methods/`:

| File | Loader | Input | What it shows |
| --- | --- | --- | --- |
| `01_text_loader.py` | `TextLoader` | `data/sample.txt` | The baseline: one Document for a whole plain-text file |
| `02_pdf_loader.py` | `PyPDFLoader` | `data/sample.pdf` | One Document per PDF page, with a `page` number in metadata |
| `03_csv_loader.py` | `CSVLoader` | `data/sample.csv` | One Document per row, formatted as `column: value` lines |
| `04_json_loader.py` | `JSONLoader` | `data/sample.json` | Pulls one Document per array element out of nested JSON with a jq schema |
| `05_directory_loader.py` | `DirectoryLoader` | `data/sample.txt` + `data/sample2.txt` | Loads every file matching a glob pattern in one call, instead of naming files one by one |
| `06_web_loader.py` | `WebBaseLoader` | a live web page | The only loader here that needs internet access, parsed with BeautifulSoup |

For the row/record-shaped formats (CSV, JSON) chunking is skipped entirely — each row or
record is already a small, self-contained unit of text, so step 2 just passes the documents
through as-is.

## Running it

```powershell
python 03-document-loaders/01_text_loader.py
python 03-document-loaders/02_pdf_loader.py
python 03-document-loaders/03_csv_loader.py
python 03-document-loaders/04_json_loader.py
python 03-document-loaders/05_directory_loader.py
python 03-document-loaders/06_web_loader.py
```
or `cd 03-document-loaders` and run each file by its bare name, same as
`01-chunking-methods/`. Either way, the venv needs to be active and `data/`/`db/` are always
resolved relative to the project root.

## Notes / gotchas

- **`04_json_loader.py` needs the `jq` package.** `JSONLoader` uses jq schemas to pull data
  out of nested JSON, so it depends on the `jq` Python bindings (in `requirements.txt`)
  rather than just `langchain-community`.
- **`06_web_loader.py` needs internet access and `beautifulsoup4`.** It's the only loader
  demo that calls out to a live URL instead of reading a local file. It also sets a
  `USER_AGENT` environment variable to avoid a harmless warning some sites' servers trigger
  when it's unset.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
