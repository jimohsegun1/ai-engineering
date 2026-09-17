# Chunking Methods

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

`01-chunking-methods/` has six standalone files, each demonstrating a different chunking
(step 2 of the RAG pipeline) strategy — everything else about the pipeline stays the same. In
every file, step 6 (the Qorebit call) is **commented out on purpose**, so you can read/run
steps 1-5 for free and uncomment step 6 yourself when you're ready to spend Qorebit credits:

| File | Method | Input document | What it shows |
| --- | --- | --- | --- |
| `01_character_splitter.py` | `CharacterTextSplitter` | `data/sample.txt` | Fixed-size, single-separator splitting — an oversized paragraph is *not* split further |
| `02_recursive_character_splitter.py` | `RecursiveCharacterTextSplitter` | `data/sample.txt` | Falls back through smaller separators to still hit the target size (same method `rag_pipeline.py` uses) |
| `03_token_splitter.py` | `TokenTextSplitter` | `data/sample.txt` | Sizes chunks by token count instead of character count |
| `04_markdown_header_splitter.py` | `MarkdownHeaderTextSplitter` | `data/sample.md` | Splits along `#`/`##` headers, keeping the heading path as metadata |
| `05_semantic_chunker.py` | `SemanticChunker` (langchain-experimental) | `data/sample.txt` | Splits where meaning shifts between sentences, using embeddings rather than a fixed size |
| `06_code_splitter.py` | `RecursiveCharacterTextSplitter.from_language(PYTHON)` | `data/sample_code.py` | Splits source code along function/class boundaries instead of prose separators |

## Running it

Every file resolves its input/output paths from its own location, not the current working
directory, so you can run them either from the project root:
```powershell
python 01-chunking-methods/01_character_splitter.py
python 01-chunking-methods/02_recursive_character_splitter.py
python 01-chunking-methods/03_token_splitter.py
python 01-chunking-methods/04_markdown_header_splitter.py
python 01-chunking-methods/05_semantic_chunker.py
python 01-chunking-methods/06_code_splitter.py
```
or from inside `01-chunking-methods/` itself:
```powershell
cd 01-chunking-methods
python 01_character_splitter.py
```
Either way, the venv still needs to be active and `data/`/`db/` are always resolved
relative to the project root, never relative to `01-chunking-methods/`.

## Notes / gotchas

- **`05_semantic_chunker.py`'s default threshold is tuned for long documents.**
  `SemanticChunker`'s default `breakpoint_threshold_amount` (95th percentile) only treats
  the single most extreme meaning-shift as a split point, which collapses a short document
  like our ~10-sentence sample into one giant chunk. The file lowers it to 50 so it finds
  multiple breakpoints instead — on a longer document you'd likely want it closer to the
  default. It can also emit an empty trailing chunk, which the file filters out.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
