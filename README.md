# Bulk Video Transcription Generator (RAG Pipeline)

A retrieval-augmented generation pipeline that builds a personal knowledge base from cleaned video transcripts, TikTok Q&A/DM pairs, and LinkedIn posts, then answers questions in the author's own voice — with an evaluation harness that scores answers against a baseline.

- **Live URL:** none published (local CLI)
- **Repository:** https://github.com/fahadguldev/bulk-video-transcription-generator

> **Note on the name.** Despite the repository name, the code here is a RAG pipeline, not a transcription tool. `transcribe.py` has its `faster-whisper` import commented out, and `main.py` is still the default `print("Hello from transcription!")` stub. The corpus it queries was produced offline from 119 already-cleaned transcripts. This README describes what the code actually does.

## Project Overview

| File | Role |
| --- | --- |
| `build_knowledge_base.py` (11.6 KB) | Assembles `knowledge_base/records.jsonl` from all Second Brain sources |
| `prepare_rag.py` | `records.jsonl` → `rag_records.jsonl` (one line per retrievable record) |
| `ask_brain.py` (10.4 KB) | Loads `system_prompt.md` persona, retrieves, queries the Gemini SDK |
| `build_eval_dataset.py` | Builds `eval_retrieval.jsonl` from **verbatim** real questions/comments/DMs |
| `evaluate_brain.py` | Scores `ask_brain` answers against the dataset |
| `evaluate_gemini.py` (24 KB) | Scores a baseline Gemini run for comparison |
| `extract-qa-pairs.py` | Extracts Q&A pairs (keyed on `CREATOR_UNIQUE_ID = "fahadgul.dev"`) |
| `transcribe.py` | `faster-whisper` transcription entry — currently commented out |
| `system_prompt.md` (4.7 KB) | The persona: "You are Fahad's Second Brain" |
| `test_embedding.py`, `test_env.py`, `test_gemini_rest.py` | Smoke checks for env, embedding, and REST access |
| `main.py` | Stub |

Data directories: `cleaned/` (119 usable video transcripts), `knowledge_base/` (built artifacts).

## Problem & Solution

A creator's knowledge is spread across 119 video transcripts, TikTok comments and DMs, and LinkedIn posts — content that is effectively unsearchable and unqueryable.

The pipeline collapses it into a single `records.jsonl`, retrieves relevant chunks per question, and answers through Gemini with a system prompt that constrains the model to the author's actual positions and communication style. The prompt's core rule is explicit: *never invent experiences, opinions, credentials, or personal history — if the retrieved context is insufficient, be honest instead.*

An evaluation harness exists because persona-prompted RAG is easy to make confidently wrong: `build_eval_dataset.py` uses **verbatim** real questions (never generated or paraphrased) so retrieval quality is measured against genuine inputs.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python ≥ 3.13 (`uv`, `uv.lock`) |
| LLM / embeddings | Google Gemini SDK (`google-genai` ≥ 2.23.0) |
| Transcription (declared) | `faster-whisper` ≥ 1.2.1 — import present but commented out |
| Config | `python-dotenv` (`GEMINI_API_KEY`) |
| Numerics | `numpy` — used by `ask_brain.py`, `build_eval_dataset.py`, `evaluate_brain.py`, `evaluate_gemini.py` (**not declared in `pyproject.toml`**) |
| Data format | JSONL (`records.jsonl`, `rag_records.jsonl`, `eval_retrieval.jsonl`) |
| Persona | `system_prompt.md` + referenced `communication_style.md` |

## System Architecture

```
   SOURCES (read-only, originals untouched)
     cleaned/transcripts/hinglish/   119 video transcripts
     TikTok Q&A + DM pairs            via extract-qa-pairs.py
     LinkedIn posts
            |
            v
   build_knowledge_base.py
            |
            v
   knowledge_base/records.jsonl          (canonical corpus)
            |
      +-----+----------------------+
      |                            |
      v                            v
   prepare_rag.py              build_eval_dataset.py
      |                            |
      v                            v
   rag_records.jsonl            eval_retrieval.jsonl
   (one line per chunk)         (verbatim real questions)
      |                            |
      v                            |
   ask_brain.py  <--- system_prompt.md (persona rules)
      |                            |
      |   retrieve -> context      |
      |   Gemini generate          |
      v                            v
   answer                  evaluate_brain.py
                        vs  evaluate_gemini.py (baseline)
                                |
                                v
                          retrieval/answer metrics
```

`main.py` sits outside this flow entirely — it is not wired to any of it.

## Key Features

- **Multi-source corpus assembly** — `build_knowledge_base.py` merges three content types into one deduplicated `records.jsonl`, leaving originals untouched.
- **RAG preparation** — `prepare_rag.py` reshapes records into retrievable per-chunk lines.
- **Persona-constrained generation** — `ask_brain.py` loads `system_prompt.md`, which enforces authenticity rules (never invent experiences; prioritise retrieved context; stay short, punchy, direct) and references a separate `communication_style.md`.
- **Retrieval evaluation on verbatim queries** — `build_eval_dataset.py` builds the eval set from real questions/comments/DMs, explicitly never generated or paraphrased, paired with the retrieved context actually used.
- **Two-sided evaluation** — `evaluate_brain.py` scores the RAG system; `evaluate_gemini.py` (the largest script at 24 KB) scores a plain-Gemini baseline for comparison.
- **Q&A extraction** — `extract-qa-pairs.py` pulls conversation pairs from source data, keyed on a creator unique ID.
- **Embedding smoke tests** — `test_embedding.py` verifies the embedding call end-to-end; `test_env.py` verifies `GEMINI_API_KEY` loads; `test_gemini_rest.py` exercises the raw REST endpoint.
- **Transcription hook** — `transcribe.py` shows the intended `faster-whisper` flow (`WhisperModel("small", ...)`) even though it is commented out.

## Setup & Run

Two steps, in order:

```bash
git clone https://github.com/fahadguldev/bulk-video-transcription-generator
cd bulk-video-transcription-generator
uv sync                      # or: pip install -e .

python build_knowledge_base.py    # writes knowledge_base/records.jsonl
python ask_brain.py               # asks the question from the console
```

Before running, put `GEMINI_API_KEY` in a `.env` file.

Prepare and evaluate:

```bash
python prepare_rag.py             # -> rag_records.jsonl
python build_eval_dataset.py      # -> eval_retrieval.jsonl
python evaluate_brain.py          # score the RAG system
python evaluate_gemini.py         # score the baseline
```

Smoke checks:

```bash
python test_env.py                # GEMINI_API_KEY loads
python test_embedding.py          # embedding call works
python test_gemini_rest.py        # raw REST access works
```

`python main.py` prints `Hello from transcription!` and does nothing else.

## Technical Decisions

- **JSONL over a vector database.** For a corpus of this size, a single flat file that can be grepped, diffed, and regenerated beats standing up Qdrant/Chroma — and it keeps the whole pipeline reviewable.
- **Evaluation on verbatim queries.** Generated questions flatter retrieval systems; real questions from real comments and DMs do not. The dataset builder explicitly forbids paraphrasing for this reason.
- **Persona as a checked-in prompt file** rather than a string literal. `system_prompt.md` can be reviewed and edited independently of code, and its rules (authenticity, brevity, no invented history) are visible in review.
- **Baseline comparison built in.** `evaluate_gemini.py` gives a plain-model reference point — without it, "the RAG system answered well" is unfalsifiable.
- **Read-only source handling.** `build_knowledge_base.py` documents that originals under `cleaned/` are untouched, so re-runs are safe.
- **`uv` with a lockfile.** Reproducible installs including transitive versions, matching how the sibling Python projects are managed.
- **Smoke tests instead of a stub test suite.** `test_env` / `test_embedding` / `test_gemini_rest` verify the three external touchpoints fail fast and legibly.

## Challenges & Solutions

- **Heterogeneous sources.** Transcripts, DMs, and posts have different shapes; each gets a reader, and all converge on one record schema so retrieval never needs per-source logic.
- **Hallucinated persona.** The system prompt's core rule is a negative constraint — never invent experiences/opinions/credentials, and prefer honesty over an answer. That turns "would say" into "can be supported by retrieved context."
- **Evaluating style, not just facts.** Answers are scored against both retrieval correctness and persona adherence, which is why `evaluate_brain.py` is separate from a pure retrieval metric.
- **Retrieval regressions going unnoticed.** The eval dataset is regenerated from fixed verbatim queries, so changes to chunking or embedding can be compared run-over-run.
- **Broken external calls during setup.** Three small test scripts isolate env, embedding, and REST failures so the first real run does not fail with an opaque 401.

## Honest Gaps

- **The repository name does not match the code.** It presents as a transcription tool; `transcribe.py` is commented out and `main.py` is a stub. Anyone discovering this repo by name will be misled until they read the README.
- **`numpy` is imported but not declared.** `ask_brain.py`, `build_eval_dataset.py`, `evaluate_brain.py`, and `evaluate_gemini.py` all `import numpy`, yet `pyproject.toml` lists only `faster-whisper`, `google-genai`, and `python-dotenv`. A clean `uv sync` install will fail on first run of any of them.
- **`communication_style.md` is referenced by `system_prompt.md` but is not present** in the repository root — the persona file points at a dependency that does not ship.
- **`main.py` is a stub.** There is no unified entry point; each stage is a separate script with its own conventions.
- **No CLI argument handling** in the stage scripts — question input is read from stdin by `ask_brain.py`, so nothing can be scripted end-to-end.
- **No tests beyond three smoke scripts.** Nothing asserts retrieval quality automatically; the evaluation harness must be run and interpreted by hand.
- **Transcription never runs.** Despite `faster-whisper` being a declared dependency, the code path that would use it is commented out — the dependency is installed but unused.
- **No `.env.example`**, though `test_env.py` documents the expected `GEMINI_API_KEY`.
- **Single squashed commit**, so no development timeline can be inferred.

## Deployment Status

| Surface | URL | Status |
| --- | --- | --- |
| CLI | `python ask_brain.py` | local only |
| Production | — | not deployed |

## Lessons Learned

- Name the repository for what the code does. A mismatch between repo name and implementation costs every future reader a detour — and costs the author credibility.
- Evaluate on verbatim real queries. Generated questions systematically overstate retrieval quality because they inherit the phrasing your chunking already favours.
- A persona prompt's most important line is the negative one: *do not invent*. Authenticity failures are silent, so the constraint has to be explicit.
- Always ship the baseline. `evaluate_gemini.py` is what makes `evaluate_brain.py` mean something.
- Declared dependencies and imported dependencies must be reconciled — a clean-install failure on the first script is a poor first impression for a pipeline whose whole value is reproducibility.

## Author

**Muhammad Fahad** - [@fahadguldev](https://github.com/fahadguldev)

Repository: https://github.com/fahadguldev/bulk-video-transcription-generator
