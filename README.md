# SciLens

SciLens is a literature analysis tool.  It asks: *what does the relevant evidence around this scientific question look like overall?*

Example question:

> Does creatine improve cognition?

SciLens then:

1. Retrieves papers from PubMed
2. Embeds the question and each title+abstract
3. Ranks papers by semantic similarity
4. **Drops off-topic papers** instead of forcing every PubMed hit into an answer
5. Classifies remaining papers as **Supports**, **Contradicts**, or **Mixed**
6. Aggregates those labels into a consensus, confidence score, and chart

This is a research prototype.

## Pipeline

```
question → PubMed API → parse abstracts
        → transformer embeddings → cosine rank → relevance filter
        → NLI (or optional OpenAI) classification
        → weighted consensus → Streamlit / CLI
```

## Quick start

Python 3.10+ recommended.

```bash
cd SciLens
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Set `NCBI_EMAIL` in `.env` to an address NCBI can contact (their E-utilities guideline).

First run downloads embedding and NLI models. That can take a few minutes.

### Web app

```bash
streamlit run app.py
```

### Command line

```bash
python -m scilens "Does creatine improve cognition?" --max-results 15
python -m scilens "Does creatine improve cognition?" --json
```

Local classification (`--backend nli`) needs no OpenAI key. To use GPT instead:

```bash
python -m scilens "Does creatine improve cognition?" --backend openai
```

## Project layout

```
app.py                 Streamlit UI
scilens/
  pubmed.py            NCBI E-utilities search + XML parse
  retriever.py         MiniLM embeddings + cosine ranking
  classifier.py        zero-shot NLI or optional OpenAI
  consensus.py         weighted vote, verdict, confidence
  pipeline.py          end-to-end analyze_question()
tests/                 unit tests (no live PubMed / model downloads)
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `NCBI_EMAIL` | Identifies you to PubMed |
| `NCBI_API_KEY` | Optional; higher request rate |
| `OPENAI_API_KEY` | Only if `--backend openai` |
| `SCILENS_NLI_MODEL` | Defaults to `facebook/bart-large-mnli` |

Never commit `.env`.

