# AI Skin Care

Intermediate LLM project: a **personalized skincare advisor** with RAG (retrieval-augmented generation).

You describe your skin, the app retrieves relevant ingredient and routine notes from a local knowledge base, then an LLM writes an AM/PM plan with citations. You can also decode an ingredient list and ask follow-up questions.

This is project is under development so advice with doctor is beneficiary

## Setup

```bash
cd ai-skin-care
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put an API key in `.env`. Any OpenAI-compatible server works:

- OpenAI: leave `OPENAI_BASE_URL` as `https://api.openai.com/v1`
- Ollama: `OPENAI_BASE_URL=http://localhost:11434/v1` and `OPENAI_API_KEY=ollama`
- LM Studio: `OPENAI_BASE_URL=http://localhost:1234/v1`

## Run

```bash
streamlit run app.py
```

## Eval (retrieval quality)

```bash
python eval_retrieval.py
```

This scores whether the retriever returns the expected knowledge chunks for a small set of skincare questions.
