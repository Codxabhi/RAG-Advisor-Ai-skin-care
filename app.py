from __future__ import annotations

import base64
from typing import Any

import streamlit as st

from src.advisor import (
    DISCLAIMER,
    RoutinePlan,
    answer_question,
    decode_ingredients,
    generate_routine,
)
from src.config import get_settings
from src.knowledge import load_chunks
from src.llm import embed_texts, make_client
from src.retrieve import EmbeddingIndex, TfidfIndex

st.set_page_config(page_title="AI Skin Care", page_icon="💧", layout="wide")

CSS = """
<style>
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #f7e7e4 0%, #fbf7f2 45%); }
.hero { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em; color: #2c2420; }
.sub { color: #6b5b55; margin-bottom: 1.2rem; }
.card {
  background: #fffdfb; border: 1px solid #eadfd6; border-radius: 16px;
  padding: 1rem 1.1rem; margin-bottom: 0.7rem;
}
.cite { font-size: 0.8rem; color: #8a6f68; }
.warn {
  background: #fff4e8; border: 1px solid #f0d2a8; border-radius: 12px;
  padding: 0.75rem 1rem; color: #5c3d1e; font-size: 0.92rem;
}
</style>
"""


def _image_data_url(upload) -> str | None:
    if upload is None:
        return None
    raw = upload.getvalue()
    mime = upload.type or "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


@st.cache_resource
def _load_index():
    settings = get_settings()
    chunks = load_chunks(settings.knowledge_path)
    return settings, chunks, TfidfIndex(chunks)


def _retrieve(query: str, k: int = 4):
    settings, chunks, tfidf = _load_index()
    client = st.session_state.get("client")
    use_embed = bool(st.session_state.get("use_embeddings") and client)
    if use_embed:
        try:
            if "emb_index" not in st.session_state:
                vectors = embed_texts(client, settings, [c.document for c in chunks])
                st.session_state.emb_index = EmbeddingIndex(chunks, vectors)
            qv = embed_texts(client, settings, [query])[0]
            return st.session_state.emb_index.search(qv, k=k)
        except Exception:
            pass
    return tfidf.search(query, k=k)


def _render_steps(title: str, steps: list[Any]) -> None:
    st.markdown(f"**{title}**")
    if not steps:
        st.caption("No steps returned.")
        return
    for step in steps:
        ingredients = ", ".join(step.example_ingredients) if step.example_ingredients else "—"
        st.markdown(
            f'<div class="card"><b>{step.slot}</b> · {step.product_type}'
            f"<br>{step.why}<div class='cite'>Look for: {ingredients}</div></div>",
            unsafe_allow_html=True,
        )


def _render_plan(plan: RoutinePlan) -> None:
    st.markdown(plan.summary)
    left, right = st.columns(2)
    with left:
        _render_steps("Morning", plan.morning)
    with right:
        _render_steps("Night", plan.night)
    if plan.weekly:
        st.markdown("**Weekly**")
        for item in plan.weekly:
            st.markdown(f"- {item}")
    if plan.avoid:
        st.markdown("**Go easy on**")
        for item in plan.avoid:
            st.markdown(f"- {item}")
    if plan.citations:
        st.caption("Grounded in: " + ", ".join(f"`{c}`" for c in plan.citations))
    st.markdown(f'<div class="warn">{plan.disclaimer}</div>', unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    settings, chunks, _tfidf = _load_index()

    st.markdown('<div class="hero">AI Skin Care</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Intermediate RAG advisor — retrieve ingredient notes, then generate a grounded routine.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="warn">{DISCLAIMER}</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Profile")
        skin_type = st.selectbox(
            "Skin type",
            ["combination", "oily", "dry", "normal", "sensitive", "acne-prone"],
        )
        concerns = st.multiselect(
            "Concerns",
            [
                "acne",
                "clogged pores",
                "dark marks",
                "redness",
                "dryness",
                "fine lines",
                "dullness",
            ],
            default=["acne"],
        )
        climate = st.selectbox("Climate", ["humid", "dry", "temperate", "hot", "cold"])
        budget = st.select_slider("Routine complexity", options=["minimal", "simple", "full"])
        allergies = st.text_input("Avoid / allergies", placeholder="fragrance, essential oils…")
        extra = st.text_area("Anything else", placeholder="Retinol stings, I wear makeup, etc.")

        st.divider()
        st.header("Model")
        api_key = st.text_input("API key", value=settings.api_key, type="password")
        base_url = st.text_input("Base URL", value=settings.base_url)
        chat_model = st.text_input("Chat model", value=settings.chat_model)
        use_embeddings = st.toggle("Use embedding retrieval", value=False)
        st.caption(f"{len(chunks)} knowledge chunks loaded.")

    profile = {
        "skin_type": skin_type,
        "concerns": ", ".join(concerns) or "general care",
        "climate": climate,
        "complexity": budget,
        "avoid": allergies or "none stated",
        "notes": extra or "none",
    }

    live = settings
    if api_key:
        live = settings.__class__(
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
            embedding_model=settings.embedding_model,
            knowledge_path=settings.knowledge_path,
            eval_path=settings.eval_path,
        )

    st.session_state.use_embeddings = use_embeddings
    st.session_state.client = make_client(live) if live.has_key else None

    tabs = st.tabs(["Routine", "Ingredient decoder", "Ask + photo", "Retrieved notes"])

    with tabs[0]:
        st.subheader("Build an AM / PM plan")
        if st.button("Generate routine", type="primary"):
            if not live.has_key:
                st.error("Add an OpenAI-compatible API key in the sidebar or `.env`.")
            else:
                query = f"{skin_type} skin; concerns: {profile['concerns']}; climate {climate}"
                hits = _retrieve(query, k=5)
                with st.spinner("Writing a conservative routine…"):
                    plan = generate_routine(st.session_state.client, live, profile, hits)
                st.session_state.plan = plan
                st.session_state.last_hits = hits
        if "plan" in st.session_state:
            _render_plan(st.session_state.plan)

    with tabs[1]:
        st.subheader("Paste an INCI list")
        inci = st.text_area("Ingredients", height=160, placeholder="Aqua, Niacinamide, Glycerin…")
        if st.button("Decode ingredients"):
            if not live.has_key:
                st.error("Add an API key first.")
            elif not inci.strip():
                st.warning("Paste a product ingredient list.")
            else:
                hits = _retrieve(inci, k=5)
                with st.spinner("Reading the label…"):
                    text = decode_ingredients(
                        st.session_state.client, live, inci, profile, hits
                    )
                st.markdown(text)
                st.session_state.last_hits = hits

    with tabs[2]:
        st.subheader("Ask a question")
        photo = st.file_uploader(
            "Optional face photo (vision models only)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        question = st.text_input(
            "Question",
            placeholder="Can I use azelaic acid with retinol?",
        )
        if st.button("Ask"):
            if not live.has_key:
                st.error("Add an API key first.")
            elif not question.strip():
                st.warning("Type a question.")
            else:
                hits = _retrieve(question, k=5)
                image_url = _image_data_url(photo)
                with st.spinner("Thinking with retrieved notes…"):
                    answer = answer_question(
                        st.session_state.client,
                        live,
                        question,
                        profile,
                        hits,
                        image_data_url=image_url,
                    )
                st.markdown(answer)
                st.session_state.last_hits = hits

    with tabs[3]:
        st.subheader("What the retriever used last")
        hits = st.session_state.get("last_hits") or []
        if not hits:
            st.caption("Run a routine, decode, or ask to see retrieved chunks.")
        for chunk, score in hits:
            with st.expander(f"{chunk.chunk_id}  ·  {score:.3f}"):
                st.write(chunk.text)


if __name__ == "__main__":
    main()
