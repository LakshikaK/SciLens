from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from scilens.pipeline import analyze_question

load_dotenv()

st.set_page_config(page_title="SciLens", page_icon="🔬", layout="wide")
st.title("SciLens")
st.caption("Ask a scientific question. See what PubMed evidence supports, contradicts, or mixes.")

with st.sidebar:
    st.header("Settings")
    max_results = st.slider("PubMed papers to retrieve", 5, 40, 15)
    min_similarity = st.slider("Relevance cutoff", 0.15, 0.7, 0.35, 0.01)
    backend = st.selectbox(
        "Classifier",
        ["nli", "openai"],
        help="nli = local transformer (no API key). openai = GPT, needs OPENAI_API_KEY.",
    )
    st.markdown(
        "SciLens **filters out** low-similarity papers instead of forcing every hit into the answer."
    )

question = st.text_input(
    "Scientific question or claim",
    placeholder="Does creatine improve cognition?",
)

run = st.button("Analyze evidence", type="primary", disabled=not question.strip())

if run:
    with st.spinner("Retrieving papers, ranking relevance, and classifying evidence..."):
        try:
            result = analyze_question(
                question,
                max_results=max_results,
                min_similarity=min_similarity,
                backend=backend,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    left, mid, right = st.columns(3)
    left.metric("Retrieved", result.n_retrieved)
    mid.metric("Relevant", result.n_relevant)
    right.metric("Confidence", f"{result.confidence:.2f}")
    st.subheader(result.verdict)
    st.write(result.explanation)

    counts_df = pd.DataFrame(
        {
            "stance": list(result.counts.keys()),
            "papers": list(result.counts.values()),
        }
    )
    if result.n_relevant:
        chart_col, table_col = st.columns([1, 2])
        with chart_col:
            fig = px.pie(
                counts_df,
                names="stance",
                values="papers",
                color="stance",
                color_discrete_map={
                    "Supports": "#2e7d32",
                    "Contradicts": "#c62828",
                    "Mixed": "#ef6c00",
                },
                title="Evidence mix",
            )
            st.plotly_chart(fig, use_container_width=True)
        with table_col:
            papers_df = pd.DataFrame(result.papers)
            display = papers_df.rename(
                columns={
                    "title": "Title",
                    "year": "Year",
                    "label": "Stance",
                    "similarity": "Relevance",
                    "confidence": "Class. score",
                    "url": "PubMed",
                }
            )
            show_cols = [c for c in ["Title", "Year", "Stance", "Relevance", "Class. score", "PubMed"] if c in display.columns]
            st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

        with st.expander("Abstracts and classification detail"):
            for paper in result.papers:
                st.markdown(
                    f"**{paper.get('label', '')}** · relevance {paper.get('similarity', 0):.2f} · "
                    f"[{paper.get('title', 'Untitled')}]({paper.get('url', '')})"
                )
                st.write(paper.get("abstract", "")[:1200])
                st.divider()
    else:
        st.info("Nothing cleared the relevance filter. Try a broader question or a lower cutoff.")

    if result.discarded:
        with st.expander("Filtered out as off-topic"):
            discarded_df = pd.DataFrame(result.discarded)
            cols = [c for c in ["title", "similarity", "year"] if c in discarded_df.columns]
            st.dataframe(discarded_df[cols], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Not medical advice. Consensus is a weighted summary of retrieved abstracts, not a systematic review."
)
