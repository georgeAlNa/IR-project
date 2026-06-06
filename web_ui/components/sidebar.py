from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from web_ui.services.api_client import get_dataset_catalog


@dataclass(frozen=True)
class SidebarState:
    base_url: str
    dataset_name: str
    max_documents: int
    representation_model: str
    k1: float
    b: float
    top_k: int
    candidate_k: int
    vector_size: int


def render_sidebar() -> SidebarState:
    catalog = get_dataset_catalog()

    st.sidebar.header("API Gateway Controls")
    base_url = st.sidebar.text_input("FastAPI Gateway URL", value="http://127.0.0.1:8000")
    dataset_name = st.sidebar.selectbox("Dataset", options=list(catalog.keys()))
    selected_dataset = catalog[dataset_name]
    max_documents = selected_dataset.default_document_limit
    if selected_dataset.ir_dataset_name:
        max_documents = st.sidebar.number_input(
            "Dataset document limit",
            min_value=1,
            max_value=10_000_000,
            value=selected_dataset.default_document_limit,
            step=10_000,
        )
    representation_model = st.sidebar.selectbox(
        "Representation Model",
        options=["TF-IDF", "BM25", "Embedding", "Hybrid Serial", "Hybrid Parallel"],
    )
    k1 = st.sidebar.slider("BM25 k1", min_value=0.1, max_value=3.0, value=1.5, step=0.1)
    b = st.sidebar.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    top_k = st.sidebar.slider("Top-K results", min_value=1, max_value=20, value=10, step=1)
    candidate_k = st.sidebar.slider("Hybrid candidate pool", min_value=1, max_value=50, value=20, step=1)
    vector_size = st.sidebar.slider("Embedding vector size", min_value=8, max_value=128, value=64, step=8)

    return SidebarState(
        base_url=base_url,
        dataset_name=dataset_name,
        max_documents=int(max_documents),
        representation_model=representation_model,
        k1=k1,
        b=b,
        top_k=top_k,
        candidate_k=candidate_k,
        vector_size=vector_size,
    )
