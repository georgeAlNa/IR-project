from __future__ import annotations

import streamlit as st

from web_ui.components.sidebar import render_sidebar
from web_ui.services.api_client import (
    build_evaluation_payload,
    build_query_vectors,
    build_ranked_documents,
    build_search_payload,
    evaluate_models,
    get_dataset_catalog,
    index_documents,
    preprocess_text,
    prepare_dataset_vectors,
    refine_query,
    search_documents,
)


MODEL_TO_REPRESENTATION = {
    "TF-IDF": "tfidf",
    "BM25": "bm25",
    "Embedding": "embeddings",
    "Hybrid Serial": "hybrid_serial",
    "Hybrid Parallel": "hybrid_parallel",
    "BERT (FAISS)": "bert",
}

HYBRID_EVALUATION_REPRESENTATION = "hybrid_parallel"


st.set_page_config(page_title="IR Web UI", layout="wide")

catalog = get_dataset_catalog()
sidebar_state = render_sidebar()
dataset = catalog[sidebar_state.dataset_name]

st.title("IR Web UI / API Gateway")
st.caption("Search, ranking, and evaluation across the IR microservices stack.")

with st.form("search_form", clear_on_submit=False):
    search_query = st.text_input("Search", value=st.session_state.get("search_query", dataset.benchmark_query))
    search_submitted = st.form_submit_button("Search")


def _safe_request(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # pragma: no cover - UI surface
        return None, f"{label} failed: {exc}"


if search_submitted:
    st.session_state["search_query"] = search_query
    selected_representation = MODEL_TO_REPRESENTATION[sidebar_state.representation_model]
    indexing_representation = selected_representation if selected_representation in {"tfidf", "bm25", "embeddings"} else "tfidf"

    if sidebar_state.enable_query_refinement:
        refined_query_result, error = _safe_request("Query refinement", refine_query, sidebar_state.base_url, search_query)
        if error:
            st.error(error)
            st.stop()
        refined_query = refined_query_result.get("Corrected_Query", search_query)
    else:
        refined_query = search_query
    preprocessed_query_result, error = _safe_request("Query preprocessing", preprocess_text, sidebar_state.base_url, refined_query)
    if error:
        st.error(error)
        st.stop()

    processed_query = preprocessed_query_result.get("Processed_Text", refined_query)

    if dataset.ir_dataset_name:
        indexing_result, error = _safe_request(
            "Dataset indexing",
            index_documents,
            sidebar_state.base_url,
            [],
            indexing_representation,
            sidebar_state.k1,
            sidebar_state.b,
            sidebar_state.vector_size,
            dataset.ir_dataset_name,
            sidebar_state.max_documents,
        )
        if error:
            st.error(error)
            st.stop()

        if selected_representation == "bert":
            query_vectors_result = {}
            search_payload = {}
        else:
            query_vectors_result = build_query_vectors(
                sidebar_state.base_url,
                processed_query,
                sidebar_state.k1,
                sidebar_state.b,
                sidebar_state.vector_size,
                dataset.ir_dataset_name,
            )
            search_payload = build_search_payload(selected_representation, query_vectors_result)
            
        ranking_result, error = _safe_request(
            "Dataset ranking",
            search_documents,
            sidebar_state.base_url,
            selected_representation,
            search_payload,
            None,
            sidebar_state.top_k,
            sidebar_state.candidate_k,
            dataset.ir_dataset_name,
            search_query,
        )
        if error:
            st.error(error)
            st.stop()

        ranked_ids = ranking_result.get("Ranked_Document_Ids", [])
        ranked_documents = ranking_result.get("Ranked_Documents") or [
            {"Document_Id": document_id, "Processed_Text": ""}
            for document_id in ranked_ids
        ]

        st.session_state["last_search"] = {
            "dataset_name": sidebar_state.dataset_name,
            "ir_dataset_name": dataset.ir_dataset_name,
            "query_id": dataset.query_id,
            "query": search_query,
            "refined_query": refined_query,
            "processed_query": processed_query,
            "indexing_result": indexing_result,
            "ranking_result": ranking_result,
            "ranked_documents": ranked_documents,
            "prepared_documents": [],
            "query_vectors": query_vectors_result,
            "representation": selected_representation,
        }
        st.rerun()

    processed_documents: list[dict[str, str]] = []
    for document in dataset.documents:
        processed_document_result, error = _safe_request("Document preprocessing", preprocess_text, sidebar_state.base_url, document.text)
        if error:
            st.error(error)
            st.stop()
        processed_documents.append(
            {
                "Document_Id": document.document_id,
                "Original_Text": document.text,
                "Processed_Text": processed_document_result.get("Processed_Text", document.text),
            }
        )

    indexing_result, error = _safe_request(
        "Indexing",
        index_documents,
        sidebar_state.base_url,
        processed_documents,
        indexing_representation,
        sidebar_state.k1,
        sidebar_state.b,
        sidebar_state.vector_size,
    )
    if error:
        st.error(error)
        st.stop()

    if selected_representation == "bert":
        query_vectors_result = {}
        search_payload = {}
    else:
        query_vectors_result = build_query_vectors(
            sidebar_state.base_url,
            processed_query,
            sidebar_state.k1,
            sidebar_state.b,
            sidebar_state.vector_size,
        )
        search_payload = build_search_payload(selected_representation, query_vectors_result)

    prepared_documents = prepare_dataset_vectors(processed_documents, sidebar_state.k1, sidebar_state.b, sidebar_state.vector_size)

    ranking_result, error = _safe_request(
        "Ranking",
        search_documents,
        sidebar_state.base_url,
        selected_representation,
        search_payload,
        prepared_documents,
        sidebar_state.top_k,
        sidebar_state.candidate_k,
        None,
        search_query,
    )
    if error:
        st.error(error)
        st.stop()

    ranked_ids = ranking_result.get("Ranked_Document_Ids", [])
    ranked_documents = build_ranked_documents(processed_documents, ranked_ids)

    st.session_state["last_search"] = {
        "dataset_name": sidebar_state.dataset_name,
        "query_id": dataset.query_id,
        "query": search_query,
        "refined_query": refined_query,
        "processed_query": processed_query,
        "indexing_result": indexing_result,
        "ranking_result": ranking_result,
        "ranked_documents": ranked_documents,
        "prepared_documents": prepared_documents,
        "query_vectors": query_vectors_result,
        "representation": selected_representation,
    }

last_search = st.session_state.get("last_search")

if last_search:
    st.subheader("Refined Query")
    st.write(last_search["refined_query"])

    if last_search.get("ir_dataset_name"):
        st.subheader("Retrieved Documents")
        ranked_documents = last_search.get("ranked_documents", [])
        if ranked_documents:
            for position, document in enumerate(ranked_documents, start=1):
                with st.container():
                    st.write(f"#{position} - {document['Document_Id']}")
                    document_text = document.get("Original_Text") or document.get("Processed_Text")
                    if document_text:
                        st.write(document_text)
                    else:
                        st.caption("Document text is not available in this response.")
        else:
            st.info("No documents were returned for the current search.")
    else:
        st.subheader("Retrieved Documents")
        if last_search["ranked_documents"]:
            for position, document in enumerate(last_search["ranked_documents"], start=1):
                with st.container():
                    st.write(f"#{position} - {document['Document_Id']}")
                    st.write(document.get("Original_Text") or document["Processed_Text"])
        else:
            st.info("No documents were returned for the current search.")

    with st.expander("Indexing Response"):
        st.json(last_search["indexing_result"])

    if last_search.get("ranking_result") is not None:
        with st.expander("Ranking Response"):
            st.json(last_search["ranking_result"])

    force_recalculate = st.checkbox("Force Recalculate (Takes time)", value=False)
    
    if last_search.get("ir_dataset_name"):
        if st.button("Run Evaluation Service", key="btn_eval_dataset"):
            evaluation_result, error = _safe_request(
                "Evaluation service",
                evaluate_models,
                sidebar_state.base_url,
                [],
                [],
                10,
                last_search["ir_dataset_name"],
                10,
                force_recalculate,
            )
            if error:
                st.error(error)
                st.stop()

            st.subheader("Evaluation Metrics")
            st.json(evaluation_result)
            
            metrics_data = evaluation_result.get("Metrics_By_Model", {})
            if metrics_data:
                import pandas as pd
                chart_data = []
                for model, metrics in metrics_data.items():
                    chart_data.append({
                        "Model": model,
                        "MAP": metrics.get("MAP", 0.0),
                        "nDCG": metrics.get("nDCG", 0.0)
                    })
                df = pd.DataFrame(chart_data).set_index("Model")
                st.bar_chart(df[["MAP", "nDCG"]], use_container_width=True)
    elif st.button("Run Evaluation Service", key="btn_eval_custom"):
        model_runs: dict[str, list[str]] = {}
        evaluation_models = ["TF-IDF", "BM25", "Embedding", "Hybrid"]

        for model_name in evaluation_models:
            if model_name == "Hybrid":
                representation = (
                    MODEL_TO_REPRESENTATION[sidebar_state.representation_model]
                    if sidebar_state.representation_model in {"Hybrid Serial", "Hybrid Parallel"}
                    else HYBRID_EVALUATION_REPRESENTATION
                )
            else:
                representation = MODEL_TO_REPRESENTATION[model_name]
            model_payload = build_search_payload(representation, last_search["query_vectors"])
            model_result, error = _safe_request(
                f"Evaluation search for {model_name}",
                search_documents,
                sidebar_state.base_url,
                representation,
                model_payload,
                last_search["prepared_documents"],
                sidebar_state.top_k,
                sidebar_state.candidate_k,
            )
            if error:
                st.error(error)
                st.stop()
            model_runs[model_name] = model_result.get("Ranked_Document_Ids", [])

        qrels_payload, system_results_payload = build_evaluation_payload(dataset.query_id, dataset.qrels, model_runs)
        evaluation_result, error = _safe_request(
            "Evaluation service",
            evaluate_models,
            sidebar_state.base_url,
            qrels_payload,
            system_results_payload,
            10,
        )
        if error:
            st.error(error)
            st.stop()

        st.subheader("Evaluation Metrics")
        st.json(evaluation_result)
        
        metrics_data = evaluation_result.get("Metrics_By_Model", {})
        if metrics_data:
            import pandas as pd
            chart_data = []
            for model, metrics in metrics_data.items():
                chart_data.append({
                    "Model": model,
                    "MAP": metrics.get("MAP", 0.0),
                    "nDCG": metrics.get("nDCG", 0.0)
                })
            df = pd.DataFrame(chart_data).set_index("Model")
            st.bar_chart(df[["MAP", "nDCG"]], use_container_width=True)
