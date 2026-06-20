import argparse
import logging
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from core.offline_store import OfflineDatasetBundle, register_bundle

log = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parent / "offline_indexes"


def _safe_dataset_name(key: str) -> str:
    """Convert directory key back to dataset name, e.g. 'beir_quora' → 'beir/quora'."""
    known = {"beir_quora_test": "beir/quora/test"}
    return known.get(key, key)


def _load_offline_bundles() -> None:
    """Scan INDEX_DIR for pre-built dataset bundles and load them into memory."""
    if not INDEX_DIR.exists():
        log.warning(
            "offline_indexes/ directory not found. "
            "Dynamic /index endpoint still works, but offline datasets are unavailable. "
            "Run: python scripts/build_offline_indexes.py"
        )
        return

    try:
        import joblib
    except ImportError:
        log.error("joblib is not installed — offline indexes cannot be loaded.")
        return

    for dataset_dir in sorted(INDEX_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        index_path = dataset_dir / "inverted_index.joblib"
        strategies_path = dataset_dir / "representation_strategies.joblib"
        ranked_docs_path = dataset_dir / "ranked_documents.joblib"

        if not (index_path.exists() and strategies_path.exists() and ranked_docs_path.exists()):
            log.warning("Skipping '%s' — missing core joblib files.", dataset_dir.name)
            continue

        dataset_name = _safe_dataset_name(dataset_dir.name)
        log.info("Loading offline index for '%s' …", dataset_name)
        t0 = time.perf_counter()

        inverted_index = joblib.load(index_path)
        strategies = joblib.load(strategies_path)
        ranked_documents = joblib.load(ranked_docs_path)

        # Optional BERT + FAISS
        faiss_index = None
        faiss_doc_ids = None
        bert_model = None

        faiss_path = dataset_dir / "faiss_bert.index"
        docids_path = dataset_dir / "faiss_doc_ids.joblib"
        model_name_path = dataset_dir / "bert_model_name.joblib"

        if faiss_path.exists() and docids_path.exists() and model_name_path.exists():
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                faiss_doc_ids = joblib.load(docids_path)
                bert_model_name = joblib.load(model_name_path)
                log.info("  Loading FAISS index and BERT model '%s' …", bert_model_name)
                faiss_index = faiss.read_index(str(faiss_path))
                bert_model = SentenceTransformer(bert_model_name)
                log.info("  FAISS index: %d vectors | dim: %d", faiss_index.ntotal, faiss_index.d)
            except ImportError:
                log.warning(
                    "  faiss-cpu or sentence-transformers not installed — "
                    "BERT dense search disabled for '%s'.",
                    dataset_name,
                )

        bundle = OfflineDatasetBundle(
            dataset_name=dataset_name,
            inverted_index=inverted_index,
            strategies=strategies,
            ranked_documents=ranked_documents,
            faiss_index=faiss_index,
            faiss_doc_ids=faiss_doc_ids,
            bert_model=bert_model,
        )
        register_bundle(bundle)
        log.info(
            "  '%s' ready — %d docs | vocab=%d | %.1fs",
            dataset_name,
            inverted_index.document_count,
            len(inverted_index.vocabulary),
            time.perf_counter() - t0,
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load offline indexes before accepting requests; clean up on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Server starting — loading offline indexes …")
    _load_offline_bundles()
    log.info("Offline indexes loaded. Server is ready.")
    yield
    log.info("Server shutting down.")

from api.indexing import router as indexing_router
from api.evaluation import router as evaluation_router
from api.matching_ranking import router as matching_ranking_router
from api.query_refinement import router as query_refinement_router
from api.preprocessing import router as preprocessing_router

app = FastAPI(
    title="IR Microservices Gateway",
    version="1.0.0",
    description="Single entrypoint that serves preprocessing, indexing/representation, matching/ranking, evaluation, and query refinement services.",
    lifespan=lifespan,
)

app.include_router(indexing_router)
app.include_router(evaluation_router)
app.include_router(matching_ranking_router)
app.include_router(query_refinement_router)
app.include_router(preprocessing_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "IR Microservices Gateway",
        "docs": "/docs",
        "web_ui": "http://127.0.0.1:8501",
        "preprocess_endpoint": "/preprocess",
        "index_endpoint": "/index",
        "represent_endpoint": "/represent",
        "search_endpoint": "/search",
        "evaluate_endpoint": "/evaluate",
        "refine_endpoint": "/refine",
    }


def _start_api_process(host: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
    )


def _start_web_process(host: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "web_ui/app.py",
            "--server.address",
            host,
            "--server.port",
            str(port),
        ]
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_combined(api_host: str, api_port: int, ui_host: str, ui_port: int) -> None:
    api_process = _start_api_process(api_host, api_port)
    web_process = _start_web_process(ui_host, ui_port)

    print(f"API running on http://{api_host}:{api_port}")
    print(f"Web UI running on http://{ui_host}:{ui_port}")

    try:
        while True:
            if api_process.poll() is not None:
                raise RuntimeError("API process exited unexpectedly.")
            if web_process.poll() is not None:
                raise RuntimeError("Web UI process exited unexpectedly.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_process(api_process)
        _stop_process(web_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IR backend and frontend from one entrypoint.")
    parser.add_argument("--mode", choices=["all", "api", "web"], default="all")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    if args.mode == "api":
        uvicorn.run("main:app", host=args.api_host, port=args.api_port, reload=False)
    elif args.mode == "web":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "web_ui/app.py",
                "--server.address",
                args.ui_host,
                "--server.port",
                str(args.ui_port),
            ],
            check=False,
        )
    else:
        run_combined(args.api_host, args.api_port, args.ui_host, args.ui_port)
