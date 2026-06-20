"""
scripts/build_offline_indexes.py
=================================
Offline Indexing Script — run this ONCE before starting the server.

What it does:
  1. Loads a dataset (e.g. msmarco-passage) via the existing DataLoader.
  2. Preprocesses every document through the existing PreprocessingPipeline.
  3. Builds the InvertedIndex and fits TF-IDF, BM25, Word2Vec/Hash strategies
     via the existing IndexingService — no algorithms are re-written.
  4. Serialises the resulting InvertedIndex, strategies, and RankedDocument list
     to disk with joblib.
  5. (Academic) Encodes every passage with a Sentence-Transformer (BERT) model
     and stores the resulting dense vectors in a FAISS flat index on disk.

Usage:
    python scripts/build_offline_indexes.py --dataset msmarco-passage --max-docs 150000
    python scripts/build_offline_indexes.py --dataset beir/quora --max-docs 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── make sure the project root is importable ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # pip install joblib  (already a transitive dep of scikit-learn)

from core.data_loader import DEFAULT_DOCUMENT_LIMIT, load_documents
from core.indexing_service import IndexingService
from core.inverted_index import InvertedIndexBuilder
from core.preprocessing_pipeline import PreprocessingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── output directory ──────────────────────────────────────────────────────────
INDEX_DIR = PROJECT_ROOT / "offline_indexes"


def _safe_dataset_key(dataset_name: str) -> str:
    """Convert 'beir/quora' → 'beir_quora' for use in filenames."""
    return dataset_name.replace("/", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 + 2 + 3 — Traditional IR models (TF-IDF, BM25, Word2Vec/Hash)
# Reuses the existing IndexingService without touching core algorithms.
# ─────────────────────────────────────────────────────────────────────────────
def build_traditional_index(
    dataset_name: str,
    max_docs: int,
    k1: float,
    b: float,
    vector_size: int,
) -> None:
    key = _safe_dataset_key(dataset_name)
    out_dir = INDEX_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Traditional IR Index ===")
    log.info("Dataset : %s  (max_docs=%d)", dataset_name, max_docs)

    # 1. Load raw documents
    log.info("Loading documents from ir_datasets …")
    t0 = time.perf_counter()
    raw_docs = load_documents(dataset_name=dataset_name, limit=max_docs)
    log.info("  Loaded %d documents in %.1fs", len(raw_docs), time.perf_counter() - t0)

    # 2. Preprocess with the existing pipeline
    log.info("Preprocessing documents …")
    pipeline = PreprocessingPipeline.default()
    t0 = time.perf_counter()
    processed: list[tuple[str, str, str]] = []
    for i, doc in enumerate(raw_docs, 1):
        processed_text = pipeline.process(doc.text)
        if processed_text.strip():
            processed.append((doc.document_id, processed_text, doc.text))
        if i % 10_000 == 0:
            log.info("  … preprocessed %d / %d docs", i, len(raw_docs))
    log.info("  Done in %.1fs — %d non-empty docs kept", time.perf_counter() - t0, len(processed))

    if not processed:
        raise RuntimeError(f"All documents were empty after preprocessing for '{dataset_name}'.")

    # 3. Index via the existing IndexingService (builds InvertedIndex + fits strategies)
    log.info("Building inverted index and fitting TF-IDF / BM25 / Embedding strategies …")
    service = IndexingService(
        builder=InvertedIndexBuilder(),
        preprocessing_pipeline=pipeline,
    )
    t0 = time.perf_counter()
    result = service.index_documents(
        documents=processed,
        representation_type="tfidf",   # doesn't restrict which strategies are fitted
        k1=k1,
        b=b,
        vector_size=vector_size,
        dataset_name=None,             # we handle persistence ourselves here
    )
    log.info(
        "  Done in %.1fs — %d docs | vocab=%d | avg_len=%.1f",
        time.perf_counter() - t0,
        result.document_count,
        result.vocabulary_size,
        result.average_document_length,
    )

    # 4. Pull out the objects that need to be saved
    #    (accessing private attributes — same package, acceptable for a build script)
    inverted_index = service._index
    strategies = service._strategies
    ranked_documents = service._build_ranked_documents(
        service._documents, inverted_index, k1, b
    )

    # 4.5 Persist raw text to SQLite and clear it from memory
    import sqlite3
    db_path = out_dir / "documents.db"
    log.info("Saving raw texts to SQLite database at %s …", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, raw_text TEXT)")
    
    for doc in ranked_documents:
        cursor.execute(
            "INSERT OR REPLACE INTO docs (doc_id, raw_text) VALUES (?, ?)",
            (doc.document_id, doc.original_text)
        )
        doc.original_text = None  # Clear to save space in joblib
        
    conn.commit()
    conn.close()

    # 5. Persist with joblib
    log.info("Saving artifacts to %s …", out_dir)
    joblib.dump(inverted_index, out_dir / "inverted_index.joblib")
    joblib.dump(strategies, out_dir / "representation_strategies.joblib")
    joblib.dump(ranked_documents, out_dir / "ranked_documents.joblib")
    log.info("  Saved: inverted_index.joblib, representation_strategies.joblib, ranked_documents.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — BERT + FAISS Dense Index (Academic Requirement)
# ─────────────────────────────────────────────────────────────────────────────
def build_bert_faiss_index(
    dataset_name: str,
    model_name: str,
    batch_size: int,
) -> None:
    """
    Encode all passages with a Sentence-Transformer model and save the resulting
    dense vectors in a FAISS flat (exact) index alongside a mapping file that
    allows us to translate FAISS result positions back to document_ids.

    Required packages (add to requirements.txt before running):
        sentence-transformers
        faiss-cpu          # or faiss-gpu if a CUDA GPU is available
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "BERT+FAISS indexing requires extra packages.\n"
            "Install them with:\n"
            "    pip install sentence-transformers faiss-cpu\n"
        ) from exc

    import numpy as np  # re-import to silence type checkers
    import faiss

    key = _safe_dataset_key(dataset_name)
    out_dir = INDEX_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== BERT + FAISS Dense Index ===")
    log.info("Dataset : %s | Model : %s | batch_size : %d", dataset_name, model_name, batch_size)

    # Load the already-saved ranked documents (original texts) so we don't
    # reload the full dataset from disk again.
    ranked_docs_path = out_dir / "ranked_documents.joblib"
    if not ranked_docs_path.exists():
        raise FileNotFoundError(
            f"{ranked_docs_path} not found.\n"
            "Run build_traditional_index() first (step 1-3)."
        )

    log.info("Loading ranked_documents from disk …")
    ranked_documents = joblib.load(ranked_docs_path)

    # Build parallel lists for encoding
    doc_ids: list[str] = [doc.document_id for doc in ranked_documents]

    # Load texts from SQLite since they were cleared from joblib
    import sqlite3
    db_path = out_dir / "documents.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    texts: list[str] = []
    doc_lookup = {d.document_id: d for d in ranked_documents}
    for doc_id in doc_ids:
        cursor.execute("SELECT raw_text FROM docs WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        raw_text = row[0] if row and row[0] else ""
        if not raw_text:
            d = doc_lookup.get(doc_id)
            raw_text = d.processed_text if d and d.processed_text else ""
        texts.append(raw_text)
    conn.close()

    # Load BERT model
    log.info("Loading SentenceTransformer model '%s' …", model_name)
    model = SentenceTransformer(model_name)
    embedding_dim: int = model.get_sentence_embedding_dimension()  # type: ignore[assignment]
    log.info("  Embedding dimension : %d", embedding_dim)

    # Encode in batches
    log.info("Encoding %d passages (batch_size=%d) …", len(texts), batch_size)
    t0 = time.perf_counter()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → inner-product == cosine
    )
    log.info("  Encoded in %.1fs — shape %s", time.perf_counter() - t0, embeddings.shape)

    # Build FAISS flat inner-product index
    log.info("Building FAISS IndexFlatIP …")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.astype("float32"))
    log.info("  FAISS index contains %d vectors", index.ntotal)

    # Persist FAISS index + doc_id mapping
    faiss_path = out_dir / "faiss_bert.index"
    docids_path = out_dir / "faiss_doc_ids.joblib"
    faiss.write_index(index, str(faiss_path))
    joblib.dump(doc_ids, docids_path)
    log.info("  Saved: faiss_bert.index, faiss_doc_ids.joblib")

    # Also save the model name so the server knows which one to reload
    joblib.dump(model_name, out_dir / "bert_model_name.joblib")
    log.info("  Saved: bert_model_name.joblib ('%s')", model_name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline IR indexes for a dataset."
    )
    parser.add_argument(
        "--dataset",
        default="msmarco-passage",
        choices=["msmarco-passage", "beir/quora"],
        help="Dataset to index (default: msmarco-passage)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_DOCUMENT_LIMIT,
        help=f"Maximum number of documents to load (default: {DEFAULT_DOCUMENT_LIMIT})",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=64,
        help="Embedding vector size for Word2Vec/Hash strategy (default: 64)",
    )
    parser.add_argument(
        "--bert-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformer model name for BERT encoding (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=256,
        help="Batch size for BERT encoding (default: 256)",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip the BERT+FAISS step (faster, for non-academic builds)",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()

    # Step 1–3: Traditional indexes
    build_traditional_index(
        dataset_name=args.dataset,
        max_docs=args.max_docs,
        k1=args.k1,
        b=args.b,
        vector_size=args.vector_size,
    )

    # Step 4: BERT + FAISS (optional but required for academic submission)
    if not args.skip_bert:
        build_bert_faiss_index(
            dataset_name=args.dataset,
            model_name=args.bert_model,
            batch_size=args.bert_batch_size,
        )
    else:
        log.info("Skipping BERT+FAISS step (--skip-bert flag set).")

    log.info("=== All done in %.1f minutes ===", (time.perf_counter() - total_start) / 60)


if __name__ == "__main__":
    main()
