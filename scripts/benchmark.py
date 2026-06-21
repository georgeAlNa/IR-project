import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_queries_and_qrels
from core.offline_store import get_bundle
from core.matching_ranking_service import MatchingRankingService
from schemas.matching_ranking_schema import SearchQueryRepresentation
from core.inverted_index import normalize_text
from schemas.evaluation_schema import SystemRun
from main import _load_offline_bundles

_load_offline_bundles()
bundle = get_bundle("beir/quora/test")
indexed_documents = bundle.ranked_documents
ranking_service = MatchingRankingService()

query_lookup, qrels = load_queries_and_qrels("beir/quora/test", max_queries=10)
qrel = qrels[0]
query_text = query_lookup.get(qrel.query_id)
query_tokens = normalize_text(query_text)
tfidf_vec = bundle.strategies.get("tfidf").represent_query(query_tokens, bundle.inverted_index) if bundle.strategies.get("tfidf") else {}
bm25_vec = bundle.strategies.get("bm25").represent_query(query_tokens, bundle.inverted_index) if bundle.strategies.get("bm25") else {}
emb_vec = bundle.strategies.get("embeddings").represent_query(query_tokens, bundle.inverted_index) if bundle.strategies.get("embeddings") else []
query_representation = SearchQueryRepresentation(
    tfidf_vector=tfidf_vec, bm25_vector=bm25_vec, embedding_vector=emb_vec
)

print(f"Dataset size: {len(indexed_documents)}")
print("Benchmarking TF-IDF search...")
t0 = time.time()
ranking_service.search_documents("tfidf", query_representation, indexed_documents, top_k=10, candidate_k=50)
print(f"TF-IDF search took: {time.time() - t0:.4f}s")

print("Benchmarking BM25 search...")
t0 = time.time()
ranking_service.search_documents("bm25", query_representation, indexed_documents, top_k=10, candidate_k=50)
print(f"BM25 search took: {time.time() - t0:.4f}s")

print("Benchmarking Embeddings search...")
t0 = time.time()
ranking_service.search_documents("embeddings", query_representation, indexed_documents, top_k=10, candidate_k=50)
print(f"Embeddings search took: {time.time() - t0:.4f}s")

print("Benchmarking Hybrid search...")
t0 = time.time()
ranking_service.search_documents("hybrid_parallel", query_representation, indexed_documents, top_k=10, candidate_k=50)
print(f"Hybrid search took: {time.time() - t0:.4f}s")
