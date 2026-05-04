"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        # Implementation: BM25 search using rank-bm25
        # 1. tokenized_query = segment_vietnamese(query).split()
        # 2. scores = self.bm25.get_scores(tokenized_query)
        # 3. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 4. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self.documents[idx]["text"],
                score=scores[idx],
                metadata=self.documents[idx]["metadata"],
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
            # Check connection
            self.client.get_collections()
        except Exception:
            print(f"  [WARNING] Qdrant server not found at {QDRANT_HOST}:{QDRANT_PORT}. Falling back to in-memory storage.")
            self.client = QdrantClient(location=":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        # Implementation: Dense indexing using Qdrant and SentenceTransformers
        # 1. from qdrant_client.models import Distance, VectorParams, PointStruct
        # 2. self.client.recreate_collection(collection, VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        # 3. texts = [c["text"] for c in chunks]
        # 4. vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        # 5. points = [PointStruct(id=i, vector=v.tolist(), payload={**c["metadata"], "text": c["text"]}) ...]
        # 6. self.client.upsert(collection, points)
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        points = []
        for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
            payload = {
                **chunk["metadata"],
                "text": chunk["text"]
            }
            points.append(PointStruct(
                id=i,
                vector=vec.tolist(),
                payload=payload
            ))

        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # Implementation: Dense search with top-k results
        # 1. query_vector = self._get_encoder().encode(query).tolist()
        # 2. hits = self.client.search(collection, query_vector, limit=top_k)
        # 3. Return [SearchResult(text=hit.payload["text"], score=hit.score, metadata=hit.payload, method="dense")]
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )
        hits = response.points

        results = []
        for hit in hits:
            results.append(SearchResult(
                text=hit.payload["text"],
                score=hit.score,
                metadata=hit.payload,
                method="dense"
            ))
        return results
        

def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # Implementation: Reciprocal Rank Fusion (RRF)
    # 1. rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    # 2. For each result_list in results_list:
    #      For rank, result in enumerate(result_list):
    #        rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    # 3. Sort by score descending
    # 4. Return top_k SearchResult with method="hybrid"
    rrf_scores = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            # Convert metadata to a hashable form (lists -> tuples)
            hashable_metadata = []
            for m_key, m_val in sorted(result.metadata.items()):
                if isinstance(m_val, list):
                    m_val = tuple(m_val)
                elif isinstance(m_val, dict):
                    # Convert nested dicts to sorted tuples
                    m_val = tuple(sorted(m_val.items()))
                hashable_metadata.append((m_key, m_val))
            
            key = (result.text, tuple(hashable_metadata))

            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}

            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)
    
    sorted_rrf = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    results = []
    for r in sorted_rrf[:top_k]:
        res = r["result"]
        results.append(SearchResult(
            text=res.text,
            score=r["score"],
            metadata=res.metadata,
            method="hybrid"
        ))
    return results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
