from __future__ import annotations

from citadel.bm25 import BM25Index
from citadel.dense import DenseIndex
from citadel.rrf import reciprocal_rank_fusion
from citadel.schema import Chunk, Hit


class HybridIndex:
    """词法 BM25 与哈希向量两路召回，再用 RRF 融合。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.bm25 = BM25Index()
        self.dense = DenseIndex()
        self.bm25.add(chunks)
        self.dense.add(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        lexical = self.bm25.search(query, k=max(k * 3, 8))
        dense = self.dense.search(query, k=max(k * 3, 8))
        fused = reciprocal_rank_fusion(
            [[doc_id for doc_id, _ in lexical], [doc_id for doc_id, _ in dense]],
            top=k,
        )
        lexical_scores = dict(lexical)
        return [
            Hit(chunk=self.chunks[doc_id], score=score, lexical=lexical_scores.get(doc_id, 0.0))
            for doc_id, score in fused
            if doc_id in self.chunks
        ]
