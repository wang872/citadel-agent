from __future__ import annotations

import hashlib
import math

from citadel.schema import Chunk
from citadel.tokenize import tokenize


class HashingEmbedder:
    """特征哈希向量。

    用来把稠密召回的接口先跑通，并和词法召回做融合。
    换成真实 embedding 模型时，只要保持 `embed()` 返回等长向量即可。
    """

    def __init__(self, dim: int = 128) -> None:
        if dim < 8:
            raise ValueError("dim 至少为 8")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class DenseIndex:
    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.ids.append(chunk.id)
            self.vectors.append(self.embedder.embed(f"{chunk.title} {chunk.text}"))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if k < 1 or not self.ids:
            return []
        query_vector = self.embedder.embed(query)
        scored = [
            (doc_id, cosine(query_vector, vector))
            for doc_id, vector in zip(self.ids, self.vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
