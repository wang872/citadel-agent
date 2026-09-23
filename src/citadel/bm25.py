from __future__ import annotations

import math
from collections import Counter

from citadel.schema import Chunk
from citadel.tokenize import tokenize


class BM25Index:
    """Okapi BM25。查询词去重，避免同一句话里的重复二字切分把分数打爆。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._lengths: list[int] = []
        self._avgdl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(f"{chunk.title} {chunk.text}")
            counts = Counter(tokens)
            self.chunks.append(chunk)
            self._tf.append(counts)
            self._lengths.append(len(tokens) or 1)
            for term in counts:
                self._df[term] += 1
        total_length = sum(self._lengths)
        self._avgdl = total_length / len(self._lengths) if self._lengths else 0.0

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if k < 1 or not self.chunks:
            return []
        terms = list(dict.fromkeys(tokenize(query)))
        total = len(self.chunks)
        scored: list[tuple[str, float]] = []
        for index, chunk in enumerate(self.chunks):
            score = 0.0
            length = self._lengths[index]
            counts = self._tf[index]
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (total - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1))
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((chunk.id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
