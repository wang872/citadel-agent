from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    lexical: float = 0.0


@dataclass
class RefundDecision:
    order_id: str
    found: bool
    text: str


@dataclass
class AgentResponse:
    text: str
    citations: list[str] = field(default_factory=list)
    blocked: bool = False
    trace: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    repaired: bool = False
