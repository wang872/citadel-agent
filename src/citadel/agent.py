from __future__ import annotations

from citadel.answer import ExtractiveAnswerer, parse_citations, verify_citations
from citadel.chunk import load_kb
from citadel.guard import Guard
from citadel.index import HybridIndex
from citadel.memory import SessionMemory
from citadel.orders import OrderBook, extract_order_id, load_orders, wants_refund
from citadel.schema import AgentResponse, Hit
from citadel.tokenize import tokenize

# 最高分切片和问题的词面重合低于该比例时，认为这轮检索没有落地，改写后再查一次。
GROUND_THRESHOLD = 0.22

_SYNONYMS = {
    "价格": ["定价", "费用", "开销", "多少钱"],
    "开销": ["价格", "定价", "费用"],
    "多少钱": ["价格", "定价"],
    "退款": ["退费", "退钱", "发票"],
    "退费": ["退款", "发票"],
    "可用性": ["sla", "宕机", "可用"],
    "sla": ["可用性"],
    "加密": ["aes-256", "静态"],
    "事故": ["故障", "复盘", "分钟"],
}


class KnowledgeAgent:
    """客服知识 Agent。

    顺序固定：输入防护 → 指代消解 → 订单工具 → 混合检索 → 必要时改写再检索
    → 带引用回答 → 引用校验 → 输出脱敏 → 写入会话记忆。
    """

    def __init__(
        self,
        index: HybridIndex | object,
        orders: OrderBook | None = None,
        answerer: object | None = None,
        memory: SessionMemory | None = None,
        guard: Guard | None = None,
    ) -> None:
        self.index = index
        self.orders = orders or OrderBook([])
        self.answerer = answerer or ExtractiveAnswerer()
        self.fallback = ExtractiveAnswerer()
        self.memory = memory or SessionMemory()
        self.guard = guard or Guard()

    def answer(self, question: str, session_id: str = "default") -> AgentResponse:
        trace: list[str] = []
        decision = self.guard.check_input(question)
        if decision.blocked:
            trace.append("guard:block")
            return AgentResponse(text=decision.message, blocked=True, trace=trace)

        safe_question = decision.text
        trace.append("guard:allow")
        resolved = self.memory.resolve(session_id, safe_question)
        if resolved != safe_question:
            trace.append("memory:followup")

        tool_facts = None
        order_id = extract_order_id(safe_question)
        if order_id and wants_refund(safe_question):
            tool_facts = self.orders.decide_refund(order_id)
            trace.append(f"tool:refund:{order_id}")
            resolved = f"{resolved} 退款政策 7 天 发票"

        hits, retrieve_trace = self._retrieve(resolved)
        trace.extend(retrieve_trace)
        raw = self.answerer.answer(safe_question, hits, tool_facts)
        verified, repaired = verify_citations(raw, hits, safe_question, self.fallback, tool_facts)
        if repaired:
            trace.append("cite:repair")
        text = self.guard.redact(verified)
        self.memory.add(session_id, safe_question, text)
        return AgentResponse(
            text=text,
            citations=parse_citations(text),
            blocked=False,
            trace=trace,
            retrieved=[hit.chunk.id for hit in hits],
            repaired=repaired,
        )

    def _retrieve(self, query: str) -> tuple[list[Hit], list[str]]:
        hits = self.index.search(query, k=4)
        trace = ["retrieve:1"]
        if _coverage(query, hits) >= GROUND_THRESHOLD:
            return hits, trace
        rewritten = expand_query(query)
        trace.append(f"rewrite:{rewritten}")
        if rewritten == query:
            return hits, trace
        retried = self.index.search(rewritten, k=4)
        trace.append("retrieve:2")
        return (retried or hits), trace


def expand_query(query: str) -> str:
    lowered = query.lower()
    extras: list[str] = []
    for key, synonyms in _SYNONYMS.items():
        if key in lowered:
            extras.extend(synonyms)
    if not extras:
        return query
    return f"{query} {' '.join(dict.fromkeys(extras))}"


def _coverage(query: str, hits: list[Hit]) -> float:
    if not hits:
        return 0.0
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    top = hits[0].chunk
    document_terms = set(tokenize(f"{top.title} {top.text}"))
    return len(query_terms & document_terms) / len(query_terms)


def build_agent() -> KnowledgeAgent:
    return KnowledgeAgent(index=HybridIndex(load_kb()), orders=OrderBook(load_orders()))
