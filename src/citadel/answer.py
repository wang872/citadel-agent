from __future__ import annotations

import re

from citadel.schema import Hit, RefundDecision
from citadel.tokenize import tokenize

_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]?")
_CITE = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]")

NO_EVIDENCE = "知识库中没有足够依据回答该问题。"


class ExtractiveAnswerer:
    """从命中的切片里抽取最相关的原句，并强制带上切片编号。

    退款问题优先采用订单工具的确定性结论，再把退款政策切片作为引用。
    """

    def answer(self, question: str, hits: list[Hit], tool_facts: RefundDecision | None = None) -> str:
        if tool_facts is not None:
            return _render_tool_answer(tool_facts, hits)
        ranked = _rank_sentences(question, hits)
        if not ranked:
            return NO_EVIDENCE
        return "".join(f"{sentence} [[{chunk_id}]]" for _, sentence, chunk_id in ranked[:2])


class StaticAnswerer:
    """测试用：固定返回一段可能带错引用的文本。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def answer(self, question: str, hits: list[Hit], tool_facts: RefundDecision | None = None) -> str:
        del question, hits, tool_facts
        return self.text


def parse_citations(text: str) -> list[str]:
    return _CITE.findall(text)


def verify_citations(
    text: str,
    hits: list[Hit],
    question: str,
    answerer: ExtractiveAnswerer,
    tool_facts: RefundDecision | None,
) -> tuple[str, bool]:
    """引用必须指向本次检索结果。对不上就丢弃原文，改用抽取式回答。"""
    valid = {hit.chunk.id for hit in hits}
    cited = parse_citations(text)
    if cited and all(item in valid for item in cited):
        return text, False
    if tool_facts is not None and not tool_facts.found and not cited:
        return text, False
    if not cited and text == NO_EVIDENCE:
        return text, False
    repaired = answerer.answer(question, hits, tool_facts)
    return repaired, repaired != text


def _render_tool_answer(tool_facts: RefundDecision, hits: list[Hit]) -> str:
    if not tool_facts.found:
        return tool_facts.text
    cite_id = ""
    for hit in hits:
        if hit.chunk.doc_id == "refund":
            cite_id = hit.chunk.id
            break
    if not cite_id and hits:
        cite_id = hits[0].chunk.id
    if not cite_id:
        return tool_facts.text
    return f"{tool_facts.text} [[{cite_id}]]"


def _rank_sentences(question: str, hits: list[Hit]) -> list[tuple[int, str, str]]:
    query_terms = set(tokenize(question))
    ranked: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for hit in hits:
        for sentence in _sentences(hit.chunk.text):
            if sentence in seen:
                continue
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                seen.add(sentence)
                ranked.append((overlap, sentence, hit.chunk.id))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.findall(text) if part.strip()]
