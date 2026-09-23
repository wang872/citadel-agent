from __future__ import annotations

_FOLLOWUP = ("它", "这个", "那个", "上面", "刚才", "那")


class SessionMemory:
    """会话记忆。出现指代时，把上一轮问题接回当前查询，方便检索补齐主语。"""

    def __init__(self) -> None:
        self.turns: dict[str, list[dict[str, str]]] = {}

    def add(self, session_id: str, question: str, answer: str) -> None:
        self.turns.setdefault(session_id, []).append({"q": question, "a": answer})

    def history(self, session_id: str) -> list[dict[str, str]]:
        return list(self.turns.get(session_id, []))

    def resolve(self, session_id: str, question: str) -> str:
        history = self.turns.get(session_id) or []
        if not history or not any(marker in question for marker in _FOLLOWUP):
            return question
        return f"{history[-1]['q']} {question}"
