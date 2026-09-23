from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")

_INJECTION = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all |any )?(previous|above) (instructions|prompts)",
        r"system prompt",
        r"reveal (your )?instructions",
        r"jailbreak",
        r"忽略(之前|以上|前面|所有)",
        r"输出系统提示",
        r"不要遵守",
        r"你现在是",
    )
]

REFUSAL = "该请求被安全策略拦截，我不能执行绕过指令或泄露系统提示。"


@dataclass(frozen=True)
class GuardResult:
    blocked: bool
    text: str
    message: str = ""


class Guard:
    """输入拦截提示注入，输入和输出都做 PII 脱敏。"""

    def check_input(self, text: str) -> GuardResult:
        for pattern in _INJECTION:
            if pattern.search(text):
                return GuardResult(blocked=True, text=text, message=REFUSAL)
        return GuardResult(blocked=False, text=self.redact(text))

    def redact(self, text: str) -> str:
        text = _EMAIL.sub("[EMAIL]", text)
        text = _PHONE.sub("[PHONE]", text)
        text = _ID_CARD.sub("[ID]", text)
        return text
