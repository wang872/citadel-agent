from __future__ import annotations

import re

_SEGMENT = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """英文和数字整词保留，中文整段保留并展开二字切分。"""
    tokens: list[str] = []
    for match in _SEGMENT.finditer(text.lower()):
        segment = match.group()
        if "\u4e00" <= segment[0] <= "\u9fff":
            if len(segment) <= 8:
                tokens.append(segment)
            if len(segment) >= 2:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens
