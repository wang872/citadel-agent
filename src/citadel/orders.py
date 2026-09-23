from __future__ import annotations

import json
import re
from pathlib import Path

from citadel.schema import RefundDecision

_ORDER_ID = re.compile(r"ORD-\d+", re.IGNORECASE)
_REFUND_WORDS = ("退款", "退费", "退钱")


class OrderBook:
    def __init__(self, orders: list[dict]) -> None:
        self.orders = {str(order["id"]).upper(): order for order in orders}

    def decide_refund(self, order_id: str) -> RefundDecision:
        order = self.orders.get(order_id.upper())
        if order is None:
            return RefundDecision(order_id, False, f"未找到订单 {order_id}，无法判断能否退款。")
        plan = order["plan"]
        amount = order["amount"]
        if order["status"] != "paid":
            text = f"订单 {order_id} 当前状态为 {order['status']}，仅已支付订单可以申请退款。"
            return RefundDecision(order_id, True, text)
        if order["invoiced"]:
            text = f"订单 {order_id}（{plan}，{amount} 元）已开具发票，不支持退款。"
            return RefundDecision(order_id, True, text)
        if int(order["age_days"]) > 7:
            days = order["age_days"]
            text = f"订单 {order_id}（{plan}，{amount} 元）已下单 {days} 天，超过 7 天无理由退款期限，不支持退款。"
            return RefundDecision(order_id, True, text)
        days = order["age_days"]
        text = (
            f"订单 {order_id}（{plan}，{amount} 元，已支付，未开票，下单 {days} 天）"
            "符合 7 天内未开票条件，可以全额退款。"
        )
        return RefundDecision(order_id, True, text)


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID.search(text)
    if not match:
        return None
    return match.group(0).upper()


def wants_refund(text: str) -> bool:
    return any(word in text for word in _REFUND_WORDS)


def load_orders(path: Path | None = None) -> list[dict]:
    file_path = path or Path(__file__).resolve().parent / "data" / "orders.json"
    return json.loads(file_path.read_text(encoding="utf-8"))
