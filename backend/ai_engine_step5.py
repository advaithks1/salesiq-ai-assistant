# ai_engine_step5.py — FINAL SAFE HACKATHON VERSION
"""
Lightweight AI engine designed for Zoho SalesIQ Hackathon.

Features:
- Intent detection (order, pricing, support, greeting, escalate)
- Emotion detection (simple keyword rules)
- Pricing metadata for SalesIQ card
- Order tracking simulation (stable deterministic stages)
- 100% SalesIQ-safe responses (no emojis, no bullets)
"""

import re
import time
import random
from collections import defaultdict, deque
from typing import Dict, Any, Tuple


# -----------------------------
# INTENT + EMOTION RULES
# -----------------------------

GREETING_WORDS = {"hi", "hello", "hey", "hii", "hiya"}

INTENT_RULES = {
    "order": ["order", "track", "tracking", "delivery", "shipment"],
    "pricing": ["price", "pricing", "plan", "subscription", "cost"],
    "login": ["login", "signin", "forgot", "password", "reset"],
    "support": ["help", "issue", "problem", "error", "bug", "crash"],
    "escalate": ["agent", "human", "representative", "supervisor"]
}


def classify_intent(text: str) -> Tuple[str, float]:
    msg = (text or "").lower().strip()

    if not msg:
        return "unknown", 0.0

    # special: greeting
    for g in GREETING_WORDS:
        if msg == g or msg.startswith(g + " "):
            return "greeting", 1.0

    # rule-based
    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if k in msg:
                return intent, 0.92

    # detect numeric order id
    if re.search(r"\b(\d{3,12})\b", msg):
        return "order", 0.95

    return "unknown", 0.35


def classify_emotion(text: str) -> Tuple[str, float]:
    msg = (text or "").lower()

    if any(w in msg for w in ["thanks", "thank", "nice", "great"]):
        return "happy", 0.8
    if any(w in msg for w in ["angry", "hate", "fuck", "frustrat"]):
        return "angry", 0.9

    return "neutral", 0.5


# -----------------------------
# MEMORY
# -----------------------------

class Memory:
    def __init__(self, ctx_size=20):
        self.data = defaultdict(
            lambda: {
                "escalations": 0,
                "last_intent": None,
                "expect": None,
                "context": deque(maxlen=ctx_size)
            }
        )

    def get(self, uid):
        return self.data[uid]

    def push_context(self, uid, speaker, text):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})


memory = Memory()


# -----------------------------
# ORDER SIMULATION
# -----------------------------

ORDER_STAGES = [
    "Order confirmed",
    "Packing",
    "Ready to ship",
    "Shipped",
    "In transit",
    "Out for delivery",
    "Delivered"
]


def simulate_order(oid: str) -> Dict[str, Any]:
    try:
        seed = int("".join(filter(str.isdigit, oid))) % 9999
    except:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)

    return {
        "order_id": oid,
        "stage": ORDER_STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": ORDER_STAGES[:idx + 1]
    }


# -----------------------------
# PRICING
# -----------------------------

def simulated_price(plan: str) -> str:
    plan = plan.lower()
    prices = {
        "basic": "₹499",
        "pro": "₹1299",
        "enterprise": "₹4999"
    }
    return prices.get(plan, prices["basic"])


# -----------------------------
# MAIN ENGINE
# -----------------------------

class AIEngine:
    def __init__(self):
        self.memory = memory

    def process(self, user_id: str, message: str) -> Dict[str, Any]:
        msg = (message or "").strip()
        mem = self.memory.get(user_id)
        self.memory.push_context(user_id, "user", msg)

        if not msg:
            return {
                "final_answer": "Please type your question.",
                "intent": "unknown",
                "emotion": "neutral",
                "confidence": 0.0,
                "metadata": {}
            }

        # Safety
        if any(w in msg.lower() for w in ["kill", "bomb", "suicide", "terror"]):
            return {
                "final_answer": "I cannot help with that.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 0.0,
                "metadata": {}
            }

        # Intent + emotion
        intent, intent_conf = classify_intent(msg)
        emotion, _ = classify_emotion(msg)

        # ORDER flow
        if intent == "order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if m:
                oid = m.group(1)
                order = simulate_order(oid)
                mem["last_intent"] = "order"
                return {
                    "final_answer": f"Order {oid} status: {order['stage']}",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 1.0,
                    "metadata": {
                        "order_id": oid,
                        "order_stage": order["stage"],
                        "eta_days": order["eta_days"]
                    }
                }

        # PRICING flow
        if intent == "pricing":
            # detect plan
            m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
            plan = m.group(1).lower() if m else "basic"

            price = simulated_price(plan)
            mem["last_intent"] = "pricing"

            # VERY IMPORTANT:
            # final_answer should be simple (NO emojis, NO formatting)
            return {
                "final_answer": plan + "_plan",
                "intent": "pricing",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": {
                    "plan": plan,
                    "price": price
                }
            }

        # GREETING
        if intent == "greeting":
            return {
                "final_answer": "Hi! How can I help you today?",
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # SUPPORT
        if intent == "support":
            return {
                "final_answer": "Please describe the issue.",
                "intent": "support",
                "emotion": emotion,
                "confidence": 0.85,
                "metadata": {}
            }

        # ESCALATE
        if intent == "escalate":
            mem["escalations"] += 1
            return {
                "final_answer": "Connecting you to an agent.",
                "intent": "escalate",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # FALLBACK
        return {
            "final_answer": "I couldn't understand that. Can you rephrase?",
            "intent": "unknown",
            "emotion": emotion,
            "confidence": 0.3,
            "metadata": {}
        }


# Manual testing
if __name__ == "__main__":
    eng = AIEngine()
    while True:
        t = input("You: ")
        print(eng.process("local", t))
