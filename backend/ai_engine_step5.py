# ai_engine_step5.py — WINNER EDITION (final)
"""
Zoho SalesIQ Hackathon – High-Scoring AI Engine
Features:
- Intent AI (order, pricing, login, support, escalate)
- Emotion AI (happy, angry, neutral)
- Self-harm + hate speech filter
- Context memory
- Deterministic order simulation
- Safe structured output for SalesIQ
"""

import re
import random
from collections import defaultdict, deque
from typing import Dict, Any, Tuple


# -----------------------------
# INTENT RULES
# -----------------------------
GREETING_WORDS = {"hi", "hello", "hey", "hii", "hiya"}

INTENT_RULES = {
    "order": ["order", "track", "tracking", "delivery", "shipment"],
    "pricing": ["price", "pricing", "plan", "subscription", "cost"],
    "login": ["login", "signin", "forgot", "password", "reset"],
    "support": ["help", "issue", "problem", "error", "bug", "crash"],
    "escalate": ["agent", "human", "representative", "supervisor"]
}

SELF_HARM = ["die", "kill myself", "suicide", "end my life", "harm myself"]
ABUSE = ["fuck", "shit", "bitch", "bastard"]
TERROR = ["bomb", "terror", "attack"]


# -----------------------------
# INTENT CLASSIFIER
# -----------------------------
def classify_intent(text: str) -> Tuple[str, float]:
    t = text.lower().strip()

    if not t:
        return "unknown", 0.0

    # greetings
    for g in GREETING_WORDS:
        if t == g or t.startswith(g + " "):
            return "greeting", 1.0

    # rule-based
    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if k in t:
                return intent, 0.92

    # numeric order id
    if re.search(r"\b(\d{3,12})\b", t):
        return "order", 0.95

    return "unknown", 0.4


# -----------------------------
# EMOTION
# -----------------------------
def classify_emotion(text: str) -> Tuple[str, float]:
    t = text.lower()
    if any(k in t for k in ["thanks", "thank", "great", "nice"]):
        return "happy", 0.9
    if any(k in t for k in ["hate", "angry", "frustrat", "annoy"]):
        return "angry", 0.8
    if any(k in t for k in ABUSE):
        return "angry", 1.0
    return "neutral", 0.5


# -----------------------------
# MEMORY
# -----------------------------
class Memory:
    def __init__(self):
        self.data = defaultdict(lambda: {
            "context": deque(maxlen=20),
            "last_intent": None,
            "escalations": 0
        })

    def get(self, uid):
        return self.data[uid]

    def push(self, uid, speaker, text):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})


memory = Memory()


# -----------------------------
# ORDER SYSTEM
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


def simulate_order(order_id: str) -> Dict[str, Any]:
    try:
        seed = int("".join(filter(str.isdigit, order_id))) % 9999
    except:
        seed = sum(ord(c) for c in order_id)

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)

    return {
        "order_id": order_id,
        "stage": ORDER_STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": ORDER_STAGES[:idx + 1]
    }


# -----------------------------
# PRICING
# -----------------------------
PRICES = {
    "basic": "₹499",
    "pro": "₹1299",
    "enterprise": "₹4999"
}

def price_lookup(plan: str):
    plan = plan.lower()
    return PRICES.get(plan, "₹499")


# -----------------------------
# MAIN ENGINE
# -----------------------------
class AIEngine:
    def __init__(self):
        self.memory = memory

    def process(self, user_id: str, message: str) -> Dict[str, Any]:
        msg = (message or "").strip()
        mem = self.memory.get(user_id)
        self.memory.push(user_id, "user", msg)

        # SAFETY FIRST
        t = msg.lower()
        if any(x in t for x in SELF_HARM):
            return {
                "final_answer": "I cannot help with harmful messages.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 1.0,
                "metadata": {}
            }

        if any(x in t for x in TERROR):
            return {
                "final_answer": "I cannot assist with such content.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 1.0,
                "metadata": {}
            }

        if any(x in t for x in ABUSE):
            return {
                "final_answer": "Please avoid inappropriate language.",
                "intent": "blocked",
                "emotion": "angry",
                "confidence": 1.0,
                "metadata": {}
            }

        # CLASSIFY
        intent, conf = classify_intent(msg)
        emotion, _ = classify_emotion(msg)
        mem["last_intent"] = intent

        # ORDER
        if intent == "order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if m:
                oid = m.group(1)
                data = simulate_order(oid)
                return {
                    "final_answer": f"order_status",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 1.0,
                    "metadata": data
                }

        # PRICING
        if intent == "pricing":
            m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
            plan = m.group(1).lower() if m else "basic"

            return {
                "final_answer": "pricing_info",
                "intent": "pricing",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {
                    "plan": plan,
                    "price": price_lookup(plan)
                }
            }

        # LOGIN
        if intent == "login":
            return {
                "final_answer": "You can reset your password using the 'Forgot Password' option.",
                "intent": "login",
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
                "confidence": 0.95,
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

        # GREETING
        if intent == "greeting":
            return {
                "final_answer": "Hi! How can I help you today?",
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # UNKNOWN
        return {
            "final_answer": "I couldn't understand that. Can you rephrase?",
            "intent": "unknown",
            "emotion": emotion,
            "confidence": 0.3,
            "metadata": {}
        }
