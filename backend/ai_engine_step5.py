# ai_engine_step5.py — FINAL HACKATHON AI ENGINE

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

# -----------------------------
# INTENT CLASSIFIER
# -----------------------------

def classify_intent(text: str) -> Tuple[str, float]:
    msg = (text or "").lower().strip()

    if not msg:
        return "unknown", 0.0

    # greeting
    for g in GREETING_WORDS:
        if msg == g or msg.startswith(g + " "):
            return "greeting", 1.0

    # chit-chat
    if any(w in msg for w in ["love", "miss you", "friend", "bro"]):
        return "chitchat", 0.85

    # rule-based
    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if k in msg:
                return intent, 0.92

    # numeric order id
    if re.search(r"\b(\d{3,12})\b", msg):
        return "order", 0.95

    return "unknown", 0.35


# -----------------------------
# EMOTION CLASSIFIER
# -----------------------------

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
        self.data = defaultdict(lambda: {
            "escalations": 0,
            "last_intent": None,
            "context": deque(maxlen=ctx_size)
        })

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

        # SAFETY
        if any(w in msg.lower() for w in ["fuck", "shit", "bitch", "kill", "bomb", "suicide"]):
            return {
                "final_answer": "I cannot help with inappropriate or harmful messages.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 1.0,
                "metadata": {}
            }

        # CLASSIFY
        intent, confidence = classify_intent(msg)
        emotion, _ = classify_emotion(msg)

        # ORDER
        if intent == "order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if m:
                oid = m.group(1)
                order = simulate_order(oid)
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

        # PRICING
        if intent == "pricing":
            m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
            plan = m.group(1).lower() if m else "basic"
            price = simulated_price(plan)

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

        # CHIT CHAT
        if intent == "chitchat":
            return {
                "final_answer": "I'm here to assist you! How can I help today?",
                "intent": "chitchat",
                "emotion": emotion,
                "confidence": 0.9,
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
