# ai_engine_step5.py — FINAL HACKATHON VERSION (WITH PRODUCT RECOMMENDATION)
"""
Lightweight AI engine for Zoho SalesIQ Hackathon.

Features:
- Rule-based intent detection
- Keyword emotion detection
- Order tracking simulation
- Pricing system
- Product recommendation system
- Safe responses (no emojis)
- Clean metadata for SalesIQ widget
"""

import re
import random
from collections import defaultdict, deque
from typing import Dict, Any, Tuple


# ------------------------------------------------------
# INTENT + EMOTION RULES
# ------------------------------------------------------

GREETING_WORDS = {"hi", "hello", "hey", "hii", "hiya"}

INTENT_RULES = {
    "order": ["order", "track", "tracking", "delivery", "shipment"],
    "pricing": ["price", "pricing", "plan", "subscription", "cost"],
    "login": ["login", "signin", "forgot", "password", "reset"],
    "support": ["help", "issue", "problem", "error", "bug", "crash"],
    "escalate": ["agent", "human", "representative", "supervisor"],
    "product": ["recommend", "suggest", "best camera", "which camera",
                "camera", "product", "cctv", "buy", "best"]
}


def classify_intent(text: str) -> Tuple[str, float]:
    msg = (text or "").lower().strip()

    if not msg:
        return "unknown", 0.0

    # Greeting
    for g in GREETING_WORDS:
        if msg == g or msg.startswith(g + " "):
            return "greeting", 1.0

    # Intent rules
    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if k in msg:
                return intent, 0.92

    # Pure numbers = order ID
    if re.search(r"\b(\d{3,12})\b", msg):
        return "order", 0.95

    return "unknown", 0.35


def classify_emotion(text: str) -> Tuple[str, float]:
    msg = (text or "").lower()

    if any(w in msg for w in ["thanks", "thank", "nice", "great", "love"]):
        return "happy", 0.8
    if any(w in msg for w in ["angry", "hate", "fuck", "frustrat"]):
        return "angry", 0.9

    return "neutral", 0.5


# ------------------------------------------------------
# MEMORY
# ------------------------------------------------------

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


# ------------------------------------------------------
# ORDER SIMULATION
# ------------------------------------------------------

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
    i = random.randint(0, len(ORDER_STAGES) - 1)

    return {
        "order_id": oid,
        "stage": ORDER_STAGES[i],
        "eta_days": max(0, 5 - i),
        "history": ORDER_STAGES[:i + 1]
    }


# ------------------------------------------------------
# PRICING
# ------------------------------------------------------

def simulated_price(plan: str) -> str:
    plan = plan.lower()
    prices = {
        "basic": "₹499",
        "pro": "₹1299",
        "enterprise": "₹4999"
    }
    return prices.get(plan, "₹499")


# ------------------------------------------------------
# PRODUCT RECOMMENDATION
# ------------------------------------------------------

PRODUCTS = {
    "EyeCam Mini": {
        "price": "₹1499",
        "rating": "4.6",
        "use": "Indoor / Small rooms"
    },
    "EyeCam 360 Pro": {
        "price": "₹2899",
        "rating": "4.8",
        "use": "Full home security coverage"
    },
    "EyeCam Outdoor Max": {
        "price": "₹3999",
        "rating": "4.7",
        "use": "Outdoor / Waterproof"
    }
}


def recommend_product():
    best_name = None
    best_info = None
    best_rating = 0.0

    for name, info in PRODUCTS.items():
        r = float(info["rating"])
        if r > best_rating:
            best_rating = r
            best_name = name
            best_info = info

    return best_name, best_info


# ------------------------------------------------------
# MAIN ENGINE
# ------------------------------------------------------

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
                "final_answer": "I cannot help with that request.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 0.0,
                "metadata": {}
            }

        # Intent & Emotion
        intent, intent_conf = classify_intent(msg)
        emotion, _ = classify_emotion(msg)

        # ------------------ ORDER ------------------
        if intent == "order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if m:
                oid = m.group(1)
                order = simulate_order(oid)
                return {
                    "final_answer": "order_details",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 1.0,
                    "metadata": order
                }

        # ------------------ PRICING ------------------
        if intent == "pricing":
            m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
            plan = m.group(1).lower() if m else "basic"
            price = simulated_price(plan)

            return {
                "final_answer": "pricing_details",
                "intent": "pricing",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": {
                    "plan": plan,
                    "price": price
                }
            }

        # ------------------ PRODUCT RECOMMENDATION ------------------
        if intent == "product":
            name, info = recommend_product()
            return {
                "final_answer": "product_recommendation",
                "intent": "product",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": {
                    "product_name": name,
                    "price": info["price"],
                    "rating": info["rating"],
                    "use_case": info["use"]
                }
            }

        # ------------------ GREETING ------------------
        if intent == "greeting":
            return {
                "final_answer": "Hi! How can I help you today?",
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # ------------------ SUPPORT ------------------
        if intent == "support":
            return {
                "final_answer": "Please describe the issue.",
                "intent": "support",
                "emotion": emotion,
                "confidence": 0.85,
                "metadata": {}
            }

        # ------------------ ESCALATE ------------------
        if intent == "escalate":
            mem["escalations"] += 1
            return {
                "final_answer": "Connecting you to an agent.",
                "intent": "escalate",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # ------------------ FALLBACK ------------------
        return {
            "final_answer": "I couldn't understand that. Can you rephrase?",
            "intent": "unknown",
            "emotion": emotion,
            "confidence": 0.3,
            "metadata": {}
        }


# Test locally
if __name__ == "__main__":
    eng = AIEngine()
    while True:
        txt = input("You: ")
        print(eng.process("local", txt))
