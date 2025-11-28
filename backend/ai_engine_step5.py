# ai_engine_step5.py — FINAL HACKATHON VERSION WITH AGENT ASSIST PRO
"""
Lightweight AI engine for Zoho SalesIQ Hackathon.
Features:
- Intent detection (order, pricing, support, greeting, escalate, login)
- Emotion detection (rule-based)
- Order tracking simulation (deterministic)
- Pricing engine
- Safety filter
- Memory (last 20 messages)
- Agent Assist PRO (summary, frustration, risk, suggestions, last 3 messages)
"""

import re
import random
from collections import defaultdict, deque
from typing import Dict, Any


# ============================================================
#  INTENT RULES
# ============================================================

GREETING_WORDS = {"hi", "hello", "hey", "hii"}

INTENT_RULES = {
    "order": ["track", "order", "delivery"],
    "pricing": ["price", "pricing", "plan", "subscription", "cost"],
    "support": ["help", "issue", "problem", "error", "bug", "crash"],
    "escalate": ["agent", "human", "supervisor"],
    "login": ["password", "reset password", "forgot password"]
}


def classify_intent(text: str):
    msg = (text or "").lower().strip()

    # Greeting
    for g in GREETING_WORDS:
        if msg == g or msg.startswith(g + " "):
            return "greeting", 1.0

    # Login is high priority
    if any(w in msg for w in ["password", "reset password", "forgot password"]):
        return "login", 1.0

    # Rule-based intents
    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if k in msg:
                return intent, 0.92

    # Detect numeric order ID
    if re.search(r"\b(\d{3,12})\b", msg):
        return "order", 0.95

    return "unknown", 0.3


# ============================================================
#  EMOTION
# ============================================================

def classify_emotion(text: str):
    msg = (text or "").lower()

    if any(w in msg for w in ["thanks", "thank", "nice", "great"]):
        return "happy", 0.9
    if any(w in msg for w in ["hate", "angry", "fuck", "stupid", "bad"]):
        return "angry", 0.9
    return "neutral", 0.5


# ============================================================
#  MEMORY
# ============================================================

class Memory:
    def __init__(self):
        self.data = defaultdict(lambda: {
            "context": deque(maxlen=20),
            "escalations": 0
        })

    def push(self, uid: str, speaker: str, text: str):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})


memory = Memory()


# ============================================================
#  ORDER SIMULATION
# ============================================================

ORDER_STAGES = [
    "Order confirmed",
    "Packing",
    "Ready to ship",
    "Shipped",
    "In transit",
    "Out for delivery",
    "Delivered"
]


def simulate_order(order_id: str):
    try:
        seed = int("".join(filter(str.isdigit, order_id))) % 9999
    except:
        seed = sum(ord(c) for c in order_id) % 9999

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)

    return {
        "order_id": order_id,
        "stage": ORDER_STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": ORDER_STAGES[:idx + 1]
    }


# ============================================================
#  PRICING
# ============================================================

def simulated_price(plan: str) -> str:
    plan = plan.lower()
    prices = {
        "basic": "₹499",
        "pro": "₹1299",
        "enterprise": "₹4999"
    }
    return prices.get(plan, "₹499")


# ============================================================
#  AGENT ASSIST PRO
# ============================================================

def suggest_reply(intent: str, emotion: str):
    if intent == "order":
        return "Reassure user and confirm order ID."
    if intent == "pricing":
        return "Offer a quick plan comparison."
    if intent == "support":
        return "Ask for screenshot or error details."
    if emotion == "angry":
        return "Stay calm and apologize politely."
    return "Guide the user to next step."


def frustration_score(context):
    score = 0
    for c in context:
        t = c["text"].lower()
        if any(w in t for w in ["angry", "hate", "bad", "worst", "fuck"]):
            score += 1
    return score


def risk_level(intent, emotion):
    if emotion == "angry":
        return "high"
    if intent == "escalate":
        return "medium"
    return "low"


def summarize(context):
    if not context:
        return "No conversation yet."

    last_text = " ".join(c["text"] for c in list(context)[-3:])
    return "Recent user intent: " + last_text


# ============================================================
#  MAIN ENGINE
# ============================================================

class AIEngine:
    def __init__(self):
        self.memory = memory

    def process(self, user_id: str, message: str) -> Dict[str, Any]:

        msg = (message or "").strip()
        mem = self.memory.data[user_id]

        # save memory
        self.memory.push(user_id, "user", msg)

        # SAFETY FILTER
        if any(w in msg.lower() for w in ["kill", "suicide", "bomb", "terror"]):
            return {
                "final_answer": "I cannot assist with that.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 1.0,
                "metadata": {}
            }

        # intent + emotion
        intent, conf = classify_intent(msg)
        emotion, emo_conf = classify_emotion(msg)

        # Common metadata builder
        def build_meta(custom: Dict[str, Any]):
            ctx = mem["context"]
            total = len(ctx)

            return {
                **custom,
                "suggestion": suggest_reply(intent, emotion),
                "frustration": frustration_score(ctx),
                "risk": risk_level(intent, emotion),
                "summary": summarize(ctx),
                "m1": ctx[-1]["text"] if total > 0 else "",
                "m2": ctx[-2]["text"] if total > 1 else "",
                "m3": ctx[-3]["text"] if total > 2 else ""
            }

        # -------------------------
        # ORDER
        # -------------------------
        if intent == "order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if m:
                oid = m.group(1)
                order = simulate_order(oid)

                return {
                    "final_answer": "order_status",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 1.0,
                    "metadata": build_meta({
                        "order_id": oid,
                        "order_stage": order["stage"],
                        "eta_days": order["eta_days"],
                        "history": order["history"]
                    })
                }

        # -------------------------
        # PRICING
        # -------------------------
        if intent == "pricing":
            m = re.search(r"(basic|pro|enterprise)", msg, re.IGNORECASE)
            plan = m.group(1).lower() if m else "basic"
            price = simulated_price(plan)

            return {
                "final_answer": "pricing_info",
                "intent": "pricing",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": build_meta({
                    "plan": plan,
                    "price": price
                })
            }

        # -------------------------
        # GREETING
        # -------------------------
        if intent == "greeting":
            return {
                "final_answer": "Hi! How can I help you today?",
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # -------------------------
        # SUPPORT
        # -------------------------
        if intent == "support":
            return {
                "final_answer": "Please describe the issue.",
                "intent": "support",
                "emotion": emotion,
                "confidence": 0.9,
                "metadata": build_meta({})
            }

        # -------------------------
        # ESCALATE
        # -------------------------
        if intent == "escalate":
            mem["escalations"] += 1
            return {
                "final_answer": "Connecting you to an agent.",
                "intent": "escalate",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": build_meta({})
            }

        # -------------------------
        # LOGIN
        # -------------------------
        if intent == "login":
            return {
                "final_answer": "Use the 'Forgot Password' option on the login page.",
                "intent": "login",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": {}
            }

        # -------------------------
        # FALLBACK
        # -------------------------
        return {
            "final_answer": "I couldn't understand that. Can you rephrase?",
            "intent": "unknown",
            "emotion": emotion,
            "confidence": 0.3,
            "metadata": build_meta({})
        }
