# ai_engine_step5.py — FINAL PRO+ HACKATHON VERSION
"""
Smart AI Engine for SalesIQ Hackathon
Features:
- Intent detection (order, pricing, support, greeting, login, escalate)
- FULL sentiment engine (lexicon + negation)
- FAQ system (shipping, refund, payment, account)
- Order tracking simulation
- Product browsing (handled by backend)
- Safety filter
- Memory & conversation context
- Agent Assist PRO (frustration, risk, summary, suggestions)
"""

import re, random
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
    "login": ["password", "reset password", "forgot password"],
}


def classify_intent(text: str):
    msg = (text or "").lower().strip()

    # Greeting
    for g in GREETING_WORDS:
        if msg == g or msg.startswith(g + " "):
            return "greeting", 1.0

    # Login high priority
    if any(w in msg for w in ["password", "reset password", "forgot password"]):
        return "login", 1.0

    # Rule-based intents
    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if k in msg:
                return intent, 0.92

    # Auto-detect numeric order ID
    if re.search(r"\b(\d{3,12})\b", msg):
        return "order", 0.95

    return "unknown", 0.3


# ============================================================
#  SENTIMENT ENGINE (Improved)
# ============================================================

POS = {"great","good","awesome","nice","thanks","thank","perfect","love","cool","happy","amazing"}
NEG = {"bad","worst","slow","issue","problem","error","bug","crash","delay","late","annoying"}
STRONG_NEG = {"hate","angry","furious","terrible","stupid","fuck","fucking","shit","useless"}
NEGATION = {"not","never","no","dont","don't","can't","cant"}


def sentiment_score(text: str):
    msg = (text or "").lower()
    words = re.findall(r"\w+", msg)

    score = 0
    i = 0
    while i < len(words):
        w = words[i]

        # Negation handling
        if w in NEGATION and i + 1 < len(words):
            nxt = words[i + 1]
            if nxt in POS: score -= 1.5; i += 2; continue
            if nxt in NEG or nxt in STRONG_NEG: score += 0.5; i += 2; continue

        if w in POS: score += 1
        if w in NEG: score -= 1
        if w in STRONG_NEG: score -= 2

        i += 1

    return score


def classify_emotion(text: str):
    score = sentiment_score(text)

    if score >= 1: return "happy", min(1.0, 0.7 + 0.1 * score)
    if score <= -1: return "angry", min(1.0, 0.7 + 0.1 * abs(score))

    return "neutral", max(0.3, 0.6 - 0.1 * abs(score))


# ============================================================
#  FAQ SYSTEM (Fixed – Now works correctly)
# ============================================================

FAQ_RULES = {
    "shipping": ["shipping", "delivery time", "how long", "arrive", "when will my order"],
    "refund": ["refund", "return", "money back", "cancel order", "cancellation"],
    "payment": ["payment", "upi", "card", "transaction", "failed", "double charged"],
    "account": ["login issue", "cant login", "account problem"],
}

FAQ_ANS = {
    "shipping": "🚚 *Shipping Info:* Most orders arrive in **3–5 days**. You can also type *track <order_id>*.",
    "refund": "💸 *Refund Policy:* Refunds available within **7 days** of delivery. Process time: 3–5 days.",
    "payment": "💳 *Payment Help:* Failed payments are usually auto-reversed in **3–7 days** by the bank.",
    "account": "🔐 *Account Help:* Try resetting your password using *Forgot Password* page.",
}

def match_faq(text: str):
    msg = text.lower()
    for key, kws in FAQ_RULES.items():
        for k in kws:
            if k in msg:
                return key
    return ""


# ============================================================
#  MEMORY SYSTEM
# ============================================================

class Memory:
    def __init__(self):
        self.data = defaultdict(lambda: {"context": deque(maxlen=20), "escalations": 0})

    def push(self, uid, speaker, text):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})


memory = Memory()


# ============================================================
#  ORDER SIMULATION
# ============================================================

STAGES = [
    "Order confirmed",
    "Packing",
    "Ready to ship",
    "Shipped",
    "In transit",
    "Out for delivery",
    "Delivered"
]

def simulate_order(oid):
    try:
        seed = int("".join(filter(str.isdigit, oid))) % 9999
    except:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(STAGES)-1)

    return {
        "order_id": oid,
        "stage": STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": STAGES[:idx+1]
    }


# ============================================================
#  AGENT ASSIST
# ============================================================

def frustration(ctx):
    score = 0
    for m in ctx:
        s = sentiment_score(m["text"])
        if s <= -2: score += 2
        elif s <= -0.5: score += 1
    return score

def risk(intent, emotion):
    if emotion == "angry": return "high"
    if intent == "escalate": return "medium"
    return "low"

def summary(ctx):
    last = " ".join(c["text"] for c in list(ctx)[-3:])
    return "Recent user messages: " + last if last else "No conversation yet."


def suggest(intent, emotion):
    if intent == "order": return "Reassure user and confirm order ID."
    if intent == "pricing": return "Offer a quick plan comparison."
    if intent == "support": return "Ask for details or screenshot."
    if emotion == "angry": return "Use empathy and offer agent escalation."
    return "Guide the user to the next step."


# ============================================================
#  MAIN ENGINE
# ============================================================

class AIEngine:
    def __init__(self):
        self.memory = memory

    def process(self, user_id, message):

        msg = message.strip()
        mem = self.memory.data[user_id]
        self.memory.push(user_id, "user", msg)

        # Safety
        if any(w in msg.lower() for w in ["kill","bomb","terror","suicide"]):
            return {"final_answer":"I cannot assist with that.","intent":"blocked","emotion":"neutral","confidence":1.0,"metadata":{}}

        # Emotion + Intent
        intent, _ = classify_intent(msg)
        emotion, _ = classify_emotion(msg)

        # Metadata generator
        def build(extra):
            ctx = mem["context"]
            return {
                **extra,
                "suggestion": suggest(intent, emotion),
                "frustration": frustration(ctx),
                "risk": risk(intent, emotion),
                "summary": summary(ctx),
                "m1": ctx[-1]["text"] if len(ctx)>0 else "",
                "m2": ctx[-2]["text"] if len(ctx)>1 else "",
                "m3": ctx[-3]["text"] if len(ctx)>2 else "",
            }

        # ------------------ FAQ (Fixed order) ------------------
        faq = match_faq(msg)
        if faq:
            return {
                "final_answer": FAQ_ANS[faq],
                "intent": "support",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": build({"faq_topic": faq})
            }

        # ------------------ ORDER ------------------
        if intent == "order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if m:
                oid = m.group(1)
                o = simulate_order(oid)
                return {
                    "final_answer": "order_status",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 1.0,
                    "metadata": build({
                        "order_id": oid,
                        "order_stage": o["stage"],
                        "eta_days": o["eta_days"],
                        "history": o["history"]
                    })
                }

        # ------------------ PRICING ------------------
        if intent == "pricing":
            plan = "basic"
            m = re.search(r"(basic|pro|enterprise)", msg, re.I)
            if m: plan = m.group(1).lower()
            price = {"basic":"₹499","pro":"₹1299","enterprise":"₹4999"}[plan]
            return {
                "final_answer": "pricing_info",
                "intent": "pricing",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": build({"plan":plan,"price":price})
            }

        # ------------------ GREETING ------------------
        if intent == "greeting":
            return {"final_answer":"Hi! How can I help you today?","intent":"greeting","emotion":emotion,"confidence":1.0,"metadata":{}}

        # ------------------ LOGIN ------------------
        if intent == "login":
            return {
                "final_answer":"Use 'Forgot Password' on login page. I can connect you to an agent if needed.",
                "intent":"login","emotion":emotion,"confidence":1.0,"metadata":{}
            }

        # ------------------ SUPPORT ------------------
        if intent == "support":
            return {
                "final_answer":"🔧 Please describe your issue.",
                "intent":"support","emotion":emotion,"confidence":0.9,"metadata":build({})
            }

        # ------------------ ESCALATE ------------------
        if intent == "escalate":
            mem["escalations"] += 1
            return {
                "final_answer":"Connecting you to a human agent...",
                "intent":"escalate","emotion":emotion,"confidence":1.0,"metadata":build({})
            }

        # ------------------ FALLBACK ------------------
        if emotion == "angry":
            ans = "I’m really sorry this has been frustrating. Tell me what went wrong — I can help or connect you to an agent."
        else:
            ans = "I couldn’t fully understand that. You can ask me to *track an order*, *show products*, or *help with shipping/refund/payment*."

        return {
            "final_answer": ans,
            "intent": "unknown",
            "emotion": emotion,
            "confidence": 0.3,
            "metadata": build({})
        }
