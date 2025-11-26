# ai_engine_step5.py — FINAL AI Engine (HuggingFace embeddings, Render-friendly SAFE version)
"""
HuggingFace-backed embeddings variant of your AI engine.
Uses HF Inference API for embeddings (no local model, no FAISS).
Caches embeddings to kb_embeddings.npy.
Uses numpy cosine search for similarity.
Graceful fallback when HF API not configured.
"""

import os
import re
import hashlib
import json
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_CSV = os.path.join(BASE_DIR, "data", "knowledge_base_Sheet1.csv")

# HuggingFace model + token from environment variable
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "").strip()

# embeddings local cache file
EMBED_CACHE = os.path.join(BASE_DIR, "kb_embeddings.npy")

# thresholds
SIMILARITY_THRESHOLD = 0.55
AUTOFLOW_MIN_CONF = 0.5

# vocab / heuristics
GREETING_WORDS = {"hi", "hello", "hey", "hii", "heya", "sup", "yo"}
BAD_WORDS = {"kill", "bomb", "suicide", "terror", "hack", "illegal", "fuck", "bitch", "die"}

INTENT_RULES = {
    "order": ["order", "track", "tracking", "delivery", "shipment", "order id", "parcel"],
    "refund": ["refund", "return", "money back", "refund status", "varala", "refund varala"],
    "pricing": ["price", "pricing", "cost", "plan", "subscription", "how much", "rate", "entha"],
    "login": ["login", "signin", "forgot", "password", "reset"],
    "support": ["help", "issue", "problem", "bug", "error", "crash"],
    "escalate": ["escalate", "agent", "human", "representative", "supervisor", "talk to agent"],
}

PRODUCT_TRIGGERS = [
    "product", "availability", "in stock", "do you have", "available", "stock",
    "kya", "hai", "milega", "iruka", "unda", "stock la"
]

EMO_KEYWORDS = {
    "angry": ["angry", "mad", "frustrated", "annoyed", "gussa"],
    "sad": ["sad", "upset", "disappointed", "dukhi", "varala"],
    "confused": ["confuse", "dont understand", "how to", "kaise", "epdi"],
    "happy": ["thanks", "thank you", "great", "awesome", "shukriya"],
}


# memory system
class Memory:
    def __init__(self, ctx_size=10):
        self.data = defaultdict(lambda: {
            "escalations": 0,
            "expect": None,
            "last_intent": None,
            "context": deque(maxlen=ctx_size)
        })

    def get(self, uid):
        return self.data[uid]

    def inc_escal(self, uid):
        self.data[uid]["escalations"] += 1

    def set_expect(self, uid, expect):
        self.data[uid]["expect"] = expect

    def pop_expect(self, uid):
        e = self.data[uid]["expect"]
        self.data[uid]["expect"] = None
        return e

    def push_context(self, uid, speaker, text):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})

    def set_last_intent(self, uid, intent):
        self.data[uid]["last_intent"] = intent


memory = Memory()


# Load KB
def load_kb(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Question", "Answer"]).astype(str)
    try:
        return pd.read_csv(path).fillna("").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Question", "Answer"]).astype(str)


def _safe_normalize(a: np.ndarray, axis=1, eps=1e-8):
    norm = np.linalg.norm(a, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return a / norm


# HuggingFace API embedding request
def _hf_request_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    if not HF_API_KEY:
        return None

    url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    try:
        resp = requests.post(url, headers=headers, json={"inputs": texts}, timeout=25)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return None

        if isinstance(data, list) and isinstance(data[0], list):
            arr = np.asarray(data, dtype="float32")
            return _safe_normalize(arr, axis=1)

        return None

    except Exception:
        return None


# Load/build embeddings
def load_or_build_embeddings(df: pd.DataFrame) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros((0, 0), dtype="float32")

    # try load cache
    if os.path.exists(EMBED_CACHE):
        try:
            emb = np.load(EMBED_CACHE)
            if emb.ndim == 2 and emb.shape[0] == len(df):
                return emb.astype("float32")
        except Exception:
            pass

    # build embeddings in batches
    texts = df["Question"].astype(str).tolist()
    batch_size = 16
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = _hf_request_embeddings(batch)
        if emb is None:
            return np.zeros((0, 0), dtype="float32")
        all_embs.append(emb)

    if not all_embs:
        return np.zeros((0, 0), dtype="float32")

    emb = np.vstack(all_embs).astype("float32")

    try:
        np.save(EMBED_CACHE, emb)
    except Exception:
        pass

    return emb


# Utils -------------------------------------------------------------

def safety_filter(text: str) -> bool:
    return any(b in text.lower() for b in BAD_WORDS)


def classify_emotion(text: str) -> Tuple[str, float]:
    t = text.lower()
    scores = defaultdict(float)

    for emo, kws in EMO_KEYWORDS.items():
        for k in kws:
            if k in t:
                scores[emo] += 1

    if not scores:
        return "neutral", 0.5

    top = max(scores, key=scores.get)
    conf = scores[top] / (sum(scores.values()) + 1e-9)
    return top, min(1.0, max(0.5, conf))


def _word_in_text(word: str, text: str) -> bool:
    try:
        return re.search(fr"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None
    except:
        return word.lower() in text.lower()


def classify_intent(text: str) -> Tuple[str, float]:
    t = text.lower().strip()

    if not t:
        return "unknown", 0.0

    for g in GREETING_WORDS:
        if t == g or t.startswith(g + " "):
            return "greeting", 1.0

    # strong rules
    if any(_word_in_text(x, t) for x in ["refund", "refund status", "return", "money back", "varala"]):
        return "refund", 0.98

    if any(_word_in_text(x, t) for x in ["price", "pricing", "how much", "cost", "plan", "rate", "entha"]):
        return "pricing", 0.95

    if any(_word_in_text(x, t) for x in ["escalate", "agent", "human", "representative"]):
        return "escalate", 0.99

    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if _word_in_text(k, t):
                return intent, 0.90

    for trig in PRODUCT_TRIGGERS:
        if _word_in_text(trig, t):
            return "product", 0.85

    return "unknown", 0.30


def _clean_product_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'[^\w\s]', ' ', s.lower())
    fillers = [
        "kya", "hai", "milega", "phone", "mobile", "iruka", "unda", "la", "kaha",
        "please", "is", "the", "do", "you", "have", "availability", "available",
        "stock", "check", "in", "in stock", "stock?"
    ]
    for f in fillers:
        s = re.sub(fr"\b{re.escape(f)}\b", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def extract_product_strict(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()

    patterns = [
        r"is\s+(?:the\s+)?(.{1,60}?)\s+available",
        r"do you have\s+(?:the\s+)?(.{1,60}?)\??",
        r"availability\s+of\s+(?:the\s+)?(.{1,60}?)\??",
        r"stock\s+of\s+(?:the\s+)?(.{1,60}?)\??",
        r"price\s+of\s+(?:the\s+)?(.{1,60}?)\??"
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = _clean_product_name(m.group(1).strip())
            if cand and len(cand) > 1:
                return cand

    tokens = text.split()
    if 1 <= len(tokens) <= 8:
        cand = _clean_product_name(text)
        if cand.lower() not in {"yes", "no", "ok", "thanks"}:
            return cand

    return None


def detect_order_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\bORD[-_ ]?(\d{3,12})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{4,12})\b", text)
    return m2.group(1) if m2 else None


def risk_from(similarity: float, intent_conf: float) -> str:
    if intent_conf < 0.2 or similarity < 0.2:
        return "high"
    if similarity < 0.45 or intent_conf < 0.35:
        return "medium"
    return "low"


def simulated_inventory_check(product_name: str) -> Dict[str, Any]:
    h = int(hashlib.md5(product_name.encode()).hexdigest()[:8], 16)
    available = (h % 2 == 0)
    qty = (h % 20) + 1 if available else 0
    return {"available": available, "quantity": qty}


def simulated_price_lookup(product_name: str) -> str:
    h = int(hashlib.sha1(product_name.encode()).hexdigest()[:8], 16)
    price = 499 + (h % 2000)
    return f"₹{price}"


TOPIC_GRAPH = {
    "order": ["delivery", "tracking"],
    "product": ["pricing", "delivery"],
    "pricing": ["discounts", "product"],
    "support": ["login", "guide"],
}


def graph_reason_suggestion(last_intent: Optional[str], new_intent: Optional[str], message: str = ""):
    if not last_intent or not new_intent:
        return None
    if last_intent == "order" and "deliver" in message.lower():
        return f"Since you asked about {last_intent}, I can connect it to delivery. Continue?"
    if last_intent in TOPIC_GRAPH and new_intent in TOPIC_GRAPH[last_intent]:
        return f"Since you asked about {last_intent}, I can connect it to {new_intent}. Continue?"
    return None


FLOW_REQUIREMENTS = {
    "order": ["order_id"],
    "pricing": ["plan_type"],
    "protein": ["product_name"]
}

FLOW_MESSAGES = {
    "order_id": "Please share your Order ID to continue.",
    "plan_type": "Which plan would you like? (Basic / Pro / Enterprise)",
    "product_name": "Which product are you asking about?"
}


class AIEngine:
    def __init__(self, kb_csv: str = KB_CSV):
        self.df = load_kb(kb_csv)
        self.embeddings = load_or_build_embeddings(self.df)
        self.memory = memory

    def _semantic_search(self, q: str) -> Tuple[Optional[int], float]:
        if self.embeddings is None or self.embeddings.size == 0:
            return None, 0.0

        emb = _hf_request_embeddings([q])
        if emb is None:
            return None, 0.0

        qv = emb[0].astype("float32")
        keys = self.embeddings.astype("float32")

        sims = np.dot(keys, qv)
        idx = int(np.argmax(sims))
        score = float(sims[idx]) if sims.size > 0 else 0.0

        return idx, max(0.0, min(1.0, score))


    def _detect_missing_fields(self, intent: str, msg: str):
        missing = []
        if intent not in FLOW_REQUIREMENTS:
            return missing

        if intent == "order":
            if not detect_order_id(msg):
                missing.append("order_id")
        elif intent == "pricing":
            if not re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE):
                missing.append("plan_type")
        elif intent == "product":
            if not extract_product_strict(msg):
                missing.append("product_name")

        return missing


    # MAIN LOGIC ------------------------------------------------------
    def process(self, user_id: str, message: str) -> Dict[str, Any]:

        msg = (message or "").strip()
        mem = self.memory.get(user_id)

        self.memory.push_context(user_id, "user", msg)

        base_meta = {
            "confidence": 0.0,
            "similarity": 0.0,
            "risk": "low",
            "autoflow": False,
            "field_required": None,
            "hint": None,
            "order_id": None,
            "product_name": None,
            "plan_type": None
        }

        if not msg:
            return {
                "final_answer": "Please type your question.",
                "matched_question": None,
                "intent": "unknown",
                "emotion": "neutral",
                "confidence": 0,
                "risk": "low",
                "priority": "medium",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # SAFETY
        if safety_filter(msg):
            return {
                "final_answer": "I’m sorry — I can’t assist with that.",
                "matched_question": None,
                "intent": "unknown",
                "emotion": "neutral",
                "confidence": 0,
                "risk": "high",
                "priority": "high",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": {**base_meta, "risk": "high"}
            }

        # INTENT + EMOTION
        intent, intent_conf = classify_intent(msg)
        emotion, emo_conf = classify_emotion(msg)

        # KG HINT
        kg_hint = graph_reason_suggestion(mem["last_intent"], intent, msg)
        if kg_hint:
            base_meta["hint"] = kg_hint

        # EXPECTATION HANDLING
        expect = mem["expect"]
        if expect:
            f = expect["field"]

            if f == "order_id":
                oid = detect_order_id(msg)
                if oid:
                    mem["expect"] = None
                    mem["last_intent"] = "order"
                    base_meta["order_id"] = oid

                    return {
                        "final_answer": f"Tracking order {oid}… It’s being processed 🚚",
                        "matched_question": "Track order",
                        "intent": "order",
                        "emotion": emotion,
                        "confidence": 1.0,
                        "risk": "low",
                        "priority": "high",
                        "missing_info": None,
                        "escalations": mem["escalations"],
                        "metadata": {**base_meta, "confidence": 1.0}
                    }

            if f == "plan_type":
                m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
                if m:
                    plan = m.group(1)
                    mem["expect"] = None
                    mem["last_intent"] = "pricing"
                    price = simulated_price_lookup(plan)

                    return {
                        "final_answer": f"The price for {plan.title()} is approx {price}. Want to buy or check delivery?",
                        "matched_question": None,
                        "intent": "pricing",
                        "emotion": emotion,
                        "confidence": 1.0,
                        "risk": "low",
                        "priority": "medium",
                        "missing_info": None,
                        "escalations": mem["escalations"],
                        "metadata": {**base_meta, "plan_type": plan}
                    }

            if f == "product_name":
                prod = extract_product_strict(msg)
                if prod:
                    mem["expect"] = None
                    mem["last_intent"] = "product"

                    return {
                        "final_answer": f"Do you want to check availability for \"{prod}\"?",
                        "matched_question": None,
                        "intent": "product",
                        "emotion": emotion,
                        "confidence": 1.0,
                        "risk": "medium",
                        "priority": "medium",
                        "missing_info": None,
                        "escalations": mem["escalations"],
                        "metadata": {**base_meta, "product_name": prod}
                    }

        # AUTOFLOW
        missing = self._detect_missing_fields(intent, msg)
        if missing and intent_conf >= AUTOFLOW_MIN_CONF:
            field = missing[0]
            mem["expect"] = {"field": field}
            base_meta["autoflow"] = True
            base_meta["field_required"] = field

            return {
                "final_answer": FLOW_MESSAGES[field],
                "matched_question": None,
                "intent": intent,
                "emotion": emotion,
                "confidence": intent_conf,
                "risk": "low",
                "priority": "medium",
                "missing_info": field,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # GREETING
        if intent == "greeting":
            mem["last_intent"] = "greeting"

            return {
                "final_answer": "Hi! How can I help you today? 😊",
                "matched_question": None,
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "risk": "low",
                "priority": "low",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # ESCALATE
        if intent == "escalate":
            mem["escalations"] += 1
            mem["last_intent"] = "escalate"

            return {
                "final_answer": "Connecting you to a live agent…",
                "matched_question": None,
                "intent": "escalate",
                "emotion": emotion,
                "confidence": 1.0,
                "risk": "high",
                "priority": "high",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # ORDER
        order_id = detect_order_id(msg)
        if intent == "order" or order_id:
            if order_id:
                mem["last_intent"] = "order"
                base_meta["order_id"] = order_id

                return {
                    "final_answer": f"Tracking order {order_id}… 🚚 It’s on the way!",
                    "matched_question": "Track order",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 1.0,
                    "risk": "low",
                    "priority": "high",
                    "missing_info": None,
                    "escalations": mem["escalations"],
                    "metadata": base_meta
                }

            if intent_conf >= AUTOFLOW_MIN_CONF:
                mem["expect"] = {"field": "order_id"}
                return {
                    "final_answer": FLOW_MESSAGES["order_id"],
                    "matched_question": None,
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": intent_conf,
                    "risk": "low",
                    "priority": "medium",
                    "missing_info": "order_id",
                    "escalations": mem["escalations"],
                    "metadata": {**base_meta, "autoflow": True, "field_required": "order_id"}
                }

        # PRICING
        if intent == "pricing":
            prod = extract_product_strict(msg)
            if prod:
                price = simulated_price_lookup(prod)
                mem["last_intent"] = "pricing"

                return {
                    "final_answer": f"The price for \"{prod}\" is approx {price}. Want delivery details?",
                    "matched_question": None,
                    "intent": "pricing",
                    "emotion": emotion,
                    "confidence": intent_conf,
                    "risk": "low",
                    "priority": "medium",
                    "missing_info": None,
                    "escalations": mem["escalations"],
                    "metadata": {**base_meta, "product_name": prod}
                }

            if intent_conf >= AUTOFLOW_MIN_CONF:
                mem["expect"] = {"field": "plan_type"}

                return {
                    "final_answer": FLOW_MESSAGES["plan_type"],
                    "matched_question": None,
                    "intent": "pricing",
                    "emotion": emotion,
                    "confidence": intent_conf,
                    "risk": "low",
                    "priority": "medium",
                    "missing_info": "plan_type",
                    "escalations": mem["escalations"],
                    "metadata": {**base_meta, "autoflow": True, "field_required": "plan_type"}
                }

        # LOGIN
        if intent == "login":
            mem["last_intent"] = "login"

            return {
                "final_answer": "You can reset using the ‘Forgot Password’ option.",
                "matched_question": None,
                "intent": "login",
                "emotion": emotion,
                "confidence": intent_conf,
                "risk": "low",
                "priority": "low",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # SUPPORT
        if intent == "support":
            mem["last_intent"] = "support"

            return {
                "final_answer": "Could you share more details about the issue?",
                "matched_question": None,
                "intent": "support",
                "emotion": emotion,
                "confidence": intent_conf,
                "risk": "medium",
                "priority": "medium",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # SEMANTIC KB SEARCH
        idx, sim = self._semantic_search(msg)
        if idx is None or sim < SIMILARITY_THRESHOLD:
            mem["last_intent"] = "unknown"

            return {
                "final_answer": "I couldn't find an exact match. Can you clarify a bit more?",
                "matched_question": None,
                "intent": "unknown",
                "emotion": emotion,
                "confidence": intent_conf,
                "risk": "medium",
                "priority": "medium",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": {**base_meta, "similarity": sim}
            }

        row = self.df.iloc[idx]
        mem["last_intent"] = "kb"

        return {
            "final_answer": row.get("Answer", ""),
            "matched_question": row.get("Question"),
            "intent": "kb",
            "emotion": emotion,
            "confidence": sim,
            "risk": risk_from(sim, intent_conf),
            "priority": "low",
            "missing_info": None,
            "escalations": mem["escalations"],
            "metadata": {**base_meta, "similarity": sim}
        }


# CLI TEST MODE
if __name__ == "__main__":
    eng = AIEngine()
    while True:
        text = input("You: ")
        print(json.dumps(eng.process("local", text), indent=2))
