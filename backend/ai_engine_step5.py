# ai_engine_step5.py — FINAL CLEAN AI ENGINE
"""
AI Engine with:
- HuggingFace API embeddings (Render-friendly)
- Knowledge Base matching
- Intent detection
- Emotion detection
- Autoflow (order, pricing, product)
- Safety filtering
- Memory per user
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

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_CSV = os.path.join(BASE_DIR, "data", "knowledge_base_Sheet1.csv")
EMBED_CACHE = os.path.join(BASE_DIR, "kb_embeddings.npy")

# HuggingFace
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "").strip()

# Thresholds
SIMILARITY_THRESHOLD = 0.55
AUTOFLOW_MIN_CONF = 0.50

# Keywords
GREETING_WORDS = {"hi", "hello", "hey", "hii"}
BAD_WORDS = {"kill", "bomb", "suicide", "terror", "hack", "illegal", "fuck", "bitch", "die"}

INTENT_RULES = {
    "order": ["order", "track", "tracking", "delivery", "shipment"],
    "refund": ["refund", "return", "money back"],
    "pricing": ["price", "pricing", "cost", "plan", "subscription"],
    "login": ["login", "signin", "forgot", "password"],
    "support": ["help", "issue", "problem", "error"],
    "escalate": ["escalate", "agent", "human"]
}

PRODUCT_TRIGGERS = ["product", "availability", "stock", "in stock"]

EMO_KEYWORDS = {
    "angry": ["angry", "mad", "frustrated"],
    "sad": ["sad", "upset", "disappointed"],
    "confused": ["confuse", "dont understand"],
    "happy": ["thanks", "thank you", "great"]
}


# ------------------------------
# Memory Class
# ------------------------------
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

    def push_context(self, uid, speaker, text):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})


memory = Memory()


# ------------------------------
# Load KB
# ------------------------------
def load_kb(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path).fillna("").reset_index(drop=True)
    except:
        pass
    return pd.DataFrame(columns=["Question", "Answer"])


# ------------------------------
# Embeddings (HuggingFace API)
# ------------------------------
def _safe_normalize(arr, axis=1):
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norm, 1e-8)


def _hf_request_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    if not HF_API_KEY:
        return None

    url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        res = requests.post(url, headers=headers, json={"inputs": texts}, timeout=25)
        if res.status_code != 200:
            return None
        data = res.json()
        if isinstance(data, list):
            arr = np.asarray(data, dtype="float32")
            return _safe_normalize(arr)
        return None
    except:
        return None


def load_or_build_embeddings(df: pd.DataFrame):
    if len(df) == 0:
        return np.zeros((0, 0), dtype="float32")

    # Try cache
    if os.path.exists(EMBED_CACHE):
        try:
            emb = np.load(EMBED_CACHE)
            if emb.shape[0] == len(df):
                return emb.astype("float32")
        except:
            pass

    # Build new embeddings
    all_vecs = []
    texts = df["Question"].tolist()
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        emb = _hf_request_embeddings(texts[i:i+batch_size])
        if emb is None:
            return np.zeros((0, 0), dtype="float32")
        all_vecs.append(emb)

    final_emb = np.vstack(all_vecs)
    try:
        np.save(EMBED_CACHE, final_emb)
    except:
        pass

    return final_emb


# ------------------------------
# Utility Functions
# ------------------------------
def safety_filter(text: str):
    t = text.lower()
    return any(bad in t for bad in BAD_WORDS)


def classify_emotion(text: str):
    t = text.lower()
    scores = defaultdict(int)
    for emo, words in EMO_KEYWORDS.items():
        for w in words:
            if w in t:
                scores[emo] += 1
    if not scores:
        return "neutral", 0.5
    top = max(scores, key=scores.get)
    return top, 1.0


def _word_in_text(word, text):
    try:
        return re.search(fr"\b{re.escape(word)}\b", text, flags=re.I)
    except:
        return word in text


def classify_intent(text: str):
    t = text.lower().strip()
    if t in GREETING_WORDS:
        return "greeting", 1.0

    for intent, kws in INTENT_RULES.items():
        for k in kws:
            if _word_in_text(k, t):
                return intent, 0.9

    for k in PRODUCT_TRIGGERS:
        if k in t:
            return "product", 0.8

    return "unknown", 0.3


def detect_order_id(text: str):
    m = re.search(r"\b(\d{4,12})\b", text)
    return m.group(1) if m else None


def extract_product(text: str):
    t = text.strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    if len(t.split()) <= 8:
        return t.title()
    return None


def simulated_price_lookup(product):
    h = int(hashlib.md5(product.encode()).hexdigest()[:8], 16)
    return f"₹{500 + (h % 2000)}"


# ------------------------------
# AI Engine Class
# ------------------------------
class AIEngine:
    def __init__(self, kb_csv=KB_CSV):
        self.df = load_kb(kb_csv)
        self.embeddings = load_or_build_embeddings(self.df)
        self.memory = memory

    def _semantic_search(self, q):
        if self.embeddings is None or self.embeddings.size == 0:
            return None, 0.0

        q_emb = _hf_request_embeddings([q])
        if q_emb is None:
            return None, 0.0

        qv = q_emb[0]
        sims = np.dot(self.embeddings, qv)
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    # -----------------------------------------
    # Main Process Function
    # -----------------------------------------
    def process(self, user_id: str, message: str) -> Dict[str, Any]:
        msg = (message or "").strip()
        mem = self.memory.get(user_id)

        # Context tracking
        self.memory.push_context(user_id, "user", msg)

        # Safety
        if safety_filter(msg):
            return {
                "final_answer": "I’m sorry — I cannot assist with that.",
                "intent": "blocked",
                "emotion": "neutral",
                "priority": "high",
                "metadata": {}
            }

        # Intent + Emotion
        intent, intent_conf = classify_intent(msg)
        emotion, emo_conf = classify_emotion(msg)

        # Greeting
        if intent == "greeting":
            return {
                "final_answer": "Hi! How can I help you today? 😊",
                "intent": "greeting",
                "emotion": emotion,
                "priority": "low",
                "metadata": {}
            }

        # Order
        oid = detect_order_id(msg)
        if intent == "order" or oid:
            if oid:
                return {
                    "final_answer": f"Tracking order {oid}… 🚚",
                    "intent": "order",
                    "emotion": emotion,
                    "priority": "high",
                    "metadata": {"order_id": oid}
                }
            return {
                "final_answer": "Please share your order ID.",
                "intent": "order",
                "emotion": emotion,
                "priority": "medium",
                "metadata": {}
            }

        # Pricing
        if intent == "pricing":
            prod = extract_product(msg)
            if prod:
                price = simulated_price_lookup(prod)
                return {
                    "final_answer": f"The price for {prod} is approx {price}.",
                    "intent": "pricing",
                    "emotion": emotion,
                    "priority": "medium",
                    "metadata": {"product_name": prod}
                }
            return {
                "final_answer": "Which product pricing would you like?",
                "intent": "pricing",
                "emotion": emotion,
                "priority": "medium",
                "metadata": {}
            }

        # Semantic KB Search
        idx, sim = self._semantic_search(msg)
        if idx is not None and sim >= SIMILARITY_THRESHOLD:
            row = self.df.iloc[idx]
            return {
                "final_answer": row["Answer"],
                "intent": "kb",
                "emotion": emotion,
                "priority": "low",
                "metadata": {"similarity": sim}
            }

        # Fallback
        return {
            "final_answer": "I couldn’t find an exact match. Can you clarify?",
            "intent": "unknown",
            "emotion": emotion,
            "priority": "medium",
            "metadata": {}
        }
