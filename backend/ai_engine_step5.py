# ai_engine_step5.py — Smart Hybrid AI Engine (Hackathon Winner Edition)
"""
Smart Hybrid AI Engine — Rule-based + Optional ML (HuggingFace) toggle.

Default mode (recommended for hackathon): USE_HF = False
Toggle USE_HF = true + set HUGGINGFACE_API_KEY to enable embeddings & HF emotion model.

Features:
- Intent classification (rules + keywords)
- Emotion detection (keywords + optional HF)
- Profanity & dangerous content filter
- Autoflow for missing fields (order_id, plan_type, product_name)
- Per-user memory (context + escalation count + expectations)
- Optional semantic KB (embeddings + cosine similarity) with caching
- Safe, predictable responses for demo
"""

import os
import re
import time
import json
import hashlib
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests

# -------------------------
# Configuration & Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_CSV = os.path.join(BASE_DIR, "data", "knowledge_base_Sheet1.csv")
EMBED_CACHE = os.path.join(BASE_DIR, "kb_embeddings.npy")

# Feature toggles (set USE_HF=true and HUGGINGFACE_API_KEY in Render to enable)
USE_HF = os.environ.get("USE_HF", "false").lower() == "true"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "").strip() if USE_HF else ""

# Models for HF (only used when USE_HF True)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"

# Thresholds
SIMILARITY_THRESHOLD = 0.55
AUTOFLOW_MIN_CONF = 0.50

# -------------------------
# Heuristics / Keywords
# -------------------------
GREETING_WORDS = {"hi", "hello", "hey", "hii", "hiya", "sup"}
CONFIRM_WORDS = {"yes", "ok", "okay", "sure", "yep", "yeah", "k", "correct"}
NEGATION_WORDS = {"no", "nah", "not", "don't", "dont", "nope"}

PROFANITY = {"fuck", "fucker", "fucking", "shit", "bitch", "bastard", "idiot", "stupid", "dumb", "asshole"}
BAD_WORDS = {"kill", "bomb", "suicide", "terror", "illegal", "hurt", "harm"}

INTENT_RULES = {
    "order": ["order", "track", "tracking", "delivery", "shipment", "order id", "ord"],
    "refund": ["refund", "return", "money back", "refund status", "claim"],
    "pricing": ["price", "pricing", "cost", "plan", "subscription", "how much", "rate"],
    "login": ["login", "signin", "forgot", "password", "reset password"],
    "support": ["help", "issue", "problem", "bug", "error", "not working", "crash"],
    "escalate": ["escalate", "agent", "human", "representative", "supervisor", "talk to agent"],
    "analytics": ["analytics", "stats", "reports", "dashboard"],
    "product": ["product", "availability", "stock", "in stock", "do you have", "available"]
}

# Emotion fallback keywords
EMO_KEYWORDS = {
    "angry": ["angry", "mad", "frustrated", "annoyed", "irritated", "pissed", "furious"],
    "sad": ["sad", "upset", "disappointed", "depressed", "unhappy", "sorrow"],
    "confused": ["confuse", "dont understand", "lost", "unclear", "how to", "kaise", "kaha"],
    "happy": ["thanks", "thank you", "great", "awesome", "happy", "glad", "nice", "cool"],
}

# -------------------------
# Memory class
# -------------------------
class Memory:
    def __init__(self, ctx_size: int = 20):
        self.data = defaultdict(lambda: {
            "escalations": 0,
            "expect": None,           # e.g., {"field": "order_id"}
            "last_intent": None,
            "context": deque(maxlen=ctx_size)
        })

    def get(self, uid: str) -> Dict[str, Any]:
        return self.data[uid]

    def push_context(self, uid: str, speaker: str, text: str):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})

    def clear(self, uid: Optional[str] = None):
        if uid:
            self.data.pop(uid, None)
        else:
            self.data.clear()

memory = Memory()

# -------------------------
# Utilities
# -------------------------
def load_kb(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path).fillna("").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=["Question", "Answer"])

def _safe_normalize(a: np.ndarray, axis=1, eps=1e-8) -> np.ndarray:
    norm = np.linalg.norm(a, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return a / norm

# -------------------------
# HuggingFace helpers (safe)
# -------------------------
def _hf_request_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """Return normalized embeddings or None on failure. Uses HF Inference API."""
    if not USE_HF or not HF_API_KEY:
        return None
    url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        # The HF inference for sentence-transformers accepts list of strings -> returns list of floats lists
        resp = requests.post(url, headers=headers, json={"inputs": texts}, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and isinstance(data[0], list):
            arr = np.asarray(data, dtype="float32")
            return _safe_normalize(arr, axis=1)
    except Exception:
        return None
    return None

def _hf_request_emotion(text: str) -> Optional[Tuple[str, float]]:
    """Returns (label, score) or None."""
    if not USE_HF or not HF_API_KEY:
        return None
    url = f"https://api-inference.huggingface.co/models/{EMOTION_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Expecting list of dicts like [{"label":"joy","score":0.9}, ...]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            preds = sorted(data, key=lambda x: x.get("score", 0), reverse=True)
            top = preds[0]
            return top.get("label", "").lower(), float(top.get("score", 0.0))
    except Exception:
        return None
    return None

# -------------------------
# Embedding caching and building
# -------------------------
def load_or_build_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Load cached embeddings or compute them (only when USE_HF True). If HF not enabled, return empty."""
    if df is None or len(df) == 0:
        return np.zeros((0, 0), dtype="float32")
    if not USE_HF:
        return np.zeros((0, 0), dtype="float32")
    # try cache
    if os.path.exists(EMBED_CACHE):
        try:
            emb = np.load(EMBED_CACHE)
            if emb.ndim == 2 and emb.shape[0] == len(df):
                return emb.astype("float32")
        except Exception:
            pass
    texts = df["Question"].astype(str).tolist()
    batch_size = 16
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = _hf_request_embeddings(batch)
        if emb is None:
            # Failure — return empty to indicate disabled embeddings
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

# -------------------------
# Safety filters
# -------------------------
def safety_filter(text: str) -> Optional[str]:
    """Return 'profanity', 'danger' or None"""
    t = (text or "").lower()
    # profanity
    if any(p in t for p in PROFANITY):
        return "profanity"
    # dangerous / violent keywords
    if any(b in t for b in BAD_WORDS) or any(phrase in t for phrase in ["kill myself", "kill me", "hurt someone"]):
        return "danger"
    return None

# -------------------------
# Small NLP helpers
# -------------------------
def _word_in_text(word: str, text: str) -> bool:
    try:
        return re.search(fr"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None
    except Exception:
        return word.lower() in text.lower()

def detect_order_id(text: str) -> Optional[str]:
    if not text:
        return None
    # Support ORD-12345 or plain numbers like 12345
    m = re.search(r"\bORD[-_ ]?(\d{3,12})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{3,12})\b", text)
    return m2.group(1) if m2 else None

def _clean_product_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'[^\w\s]', ' ', s.lower())
    fillers = ["kya", "hai", "milega", "phone", "mobile", "please", "is", "the", "do", "you", "have", "availability", "available", "stock", "check", "in"]
    for f in fillers:
        s = re.sub(fr"\b{re.escape(f)}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()

def extract_product_strict(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"is\s+(?:the\s+)?(.{1,80}?)\s+available",
        r"do you have\s+(?:the\s+)?(.{1,80}?)\??",
        r"availability\s+of\s+(?:the\s+)?(.{1,80}?)\??",
        r"stock\s+of\s+(?:the\s+)?(.{1,80}?)\??",
        r"price\s+of\s+(?:the\s+)?(.{1,80}?)\??"
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

# -------------------------
# Simulated helpers (demo)
# -------------------------
def simulated_inventory_check(product_name: str) -> Dict[str, Any]:
    h = int(hashlib.md5(product_name.encode()).hexdigest()[:8], 16)
    available = (h % 2 == 0)
    qty = (h % 20) + 1 if available else 0
    return {"available": available, "quantity": qty}

def simulated_price_lookup(product_name: str) -> str:
    h = int(hashlib.sha1(product_name.encode()).hexdigest()[:8], 16)
    price = 499 + (h % 2000)
    return f"₹{price}"

ORDER_STAGES = [
    "Order confirmed", "Packing", "Ready to ship", "Shipped",
    "In transit", "Out for delivery", "Delivered"
]

def simulate_order(oid: str):
    if not oid:
        oid = "0"
    try:
        seed = int(re.sub(r'\D', '', oid) or 0) % 9999
    except Exception:
        seed = sum(ord(c) for c in oid) % 9999
    np.random.seed(seed)
    idx = int(np.random.randint(0, len(ORDER_STAGES)))
    stage = ORDER_STAGES[idx]
    eta = max(0, 5 - idx)
    return {"order_id": oid, "stage": stage, "eta_days": eta, "status": "Delivered" if stage == "Delivered" else "In Progress", "history": ORDER_STAGES[:idx + 1]}

# -------------------------
# Topic graph and flows
# -------------------------
TOPIC_GRAPH = {
    "order": ["delivery", "tracking"],
    "product": ["pricing", "delivery"],
    "pricing": ["discounts", "product"],
    "support": ["login", "guide"],
}

def graph_reason_suggestion(last_intent: Optional[str], new_intent: Optional[str], message: str = "") -> Optional[str]:
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
    "product": ["product_name"]
}

FLOW_MESSAGES = {
    "order_id": "Please share your Order ID to continue.",
    "plan_type": "Which plan would you like? (Basic / Pro / Enterprise)",
    "product_name": "Which product are you asking about?"
}

# -------------------------
# AI Engine
# -------------------------
class AIEngine:
    def __init__(self, kb_csv: str = KB_CSV):
        self.df = load_kb(kb_csv)
        self.embeddings = load_or_build_embeddings(self.df)
        self.memory = memory

    def _semantic_search(self, q: str) -> Tuple[Optional[int], float]:
        """Return (idx, similarity) if embedding-based KB search possible, else (None, 0.0)."""
        if self.embeddings is None or getattr(self.embeddings, "size", 0) == 0:
            return None, 0.0
        emb = _hf_request_embeddings([q])
        if emb is None:
            return None, 0.0
        qv = emb[0].astype("float32")
        keys = self.embeddings.astype("float32")
        sims = np.dot(keys, qv)
        if sims.size == 0:
            return None, 0.0
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        # Clip score to 0..1 (dot product may vary)
        return idx, max(0.0, min(1.0, score))

    def _detect_missing_fields(self, intent: str, msg: str) -> List[str]:
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

    def process(self, user_id: str, message: str) -> Dict[str, Any]:
        start_t = time.time()
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
            "plan_type": None,
        }

        if not msg:
            return {
                "final_answer": "Please type your question.",
                "matched_question": None,
                "intent": "unknown",
                "emotion": "neutral",
                "confidence": 0.0,
                "risk": "low",
                "priority": "medium",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # Safety check
        safety = safety_filter(msg)
        if safety == "profanity":
            return {
                "final_answer": "I’m here to help — let’s keep the conversation respectful 😊",
                "matched_question": None,
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 0.0,
                "risk": "medium",
                "priority": "medium",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": {**base_meta, "safety": "profanity"}
            }
        if safety == "danger":
            return {
                "final_answer": "I can’t assist with that request. If this is an emergency, please contact local authorities.",
                "matched_question": None,
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 0.0,
                "risk": "high",
                "priority": "high",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": {**base_meta, "safety": "danger"}
            }

        # Intent + emotion
        intent, intent_conf = self._classify_intent(msg)
        emotion, emo_conf = self._classify_emotion(msg)

        # Graph hint
        kg_hint = graph_reason_suggestion(mem["last_intent"], intent, msg)
        if kg_hint:
            base_meta["hint"] = kg_hint

        # Handle expectation (autoflow)
        expect = mem["expect"]
        if expect:
            f = expect.get("field")
            # order id expected
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
            # plan type expected
            if f == "plan_type":
                m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
                if m:
                    plan = m.group(1).lower()
                    mem["expect"] = None
                    mem["last_intent"] = "pricing"
                    price = simulated_price_lookup(plan)
                    return {
                        "final_answer": f"The price for {plan.title()} is approx {price}. Want to proceed?",
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
            # product name expected
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

        # Ask for missing fields (autoflow)
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

        # Greeting
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

        # Escalate
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

        # Order handling
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

        # Pricing
        if intent == "pricing":
            prod = extract_product_strict(msg)
            if not prod:
                # fallback to simple plan detection
                m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
                if m:
                    prod = m.group(1).lower()
            if prod:
                price = simulated_price_lookup(prod)
                mem["last_intent"] = "pricing"
                return {
                    "final_answer": f"The price for \"{prod}\" is approx {price}. Need delivery info?",
                    "matched_question": None,
                    "intent": "pricing",
                    "emotion": emotion,
                    "confidence": max(0.6, intent_conf),
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

        # Login
        if intent == "login":
            mem["last_intent"] = "login"
            return {
                "final_answer": "You can reset using the ‘Forgot Password’ option on login page.",
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

        # Support
        if intent == "support":
            mem["last_intent"] = "support"
            return {
                "final_answer": "Could you share more details about the issue? (screenshots or error message helps)",
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

        # Analytics intent
        if intent == "analytics":
            mem["last_intent"] = "analytics"
            return {
                "final_answer": "You can view analytics at /analytics endpoint or open the analytics dashboard in the demo.",
                "matched_question": None,
                "intent": "analytics",
                "emotion": emotion,
                "confidence": 0.95,
                "risk": "low",
                "priority": "low",
                "missing_info": None,
                "escalations": mem["escalations"],
                "metadata": base_meta
            }

        # Confirm (if user replies 'yes' etc.) — continue last intent flow if possible
        if intent == "confirm":
            last = mem["last_intent"]
            if last == "order":
                mem["expect"] = {"field": "order_id"}
                return {
                    "final_answer": "Sure — please share the Order ID.",
                    "matched_question": None,
                    "intent": "confirm",
                    "emotion": emotion,
                    "confidence": 0.9,
                    "priority": "medium",
                    "metadata": base_meta
                }
            if last == "pricing":
                mem["expect"] = {"field": "plan_type"}
                return {
                    "final_answer": "Which plan? (Basic / Pro / Enterprise)",
                    "matched_question": None,
                    "intent": "confirm",
                    "emotion": emotion,
                    "confidence": 0.9,
                    "priority": "medium",
                    "metadata": base_meta
                }
            return {
                "final_answer": "Okay! What would you like to do next?",
                "matched_question": None,
                "intent": "confirm",
                "emotion": emotion,
                "confidence": 0.9,
                "priority": "low",
                "metadata": base_meta
            }

        # Semantic KB search (if available)
        idx, sim = self._semantic_search(msg)
        if idx is None or sim < SIMILARITY_THRESHOLD:
            mem["last_intent"] = "unknown"
            elapsed = time.time() - start_t
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
                "metadata": {**base_meta, "similarity": sim, "response_time": elapsed}
            }

        # KB answer
        row = self.df.iloc[idx]
        mem["last_intent"] = "kb"
        elapsed = time.time() - start_t
        return {
            "final_answer": row.get("Answer", ""),
            "matched_question": row.get("Question"),
            "intent": "kb",
            "emotion": emotion,
            "confidence": sim,
            "risk": "low" if sim > 0.6 else "medium",
            "priority": "low",
            "missing_info": None,
            "escalations": mem["escalations"],
            "metadata": {**base_meta, "similarity": sim, "response_time": elapsed}
        }

    # -------------------------
    # Internal helper wrappers for classification (keeps code organized)
    # -------------------------
    def _classify_intent(self, text: str) -> Tuple[str, float]:
        t = (text or "").lower().strip()
        if not t:
            return "unknown", 0.0
        # greeting
        for g in GREETING_WORDS:
            if t == g or t.startswith(g + " "):
                return "greeting", 1.0
        # confirm
        if any(t == c or t.startswith(c + " ") for c in CONFIRM_WORDS):
            return "confirm", 0.95
        # direct rules
        for intent, kws in INTENT_RULES.items():
            for k in kws:
                if _word_in_text(k, t):
                    return intent, 0.92
        # product triggers
        for p in INTENT_RULES.get("product", []):
            if p in t:
                return "product", 0.82
        # order id detection
        if detect_order_id(t):
            return "order", 0.95
        return "unknown", 0.35

    def _classify_emotion(self, text: str) -> Tuple[str, float]:
        t = (text or "").strip()
        if not t:
            return "neutral", 0.5
        # HF emotion if available
        if USE_HF and HF_API_KEY:
            res = _hf_request_emotion(t)
            if res:
                label, score = res
                if "anger" in label or "annoy" in label:
                    return "angry", score
                if "sad" in label:
                    return "sad", score
                if "confus" in label or "curio" in label:
                    return "confused", score
                if "joy" in label or "happy" in label:
                    return "happy", score
        # keyword fallback
        kw_scores = defaultdict(int)
        low = t.lower()
        for emo, kws in EMO_KEYWORDS.items():
            for k in kws:
                if k in low:
                    kw_scores[emo] += 1
        if kw_scores:
            top = max(kw_scores, key=kw_scores.get)
            return top, 0.8
        return "neutral", 0.5

# -------------------------
# CLI quick test
# -------------------------
if __name__ == "__main__":
    eng = AIEngine()
    print("Smart Hybrid AI Engine — CLI test. Type a message.")
    while True:
        try:
            t = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        out = eng.process("local", t)
        print(json.dumps(out, indent=2, ensure_ascii=False))
