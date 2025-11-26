# ai_engine_step5.py — FINAL AI Engine (final fixes)
"""
Final engine for hackathon:
- Autoflow isolation with confidence threshold
- Expectation handling validates user replies
- KG hint: delivery-aware when order context exists
- Stronger product-memory behavior (avoid noisy KB matches when user meant product)
- Product extraction cleaned (removes trailing 'in', 'in stock' noise)
- Risk logic fixed (safe answers -> low)
- Metadata always present and consistent
"""

import os
import re
import hashlib
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# optional heavy deps (engine still works without them)
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    faiss = None
    SentenceTransformer = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_CSV = os.path.join(BASE_DIR, "data", "knowledge_base_Sheet1.csv")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
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

PRODUCT_TRIGGERS = ["product", "availability", "in stock", "do you have", "available", "stock", "kya", "hai", "milega", "iruka", "unda", "stock la"]

EMO_KEYWORDS = {
    "angry": ["angry", "mad", "frustrated", "annoyed", "gussa"],
    "sad": ["sad", "upset", "disappointed", "dukhi", "varala"],
    "confused": ["confuse", "dont understand", "how to", "kaise", "epdi"],
    "happy": ["thanks", "thank you", "great", "awesome", "shukriya"],
}

# memory
class Memory:
    def __init__(self, ctx_size=10):
        self.data = defaultdict(lambda: {"escalations": 0, "expect": None, "last_intent": None, "context": deque(maxlen=ctx_size)})

    def get(self, uid):
        return self.data[uid]

    def inc_escal(self, uid):
        self.data[uid]["escalations"] += 1

    def set_expect(self, uid, expect):
        self.data[uid]["expect"] = expect

    def pop_expect(self, uid):
        e = self.data[uid].get("expect")
        self.data[uid]["expect"] = None
        return e

    def push_context(self, uid, speaker, text):
        self.data[uid]["context"].append({"speaker": speaker, "text": text})

    def set_last_intent(self, uid, intent):
        self.data[uid]["last_intent"] = intent

memory = Memory()

# KB + embeddings loaders (graceful)
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

def load_or_build_embeddings(df: pd.DataFrame, model) -> np.ndarray:
    if model is None or df is None or len(df) == 0:
        return np.zeros((0, 0), dtype="float32")
    if os.path.exists(EMBED_CACHE):
        try:
            emb = np.load(EMBED_CACHE)
            if emb.ndim == 2 and emb.shape[0] == len(df):
                return emb.astype("float32")
        except Exception:
            pass
    texts = df["Question"].astype(str).tolist()
    raw = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    raw = np.asarray(raw, dtype="float32")
    emb = _safe_normalize(raw, axis=1)
    try:
        np.save(EMBED_CACHE, emb)
    except Exception:
        pass
    return emb

# utils
def safety_filter(text: str) -> bool:
    t = (text or "").lower()
    return any(b in t for b in BAD_WORDS)

def classify_emotion(text: str) -> Tuple[str, float]:
    t = (text or "").lower()
    scores = defaultdict(float)
    for emo, kws in EMO_KEYWORDS.items():
        for k in kws:
            if k in t:
                scores[emo] += 1.0
    if not scores:
        return "neutral", 0.5
    top = max(scores, key=scores.get)
    conf = scores[top] / (sum(scores.values()) + 1e-9)
    return top, float(max(0.5, min(1.0, conf)))

def _word_in_text(word: str, text: str) -> bool:
    try:
        return re.search(r"\b" + re.escape(word) + r"\b", text, flags=re.IGNORECASE) is not None
    except re.error:
        return word.lower() in text.lower()

def classify_intent(text: str) -> Tuple[str, float]:
    t = (text or "").lower().strip()
    if not t:
        return "unknown", 0.0
    for g in GREETING_WORDS:
        if t == g or t.startswith(g + " ") or t.startswith(g + "!") or t.startswith(g + ","):
            return "greeting", 1.0
    # high priority
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
    # remove punctuation and common fillers; remove trailing 'in' or 'in stock' leftovers
    s = re.sub(r'[\?\!\(\)\.\"]', ' ', s)
    s = s.lower()
    fillers = ["kya", "hai", "milega", "phone", "mobile", "iruka", "unda", "la", "kaha", "please",
               "is", "the", "do", "you", "have", "availability", "available", "stock", "check", "in", "in stock", "stock?"]
    for f in fillers:
        s = re.sub(r'\b' + re.escape(f) + r'\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s+in\s*$', '', s)  # strip any trailing 'in'
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
        r"price\s+of\s+(?:the\s+)?(.{1,60}?)\??",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r'^(the|a|an)\s+', '', cand, flags=re.IGNORECASE).strip()
            cand = _clean_product_name(cand)
            if cand and len(cand) > 1:
                return cand
    tokens = text.split()
    if 1 <= len(tokens) <= 8:
        cand = _clean_product_name(text)
        if cand and cand.lower() not in {"yes", "no", "ok", "thanks", "refund", "order", "price", "tracking"}:
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
    # Simplified: treat normal replies as low risk unless similarity/intent low or safety flagged
    s = float(similarity)
    ic = float(intent_conf)
    if ic < 0.2 or s < 0.2:
        return "high"
    if s < 0.45 or ic < 0.35:
        return "medium"
    return "low"

def simulated_inventory_check(product_name: str) -> Dict[str, Any]:
    h = int(hashlib.md5(product_name.encode("utf-8")).hexdigest()[:8], 16)
    available = (h % 2 == 0)
    qty = (h % 20) + 1 if available else 0
    return {"available": available, "quantity": int(qty)}

def simulated_price_lookup(product_name: str) -> str:
    h = int(hashlib.sha1(product_name.encode("utf-8")).hexdigest()[:8], 16)
    price = 499 + (h % 2000)
    return f"₹{price}"

# KG graph
TOPIC_GRAPH = {
    "order": ["delivery", "tracking"],
    "product": ["pricing", "delivery"],
    "pricing": ["discounts", "product"],
    "support": ["login", "guide"],
}

def graph_reason_suggestion(last_intent: Optional[str], new_intent: Optional[str], message: str = "") -> Optional[str]:
    if not last_intent or not new_intent:
        return None
    # special-case: if last_intent was 'order' and message asks about delivery, prefer hint
    if last_intent == "order" and "deliver" in message.lower():
        return f"Since you asked about {last_intent} earlier, I can connect it to delivery. Want me to continue?"
    if last_intent in TOPIC_GRAPH and new_intent in TOPIC_GRAPH[last_intent]:
        return f"Since you asked about {last_intent} earlier, I can connect it to {new_intent}. Want me to continue?"
    return None

FLOW_REQUIREMENTS = {
    "order": ["order_id"],
    "pricing": ["plan_type"],
    "product": ["product_name"],
}

FLOW_MESSAGES = {
    "order_id": "Please share your Order ID so I can continue tracking.",
    "plan_type": "Which plan would you like to know about? (Basic / Pro / Enterprise)",
    "product_name": "Which product would you like to check?",
}

class AIEngine:
    def __init__(self, kb_csv: str = KB_CSV, embed_model: str = EMBED_MODEL):
        self.df = load_kb(kb_csv)
        self.model = None
        self.embeddings = np.zeros((0, 0), dtype="float32")
        self.index = None
        if SentenceTransformer is not None and faiss is not None and len(self.df) > 0:
            try:
                self.model = SentenceTransformer(embed_model)
                self.embeddings = load_or_build_embeddings(self.df, self.model)
                if self.embeddings.size != 0:
                    dim = int(self.embeddings.shape[1])
                    self.index = faiss.IndexFlatIP(dim)
                    self.index.add(self.embeddings.astype("float32"))
            except Exception:
                self.model = None
                self.index = None
        self.memory = memory

    def _semantic_search(self, q: str) -> Tuple[Optional[int], float]:
        if self.model is None or self.index is None or self.embeddings.size == 0:
            return None, 0.0
        try:
            vec = self.model.encode([q], convert_to_numpy=True)
            vec = np.asarray(vec, dtype="float32")
            vec = _safe_normalize(vec, axis=1)
            D, I = self.index.search(vec, 1)
            score = float(D[0][0]) if D is not None else 0.0
            score = max(0.0, min(1.0, score))
            idx = int(I[0][0]) if I is not None and I[0][0] >= 0 else None
            return idx, score
        except Exception:
            return None, 0.0

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
            "plan_type": None,
            "availability": None
        }

        if not msg:
            self.memory.set_last_intent(user_id, "unknown")
            return {
                "final_answer": "Please type your question.",
                "matched_question": None,
                "intent": "unknown",
                "emotion": "neutral",
                "confidence": 0.0,
                "risk": "low",
                "priority": "high",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": base_meta
            }

        # safety first
        if safety_filter(msg):
            self.memory.set_last_intent(user_id, "unknown")
            return {
                "final_answer": "I'm sorry — I can't assist with that request.",
                "matched_question": None,
                "intent": "unknown",
                "emotion": "neutral",
                "confidence": 0.0,
                "risk": "high",
                "priority": "high",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": {**base_meta, "risk": "high"}
            }

        # classify & emotion
        intent, intent_conf = classify_intent(msg)
        emotion, emo_conf = classify_emotion(msg)

        # KG hint (improved)
        kg_hint = graph_reason_suggestion(mem.get("last_intent"), intent, msg)
        if kg_hint:
            base_meta["hint"] = kg_hint

        # validate expectation replies first (do not hijack unrelated inputs)
        expect = mem.get("expect")
        if expect:
            etype = expect.get("type")
            if etype == "need_field":
                field = expect["field"]
                if field == "order_id":
                    val = detect_order_id(msg)
                    if val:
                        mem["context"].append({"field": field, "value": val})
                        self.memory.pop_expect(user_id)
                        self.memory.set_last_intent(user_id, "order")
                        base_meta["order_id"] = val
                        return {
                            "final_answer": f"Sure! Tracking order {val}... Your order is being processed. 🚚",
                            "matched_question": "Track order",
                            "intent": "order",
                            "emotion": emotion,
                            "confidence": 1.0,
                            "risk": "low",
                            "priority": "high",
                            "missing_info": None,
                            "escalations": int(mem["escalations"]),
                            "metadata": {**base_meta, "confidence": 1.0}
                        }
                if field == "plan_type":
                    m = re.search(r"\b(basic|pro|enterprise)\b", msg, flags=re.IGNORECASE)
                    if m:
                        plan = m.group(1)
                        mem["context"].append({"field": field, "value": plan})
                        self.memory.pop_expect(user_id)
                        self.memory.set_last_intent(user_id, "pricing")
                        price = simulated_price_lookup(plan)
                        base_meta["plan_type"] = plan
                        return {
                            "final_answer": f"The price for \"{plan.title()}\" is approximately {price}. Would you like to check delivery or buy?",
                            "matched_question": None,
                            "intent": "pricing",
                            "emotion": emotion,
                            "confidence": 1.0,
                            "risk": "low",
                            "priority": "medium",
                            "missing_info": None,
                            "escalations": int(mem["escalations"]),
                            "metadata": {**base_meta, "confidence": 1.0}
                        }
                if field == "product_name":
                    prod = extract_product_strict(msg)
                    if prod:
                        mem["context"].append({"field": field, "value": prod})
                        self.memory.pop_expect(user_id)
                        self.memory.set_last_intent(user_id, "product")
                        return {
                            "final_answer": f'Do you want to check availability for \"{prod}\"?',
                            "matched_question": None,
                            "intent": "product",
                            "emotion": emotion,
                            "confidence": 1.0,
                            "risk": "medium",
                            "priority": "medium",
                            "missing_info": None,
                            "escalations": int(mem["escalations"]),
                            "metadata": {**base_meta, "product_name": prod, "confidence": 1.0}
                        }

        # detect missing fields (autoflow) only when intent confident
        missing_fields = self._detect_missing_fields(intent, msg)
        if missing_fields and intent_conf >= AUTOFLOW_MIN_CONF:
            field = missing_fields[0]
            self.memory.set_expect(user_id, {"type": "need_field", "field": field, "intent": intent})
            base_meta.update({"autoflow": True, "field_required": field, "hint": base_meta.get("hint")})
            return {
                "final_answer": FLOW_MESSAGES.get(field, "Please provide the required information."),
                "matched_question": None,
                "intent": intent,
                "emotion": emotion,
                "confidence": float(intent_conf),
                "risk": "low",
                "priority": "medium",
                "missing_info": field,
                "escalations": int(mem["escalations"]),
                "metadata": base_meta
            }

        # greeting
        if intent == "greeting":
            self.memory.set_last_intent(user_id, "greeting")
            return {
                "final_answer": "Hi! How can I help you today? 😊",
                "matched_question": None,
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "risk": "low",
                "priority": "low",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": {**base_meta, "confidence": 1.0, "similarity": 1.0}
            }

        # escalate
        if intent == "escalate":
            self.memory.inc_escal(user_id)
            self.memory.set_last_intent(user_id, "escalate")
            return {
                "final_answer": "Sure — connecting you to a live agent now.",
                "matched_question": None,
                "intent": "escalate",
                "emotion": emotion,
                "confidence": 1.0,
                "risk": "high",
                "priority": "high",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": {**base_meta, "confidence": 1.0, "similarity": 1.0, "risk": "high"}
            }

        # order handling
        order_id = detect_order_id(msg)
        if intent == "order" or order_id:
            if order_id:
                self.memory.set_last_intent(user_id, "order")
                base_meta["order_id"] = order_id
                meta = {**base_meta, "confidence": float(intent_conf) if intent == "order" else 1.0, "order_id": order_id}
                meta["risk"] = risk_from(meta.get("similarity", 0.0), meta.get("confidence", 1.0))
                return {
                    "final_answer": f"Sure! Tracking order {order_id}... Your order is being processed. 🚚",
                    "matched_question": "Track order",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": float(intent_conf) if intent == "order" else 1.0,
                    "risk": meta["risk"],
                    "priority": "high",
                    "missing_info": None,
                    "escalations": int(mem["escalations"]),
                    "metadata": meta
                }
            else:
                if intent_conf >= AUTOFLOW_MIN_CONF:
                    # order autoflow
                    self.memory.set_last_intent(user_id, "order")
                    self.memory.set_expect(user_id, {"type": "need_field", "field": "order_id", "intent": "order"})
                    meta = {**base_meta, "autoflow": True, "field_required": "order_id"}
                    meta["risk"] = "low"
                    return {
                        "final_answer": FLOW_MESSAGES["order_id"],
                        "matched_question": None,
                        "intent": "order",
                        "emotion": emotion,
                        "confidence": float(intent_conf),
                        "risk": meta["risk"],
                        "priority": "medium",
                        "missing_info": "order_id",
                        "escalations": int(mem["escalations"]),
                        "metadata": meta
                    }

        # pricing handling
        if intent == "pricing":
            prod = extract_product_strict(msg)
            if prod:
                price = simulated_price_lookup(prod)
                self.memory.set_last_intent(user_id, "pricing")
                meta = {**base_meta, "confidence": float(intent_conf), "product_name": prod}
                meta["risk"] = risk_from(meta.get("similarity", 0.0), float(intent_conf))
                return {
                    "final_answer": f"The price for \"{prod}\" is approximately {price}. Would you like to check delivery or buy?",
                    "matched_question": None,
                    "intent": "pricing",
                    "emotion": emotion,
                    "confidence": float(intent_conf),
                    "risk": meta["risk"],
                    "priority": "medium",
                    "missing_info": None,
                    "escalations": int(mem["escalations"]),
                    "metadata": meta
                }
            if intent_conf >= AUTOFLOW_MIN_CONF:
                self.memory.set_last_intent(user_id, "pricing")
                self.memory.set_expect(user_id, {"type": "need_field", "field": "plan_type", "intent": "pricing"})
                meta = {**base_meta, "autoflow": True, "field_required": "plan_type"}
                meta["risk"] = "low"
                return {
                    "final_answer": FLOW_MESSAGES["plan_type"],
                    "matched_question": None,
                    "intent": "pricing",
                    "emotion": emotion,
                    "confidence": float(intent_conf),
                    "risk": meta["risk"],
                    "priority": "medium",
                    "missing_info": "plan_type",
                    "escalations": int(mem["escalations"]),
                    "metadata": meta
                }

        # login/support
        if intent == "login":
            self.memory.set_last_intent(user_id, "login")
            meta = {**base_meta, "confidence": float(intent_conf)}
            meta["risk"] = "low"
            return {
                "final_answer": "No worries — you can reset it using the 'Forgot Password' option on the login page.",
                "matched_question": None,
                "intent": "login",
                "emotion": emotion,
                "confidence": float(intent_conf),
                "risk": "low",
                "priority": "low",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": meta
            }

        if intent == "support":
            self.memory.set_last_intent(user_id, "support")
            meta = {**base_meta, "confidence": float(intent_conf)}
            meta["risk"] = "medium"
            return {
                "final_answer": "I couldn't find an exact match in the knowledge base. Could you provide more details about the issue?",
                "matched_question": None,
                "intent": "support",
                "emotion": emotion,
                "confidence": float(intent_conf),
                "risk": meta["risk"],
                "priority": "medium",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": meta
            }

        # product flows — PRODUCT MEMORY BEHAVIOR FIX:
        # If last_intent is product, prefer product flow/confirm rather than unrelated KB rows.
        last_intent = mem.get("last_intent")
        prod_candidate = extract_product_strict(msg)
        if last_intent == "product" and prod_candidate:
            # treat as product followup
            self.memory.set_last_intent(user_id, "product")
            # semantic search only if embeddings available; otherwise ask confirm
            idx, sim = self._semantic_search(prod_candidate)
            if idx is not None and sim >= SIMILARITY_THRESHOLD:
                kb_ans = self.df.iloc[idx].get("Answer", "")
                meta = {**base_meta, "confidence": float(intent_conf), "similarity": float(sim), "product_name": prod_candidate}
                meta["risk"] = risk_from(float(sim), float(intent_conf))
                return {
                    "final_answer": kb_ans or "I couldn't find an exact match in the knowledge base.",
                    "matched_question": self.df.iloc[idx].get("Question"),
                    "intent": "product",
                    "emotion": emotion,
                    "confidence": float(intent_conf),
                    "risk": meta["risk"],
                    "priority": "medium",
                    "missing_info": None,
                    "escalations": int(mem["escalations"]),
                    "metadata": meta
                }
            # weak semantic match -> ask confirm
            self.memory.set_expect(user_id, {"type": "confirm_product", "product": prod_candidate})
            return {
                "final_answer": f'Do you want to check availability for "{prod_candidate}"?',
                "matched_question": None,
                "intent": "product",
                "emotion": emotion,
                "confidence": 0.9,
                "risk": "medium",
                "priority": "medium",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": {**base_meta, "confidence": 0.9, "product_name": prod_candidate}
            }

        # regular product handling
        if prod_candidate:
            idx, sim = self._semantic_search(prod_candidate)
            if idx is not None and sim >= SIMILARITY_THRESHOLD:
                kb_ans = self.df.iloc[idx].get("Answer", "")
                self.memory.set_last_intent(user_id, "product")
                meta = {**base_meta, "confidence": float(intent_conf), "similarity": float(sim), "product_name": prod_candidate}
                meta["risk"] = risk_from(float(sim), float(intent_conf))
                return {
                    "final_answer": kb_ans or "I couldn't find an exact match in the knowledge base.",
                    "matched_question": self.df.iloc[idx].get("Question"),
                    "intent": "product",
                    "emotion": emotion,
                    "confidence": float(intent_conf),
                    "risk": meta["risk"],
                    "priority": "medium",
                    "missing_info": None,
                    "escalations": int(mem["escalations"]),
                    "metadata": meta
                }
            # ask confirm
            self.memory.set_expect(user_id, {"type": "confirm_product", "product": prod_candidate})
            self.memory.set_last_intent(user_id, "product")
            return {
                "final_answer": f'Do you want to check availability for "{prod_candidate}"?',
                "matched_question": None,
                "intent": "product",
                "emotion": emotion,
                "confidence": 0.9,
                "risk": "medium",
                "priority": "medium",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": {**base_meta, "confidence": 0.9, "product_name": prod_candidate}
            }

        # explicit availability without product
        lower = msg.lower()
        if ("product availability" in lower or lower.strip() == "product" or "availability" in lower or any(x in lower for x in ["stock la", "iruka", "unda"])):
            self.memory.set_expect(user_id, {"type": "await_product"})
            self.memory.set_last_intent(user_id, "product")
            return {
                "final_answer": "Sure — which product would you like to check availability for?",
                "matched_question": None,
                "intent": "product",
                "emotion": emotion,
                "confidence": 0.9,
                "risk": "medium",
                "priority": "medium",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": {**base_meta, "confidence": 0.9}
            }

        # semantic KB fallback
        kb_idx, sim = self._semantic_search(msg)
        kb_row = self.df.iloc[kb_idx] if kb_idx is not None and kb_idx >= 0 and kb_idx < len(self.df) else None

        if kb_row is None or sim < SIMILARITY_THRESHOLD:
            ans = "I couldn't find an exact match in the knowledge base. Could you provide more details?"
            self.memory.set_last_intent(user_id, "unknown")
            meta = {**base_meta, "confidence": float(intent_conf), "similarity": float(sim)}
            meta["risk"] = risk_from(float(sim), float(intent_conf))
            return {
                "final_answer": ans,
                "matched_question": None,
                "intent": "unknown",
                "emotion": emotion,
                "confidence": float(intent_conf) if intent_conf is not None else 0.0,
                "risk": meta["risk"],
                "priority": "medium",
                "missing_info": None,
                "escalations": int(mem["escalations"]),
                "metadata": meta
            }

        # KB hit
        final_answer = kb_row.get("Answer", "") or "Sorry—I don't have that information right now."
        sim = float(sim)
        meta = {**base_meta, "confidence": float(intent_conf), "similarity": sim}
        meta["risk"] = risk_from(sim, float(intent_conf))
        if meta["risk"] == "high" or mem["escalations"] >= 3 or emotion == "angry":
            self.memory.inc_escal(user_id)
        if meta["risk"] == "high" or mem["escalations"] >= 3:
            priority = "high"
        elif meta["risk"] == "medium" or mem["escalations"] >= 1:
            priority = "medium"
        else:
            priority = "low"
        self.memory.set_last_intent(user_id, "kb")
        return {
            "final_answer": str(final_answer),
            "matched_question": str(kb_row.get("Question")) if kb_row is not None else None,
            "intent": str("unknown" if not intent else intent),
            "emotion": str(emotion),
            "confidence": float(sim),
            "risk": str(meta["risk"]),
            "priority": str(priority),
            "missing_info": None,
            "escalations": int(mem["escalations"]),
            "metadata": meta
        }

    def _detect_missing_fields(self, intent: str, message: str):
        missing = []
        if intent not in FLOW_REQUIREMENTS:
            return missing
        if intent == "order":
            if not detect_order_id(message):
                missing.append("order_id")
        elif intent == "pricing":
            if not re.search(r"\b(basic|pro|enterprise)\b", message, flags=re.IGNORECASE):
                missing.append("plan_type")
        elif intent == "product":
            if not extract_product_strict(message):
                missing.append("product_name")
        return missing

# quick CLI test
if __name__ == "__main__":
    print("AI Engine (final) ready.")
    eng = AIEngine()
    while True:
        try:
            s = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break
        if not s:
            continue
        if s.lower() in ("exit", "quit"):
            break
        import json
        print(json.dumps(eng.process("demo_user", s), indent=2, ensure_ascii=False))
