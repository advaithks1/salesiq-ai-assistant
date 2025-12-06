# ai_engine_step5.py — FINAL PRO+ HACKATHON VERSION (with E-Commerce Intents)
"""
Smart AI Engine for SalesIQ Hackathon
Features:
- Intent detection (order, pricing, support, greeting, login, escalate)
- E-commerce flows: add to cart, view cart, place order, my orders, cancel, reorder,
  product details & basic product search intent
- FULL sentiment engine (lexicon + negation)
- FAQ system (shipping, refund, payment, account)
- Order tracking simulation
- Product browsing support (frontend / Deluge use metadata)
- Safety filter
- Memory & conversation context (cart + orders per user)
- Agent Assist PRO (frustration, risk, summary, suggestions)
"""

import re
import random
from collections import defaultdict, deque
from typing import Dict, Any


# ============================================================
#  SIMPLE PRODUCT DB (aligned with frontend demo)
# ============================================================

PRODUCT_DB: Dict[str, Dict[str, str]] = {
    "101": {"name": "Wireless Earbuds", "price": "₹1,299", "tag": "Best Seller"},
    "102": {"name": "Smartwatch", "price": "₹2,499", "tag": "Trending"},
    "103": {"name": "Bluetooth Speaker", "price": "₹1,999", "tag": "New"},
    "104": {"name": "Laptop Stand", "price": "₹799", "tag": "Popular"},
}


# ============================================================
#  INTENT RULES
# ============================================================

GREETING_WORDS = {"hi", "hello", "hey", "hii"}

# base rules (used as fallback after some special handling)
INTENT_RULES = {
    "order": ["track", "order", "delivery"],
    "pricing": ["price", "pricing", "plan", "subscription", "cost"],
    "support": ["help", "issue", "problem", "error", "bug", "crash"],
    "escalate": ["agent", "human", "supervisor"],
    "login": ["password", "reset password", "forgot password"],
    # note: e-commerce intents are handled with custom logic below
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

    # -----------------------------
    # E-COMMERCE SPECIFIC INTENTS
    # -----------------------------

    # Add to cart
    if "add to cart" in msg or ("add " in msg and "cart" in msg):
        return "add_to_cart", 0.96

    # View cart
    if msg in {"cart", "my cart"} or "show my cart" in msg or "show cart" in msg:
        return "view_cart", 0.95

    # Place order / checkout
    if (
        "checkout" in msg
        or "place order" in msg
        or "confirm order" in msg
        or "buy now" in msg
    ):
        return "place_order", 0.95

    # My orders / order history
    if (
        "my orders" in msg
        or "order history" in msg
        or "previous orders" in msg
        or "past orders" in msg
    ):
        return "my_orders", 0.95

    # Cancel order
    if "cancel order" in msg or msg.startswith("cancel "):
        return "cancel_order", 0.95

    # Reorder
    if "reorder" in msg or "order again" in msg or "buy again" in msg:
        return "reorder", 0.95

    # Product search
    if msg.startswith("search ") or msg.startswith("find "):
        return "product_search", 0.9

    # Product details
    if (
        msg.startswith("details ")
        or msg.startswith("detail ")
        or msg.startswith("show product")
        or "product info" in msg
    ):
        return "product_details", 0.9

    # -----------------------------
    # Generic rule-based intents
    # -----------------------------
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

POS = {
    "great",
    "good",
    "awesome",
    "nice",
    "thanks",
    "thank",
    "perfect",
    "love",
    "cool",
    "happy",
    "amazing",
}
NEG = {
    "bad",
    "worst",
    "slow",
    "issue",
    "problem",
    "error",
    "bug",
    "crash",
    "delay",
    "late",
    "annoying",
}
STRONG_NEG = {
    "hate",
    "angry",
    "furious",
    "terrible",
    "stupid",
    "fuck",
    "fucking",
    "shit",
    "useless",
}
NEGATION = {"not", "never", "no", "dont", "don't", "can't", "cant"}


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
            if nxt in POS:
                score -= 1.5
                i += 2
                continue
            if nxt in NEG or nxt in STRONG_NEG:
                score += 0.5
                i += 2
                continue

        if w in POS:
            score += 1
        if w in NEG:
            score -= 1
        if w in STRONG_NEG:
            score -= 2

        i += 1

    return score


def classify_emotion(text: str):
    score = sentiment_score(text)

    if score >= 1:
        return "happy", min(1.0, 0.7 + 0.1 * score)
    if score <= -1:
        return "angry", min(1.0, 0.7 + 0.1 * abs(score))

    return "neutral", max(0.3, 0.6 - 0.1 * abs(score))


# ============================================================
#  FAQ SYSTEM
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
    msg = (text or "").lower()
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
        # per-user:
        # - context: last messages
        # - escalations: count
        # - cart: list of product IDs
        # - orders: dict[order_id] -> dict(items, status)
        self.data = defaultdict(
            lambda: {
                "context": deque(maxlen=20),
                "escalations": 0,
                "cart": [],
                "orders": {},
            }
        )

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
    "Delivered",
]


def simulate_order(oid: str):
    try:
        seed = int("".join(filter(str.isdigit, oid))) % 9999
    except Exception:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(STAGES) - 1)

    return {
        "order_id": oid,
        "stage": STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": STAGES[: idx + 1],
    }


# ============================================================
#  AGENT ASSIST
# ============================================================

def frustration(ctx):
    score = 0
    for m in ctx:
        s = sentiment_score(m["text"])
        if s <= -2:
            score += 2
        elif s <= -0.5:
            score += 1
    return score


def risk(intent, emotion):
    if emotion == "angry":
        return "high"
    if intent in {"escalate", "cancel_order"}:
        return "medium"
    return "low"


def summary(ctx):
    last = " ".join(c["text"] for c in list(ctx)[-3:])
    return "Recent user messages: " + last if last else "No conversation yet."


def suggest(intent, emotion):
    if intent == "order":
        return "Reassure user and confirm order ID."
    if intent == "pricing":
        return "Offer a quick plan comparison."
    if intent == "support":
        return "Ask for details or screenshot."
    if intent == "add_to_cart":
        return "Confirm the product and suggest viewing cart or checkout."
    if intent == "view_cart":
        return "Walk the user through checkout or product changes."
    if intent == "place_order":
        return "Confirm address/payment and reassure about delivery timeline."
    if intent == "my_orders":
        return "Highlight most recent order and offer tracking."
    if intent == "cancel_order":
        return "Acknowledge issue, confirm policy and offer alternatives."
    if intent == "reorder":
        return "Confirm items and suggest similar products."
    if intent in {"product_search", "product_details"}:
        return "Provide clear product info and next steps."
    if emotion == "angry":
        return "Use empathy and offer agent escalation."
    return "Guide the user to the next step."


# ============================================================
#  MAIN ENGINE
# ============================================================

class AIEngine:
    def __init__(self):
        self.memory = memory

    def process(self, user_id, message):
        msg = (message or "").strip()
        mem = self.memory.data[user_id]
        self.memory.push(user_id, "user", msg)

        # Safety
        if any(w in msg.lower() for w in ["kill", "bomb", "terror", "suicide"]):
            return {
                "final_answer": "I cannot assist with that.",
                "intent": "blocked",
                "emotion": "neutral",
                "confidence": 1.0,
                "metadata": {},
            }

        # Emotion + Intent
        intent, _ = classify_intent(msg)
        emotion, _ = classify_emotion(msg)

        # Metadata generator
        def build(extra: Dict[str, Any]):
            ctx = mem["context"]
            return {
                **extra,
                "suggestion": suggest(intent, emotion),
                "frustration": frustration(ctx),
                "risk": risk(intent, emotion),
                "summary": summary(ctx),
                "m1": ctx[-1]["text"] if len(ctx) > 0 else "",
                "m2": ctx[-2]["text"] if len(ctx) > 1 else "",
                "m3": ctx[-3]["text"] if len(ctx) > 2 else "",
            }

        # ------------------ FAQ ------------------
        faq = match_faq(msg)
        if faq:
            return {
                "final_answer": FAQ_ANS[faq],
                "intent": "support",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": build({"faq_topic": faq}),
            }

        # ------------------ ORDER (tracking by ID) ------------------
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
                    "metadata": build(
                        {
                            "order_id": oid,
                            "order_stage": o["stage"],
                            "eta_days": o["eta_days"],
                            "history": o["history"],
                        }
                    ),
                }
            else:
                # No order ID provided
                return {
                    "final_answer": "Please share your order ID (like *101* or *500234*) so I can track it for you.",
                    "intent": "order",
                    "emotion": emotion,
                    "confidence": 0.8,
                    "metadata": build({"missing_order_id": True}),
                }

        # ------------------ E-COMMERCE FLOWS ------------------

        # ADD TO CART
        if intent == "add_to_cart":
            ids = re.findall(r"\b(\d{3,12})\b", msg)
            cart = mem["cart"]
            added = []
            for pid in ids:
                cart.append(pid)
                added.append(pid)

            if not added:
                ans = "Tell me which product ID to add, for example: *add 101 to cart*."
            else:
                # build human-friendly description
                lines = []
                for pid in added:
                    info = PRODUCT_DB.get(pid)
                    if info:
                        lines.append(f"{pid} — {info['name']} ({info['price']})")
                    else:
                        lines.append(pid)
                ans = "🛒 Added to your cart:\n" + "\n".join(f"- {l}" for l in lines)

            return {
                "final_answer": ans,
                "intent": "add_to_cart",
                "emotion": emotion,
                "confidence": 0.96,
                "metadata": build({"cart_items": list(mem["cart"])}),
            }

        # VIEW CART
        if intent == "view_cart":
            cart = mem["cart"]
            if not cart:
                ans = (
                    "Your cart is empty. You can say *show products* or "
                    "*add 101 to cart* to start."
                )
                meta_extra: Dict[str, Any] = {"cart_items": []}
            else:
                lines = []
                for pid in cart:
                    info = PRODUCT_DB.get(pid)
                    if info:
                        lines.append(f"{pid} — {info['name']} ({info['price']})")
                    else:
                        lines.append(pid)
                ans = "🛒 *Your Cart*\n\n" + "\n".join(f"- {l}" for l in lines)
                meta_extra = {"cart_items": list(cart), "cart_size": len(cart)}

            return {
                "final_answer": ans,
                "intent": "view_cart",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": build(meta_extra),
            }

        # PLACE ORDER / CHECKOUT
        if intent == "place_order":
            cart = mem["cart"]
            if not cart:
                ans = "Your cart is empty. Add some items first, e.g. *add 101 to cart*."
                return {
                    "final_answer": ans,
                    "intent": "place_order",
                    "emotion": emotion,
                    "confidence": 0.8,
                    "metadata": build({"cart_items": [], "order_created": False}),
                }

            # create synthetic order id
            oid = str(random.randint(100000, 999999))
            mem["orders"][oid] = {"items": list(cart), "status": "Order confirmed"}
            # we can reuse simulation to provide status & ETA
            o = simulate_order(oid)
            mem["cart"] = []

            items_line = ", ".join(mem["orders"][oid]["items"])
            ans = (
                "🎉 *Order Placed Successfully!*\n\n"
                f"Order ID: {oid}\n"
                f"Items: {items_line}\n"
                f"Current Status: {o['stage']}\n"
                f"ETA: {o['eta_days']} day(s)\n\n"
                "You can type *track {order_id}* in the bot or website to check status."
            )

            return {
                "final_answer": ans,
                "intent": "place_order",
                "emotion": emotion,
                "confidence": 0.98,
                "metadata": build(
                    {
                        "order_id": oid,
                        "order_stage": o["stage"],
                        "eta_days": o["eta_days"],
                        "history": o["history"],
                        "order_items": mem["orders"][oid]["items"],
                        "order_created": True,
                    }
                ),
            }

        # MY ORDERS
        if intent == "my_orders":
            orders = mem["orders"]
            if not orders:
                ans = (
                    "You have no recent orders yet. You can place one by saying "
                    "*checkout* after adding items to your cart."
                )
                return {
                    "final_answer": ans,
                    "intent": "my_orders",
                    "emotion": emotion,
                    "confidence": 0.9,
                    "metadata": build({"orders_count": 0}),
                }

            lines = []
            meta_orders = []
            for oid, info in list(orders.items())[-5:]:
                # get richer simulated status for display
                sim = simulate_order(oid)
                line = f"{oid} → {sim['stage']} (ETA {sim['eta_days']} day(s))"
                lines.append(line)
                meta_orders.append(
                    {
                        "order_id": oid,
                        "stage": sim["stage"],
                        "eta_days": sim["eta_days"],
                        "items": info.get("items", []),
                    }
                )

            ans = "🧾 *Your Recent Orders*\n\n" + "\n".join(f"- {l}" for l in lines)
            return {
                "final_answer": ans,
                "intent": "my_orders",
                "emotion": emotion,
                "confidence": 0.96,
                "metadata": build(
                    {"orders_count": len(orders), "orders_preview": meta_orders}
                ),
            }

        # CANCEL ORDER
        if intent == "cancel_order":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if not m:
                ans = "Please tell me which order to cancel, e.g. *cancel order 500123*."
                return {
                    "final_answer": ans,
                    "intent": "cancel_order",
                    "emotion": emotion,
                    "confidence": 0.8,
                    "metadata": build({"cancelled": False}),
                }

            oid = m.group(1)
            orders = mem["orders"]
            if oid not in orders:
                ans = (
                    f"I couldn’t find order *{oid}* in your recent orders. "
                    "If this is important, I can connect you to a human agent."
                )
                return {
                    "final_answer": ans,
                    "intent": "cancel_order",
                    "emotion": emotion,
                    "confidence": 0.85,
                    "metadata": build({"cancelled": False, "order_id": oid}),
                }

            orders[oid]["status"] = "Cancelled"
            ans = (
                f"❌ Your order *{oid}* has been marked as cancelled in this session.\n"
                "If the order was already processed by the store, an agent can confirm the final status."
            )
            return {
                "final_answer": ans,
                "intent": "cancel_order",
                "emotion": emotion,
                "confidence": 0.96,
                "metadata": build({"cancelled": True, "order_id": oid}),
            }

        # REORDER
        if intent == "reorder":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if not m:
                ans = "Please tell me which order to reorder, e.g. *reorder 500123*."
                return {
                    "final_answer": ans,
                    "intent": "reorder",
                    "emotion": emotion,
                    "confidence": 0.8,
                    "metadata": build({"reorder_done": False}),
                }

            oid = m.group(1)
            orders = mem["orders"]
            if oid not in orders:
                ans = (
                    f"I couldn’t find order *{oid}* in your recent orders. "
                    "You can still add items again manually."
                )
                return {
                    "final_answer": ans,
                    "intent": "reorder",
                    "emotion": emotion,
                    "confidence": 0.85,
                    "metadata": build({"reorder_done": False, "order_id": oid}),
                }

            items = orders[oid].get("items", [])
            mem["cart"].extend(items)
            ans = (
                f"🛒 I’ve added items from order *{oid}* back to your cart.\n"
                "You can say *show my cart* or *checkout* to continue."
            )
            return {
                "final_answer": ans,
                "intent": "reorder",
                "emotion": emotion,
                "confidence": 0.96,
                "metadata": build(
                    {"reorder_done": True, "order_id": oid, "cart_items": mem["cart"]}
                ),
            }

        # PRODUCT SEARCH (high-level intent only, actual API via backend if needed)
        if intent == "product_search":
            # extract query words after 'search' or 'find'
            q = msg
            m = re.match(r"(search|find)\s+(.*)", msg, re.I)
            if m:
                q = m.group(2).strip()

            if not q:
                ans = (
                    "Tell me what you are looking for, e.g. *search earbuds* "
                    "or *find smartwatch*."
                )
            else:
                ans = (
                    f"🔍 I’ll look for products matching *{q}*.\n"
                    "In this demo, you can also type *show products* to see the catalog."
                )

            return {
                "final_answer": ans,
                "intent": "product_search",
                "emotion": emotion,
                "confidence": 0.9,
                "metadata": build({"search_query": q}),
            }

        # PRODUCT DETAILS
        if intent == "product_details":
            m = re.search(r"\b(\d{3,12})\b", msg)
            if not m:
                ans = "Tell me the product ID, e.g. *details 101* or *show product 102*."
                return {
                    "final_answer": ans,
                    "intent": "product_details",
                    "emotion": emotion,
                    "confidence": 0.85,
                    "metadata": build({}),
                }

            pid = m.group(1)
            info = PRODUCT_DB.get(pid)
            if not info:
                ans = (
                    f"I don’t have detailed info for product *{pid}* yet. "
                    "You can still track it by order ID or say *show products*."
                )
                meta_extra = {"product_id": pid, "known": False}
            else:
                ans = (
                    "📄 *Product Details*\n\n"
                    f"ID: {pid}\n"
                    f"Name: {info['name']}\n"
                    f"Price: {info['price']}\n"
                    f"Tag: {info.get('tag', '-')}\n\n"
                    "You can say *add {pid} to cart* or *show my cart* next."
                )
                meta_extra = {"product_id": pid, "known": True, "product": info}

            return {
                "final_answer": ans,
                "intent": "product_details",
                "emotion": emotion,
                "confidence": 0.94,
                "metadata": build(meta_extra),
            }

        # ------------------ PRICING ------------------
        if intent == "pricing":
            plan = "basic"
            m = re.search(r"(basic|pro|enterprise)", msg, re.I)
            if m:
                plan = m.group(1).lower()
            price = {"basic": "₹499", "pro": "₹1299", "enterprise": "₹4999"}[plan]
            return {
                "final_answer": "pricing_info",
                "intent": "pricing",
                "emotion": emotion,
                "confidence": 0.95,
                "metadata": build({"plan": plan, "price": price}),
            }

        # ------------------ GREETING ------------------
        if intent == "greeting":
            return {
                "final_answer": "Hi! How can I help you today? 😊",
                "intent": "greeting",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": build({}),
            }

        # ------------------ LOGIN ------------------
        if intent == "login":
            return {
                "final_answer": "Use the *Forgot Password* option on the login page. I can also connect you to an agent if needed.",
                "intent": "login",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": build({}),
            }

        # ------------------ SUPPORT ------------------
        if intent == "support":
            return {
                "final_answer": "🔧 Please describe your issue.",
                "intent": "support",
                "emotion": emotion,
                "confidence": 0.9,
                "metadata": build({}),
            }

        # ------------------ ESCALATE ------------------
        if intent == "escalate":
            mem["escalations"] += 1
            return {
                "final_answer": "Connecting you to a human agent...",
                "intent": "escalate",
                "emotion": emotion,
                "confidence": 1.0,
                "metadata": build({}),
            }

        # ------------------ FALLBACK ------------------
        if emotion == "angry":
            ans = (
                "I’m really sorry this has been frustrating. "
                "Tell me what went wrong — I can help or connect you to an agent."
            )
        else:
            ans = (
                "I couldn’t fully understand that. "
                "You can ask me to *track an order*, *show products*, "
                "manage your *cart* or *order*, or *help with shipping/refund/payment*."
            )

        return {
            "final_answer": ans,
            "intent": "unknown",
            "emotion": emotion,
            "confidence": 0.3,
            "metadata": build({}),
        }
