# server_fixed.py — FINAL VERSION
"""
FastAPI backend for Smart AI Assistant.

Features:
- /chat      -> main AI engine endpoint (uses AIEngine from ai_engine_step5.py)
- /order     -> order tracking integrated with external API + fallback simulation
- /products  -> static product catalog (shared with bot, aligned with AI engine)
- /cart      -> shared cart APIs (frontend + chatbot)
- /health    -> basic status
"""

import os
import random
from datetime import datetime
from typing import Dict, Any

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_engine_step5 import AIEngine, PRODUCT_DB, memory

# -------------------------------------------------------
# App Setup
# -------------------------------------------------------
app = FastAPI(title="SalesIQ AI Engine with Agent Assist PRO + E-Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # open for demo / hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AIEngine()

# -------------------------------------------------------
# Health Check
# -------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "service": "smart-ai-assistant",
    }


# -------------------------------------------------------
# Helper: build static product list from PRODUCT_DB
# -------------------------------------------------------
def build_product_list():
    """
    Build a static product list from the AI engine's PRODUCT_DB.
    Ensures IDs and prices are consistent across bot + frontend.
    """
    products = []
    for pid_str, info in PRODUCT_DB.items():
        name = info.get("name", f"Product {pid_str}")
        price_str = info.get("price", "0")
        tag = info.get("tag", "")

        # Convert "₹1,299" -> 1299
        digits = "".join(ch for ch in price_str if ch.isdigit())
        price = int(digits) if digits else 0

        products.append(
            {
                "id": int(pid_str),
                "title": name,
                "price": price,
                "tag": tag,
            }
        )
    # Sort by id for nice display
    products.sort(key=lambda p: p["id"])
    return products


# -------------------------------------------------------
# Helper: order simulation (fallback)
# -------------------------------------------------------
ORDER_STAGES = [
    "Order confirmed",
    "Packing",
    "Ready to ship",
    "Shipped",
    "In transit",
    "Out for delivery",
    "Delivered",
]


def simulate_order(oid: str) -> Dict[str, Any]:
    """
    Deterministic fallback simulation used when external API fails.
    Keeps behavior stable for the demo.
    """
    try:
        seed = int("".join([c for c in oid if c.isdigit()])) % 9999
    except Exception:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)

    return {
        "order_id": str(oid),
        "stage": ORDER_STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": ORDER_STAGES[: idx + 1],
        "source": "simulation",
    }


# -------------------------------------------------------
# Helper: shared CART snapshot (AIEngine memory -> API)
# -------------------------------------------------------
def build_cart_snapshot(user_id: str):
    """
    Build a unified cart view for the given user_id based on the
    in-memory cart used by AIEngine (same cart as chatbot).
    """
    mem = memory.data[user_id]
    id_list = mem["cart"]  # list of product_id strings like "101"

    # Count quantities
    counts: Dict[str, int] = {}
    for pid in id_list:
        counts[pid] = counts.get(pid, 0) + 1

    items = []
    for pid_str, qty in counts.items():
        info = PRODUCT_DB.get(pid_str, {})
        name = info.get("name", f"Product {pid_str}")
        price_str = info.get("price", "0")

        # Convert "₹1,299" -> 1299
        digits = "".join(ch for ch in price_str if ch.isdigit())
        price = int(digits) if digits else 0

        items.append(
            {
                "id": int(pid_str),
                "title": name,
                "price": price,
                "qty": qty,
            }
        )

    # Sort by id for readability
    items.sort(key=lambda i: i["id"])

    return {
        "user_id": user_id,
        "items": items,
    }


# -------------------------------------------------------
# CHAT ENDPOINT — MAIN BOT LOGIC
# -------------------------------------------------------
@app.post("/chat")
async def chat_post(request: Request):
    """
    Main endpoint used by Zoho SalesIQ Deluge script.
    Accepts JSON with fields like:
    {
      "user_id": "...",
      "message": "hi there"
    }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON received")

    user_id = (
        data.get("user_id")
        or data.get("visitor", {}).get("id")
        or data.get("client_id")
        or "visitor"
    )

    message = (
        data.get("message")
        or data.get("text")
        or data.get("query")
        or ""
    )

    if not message:
        return JSONResponse({"error": "No message provided"}, status_code=400)

    # AI Engine processing
    try:
        result = engine.process(user_id, message)
    except Exception as e:
        return JSONResponse(
            {"error": "Engine Failure", "detail": str(e)},
            status_code=500,
        )

    # API Response (stable shape for frontend / SalesIQ)
    return {
        "response": result.get("final_answer"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "confidence": result.get("confidence"),
        "metadata": result.get("metadata"),
    }


# -------------------------------------------------------
# ORDER LOOKUP — external API + fallback simulation
# -------------------------------------------------------
@app.get("/order")
async def order_lookup(oid: str):
    """
    Order lookup integrated with an external e-commerce style API (fakestoreapi carts),
    with a deterministic simulation as fallback.

    Used by:
    - Zoho SalesIQ bot (for messages like "track 101")
    - Frontend page.
    """
    if not oid:
        return {"error": "Missing order ID"}

    # 1) Try external "e-commerce" backend (Fakestore carts)
    try:
        cart_id = int(oid)  # simple mapping: order id -> cart id
        resp = requests.get(
            f"https://fakestoreapi.com/carts/{cart_id}",
            timeout=10,
        )

        if resp.status_code == 200:
            data = resp.json()

            # Simple readable status + ETA for demo
            stage = "Processing"
            eta_days = 2
            history = [
                "Order created in external store",
                "Payment verified",
                "Items packed & ready to ship",
            ]

            return {
                "order_id": str(oid),
                "stage": stage,
                "eta_days": eta_days,
                "history": history,
                "items": data.get("products", []),
                "source": "fakestoreapi",
            }
    except Exception:
        # If anything fails we don't break the bot; just fall back.
        pass

    # 2) Fallback to deterministic simulation
    return simulate_order(oid)


# -------------------------------------------------------
# PRODUCT LIST — STATIC (shared with AI engine & bot)
# -------------------------------------------------------
@app.get("/products")
async def get_products():
    """
    Static product catalog, derived from PRODUCT_DB.

    Used by:
    - Zoho SalesIQ bot (commands like "show products")
    - (Optional) frontend if you ever want to fetch instead of static array.
    """
    products = build_product_list()
    return {
        "products": products,
        "source": "static",
    }


# -------------------------------------------------------
# CART API — shared between chatbot and frontend
# -------------------------------------------------------
@app.get("/cart")
async def get_cart(user_id: str = "demo-user"):
    """
    Return the current cart for the given user_id,
    based on AIEngine's in-memory cart.
    """
    return build_cart_snapshot(user_id)


@app.post("/cart/add")
async def cart_add(payload: Dict[str, Any]):
    """
    Add a product_id to the user's cart and return updated snapshot.
    Expected JSON:
    { "user_id": "demo-user", "product_id": 101 }
    """
    user_id = str(payload.get("user_id") or "demo-user")
    pid = payload.get("product_id")
    if pid is None:
        raise HTTPException(status_code=400, detail="Missing product_id")

    pid_str = str(pid)
    mem = memory.data[user_id]
    mem["cart"].append(pid_str)

    return build_cart_snapshot(user_id)


@app.post("/cart/remove")
async def cart_remove(payload: Dict[str, Any]):
    """
    Remove a product_id from the user's cart (all occurrences)
    and return updated snapshot.
    Expected JSON:
    { "user_id": "demo-user", "product_id": 101 }
    """
    user_id = str(payload.get("user_id") or "demo-user")
    pid = payload.get("product_id")
    if pid is None:
        raise HTTPException(status_code=400, detail="Missing product_id")

    pid_str = str(pid)
    mem = memory.data[user_id]
    mem["cart"] = [x for x in mem["cart"] if x != pid_str]

    return build_cart_snapshot(user_id)


# -------------------------------------------------------
# ROOT
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "SalesIQ AI Engine Ready",
        "endpoints": ["/chat", "/order", "/products", "/cart", "/health"],
    }


# -------------------------------------------------------
# Local Dev Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_fixed:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
