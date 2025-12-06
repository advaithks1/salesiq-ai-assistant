# server_fixed.py — FINAL VERSION (Agent Assist PRO + E-Commerce Integration)
"""
FastAPI backend for Smart AI Assistant.

Features:
- /chat      -> main AI engine endpoint (uses AIEngine from ai_engine_step5.py)
- /order     -> order tracking integrated with external e-commerce style API (fakestoreapi carts) + fallback simulation
- /products  -> product catalog endpoint using DummyJSON API (stable)
- /health    -> basic status
"""

import os
import json
import random
from datetime import datetime
from typing import Dict, Any

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_engine_step5 import AIEngine

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


@app.get("/order")
async def order_lookup(oid: str):
    """
    Order lookup integrated with an external e-commerce style API (fakestoreapi carts),
    with a deterministic simulation as fallback.

    Used by:
    - Zoho SalesIQ bot (for messages like "track 101")
    - Documented in the frontend page.
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
# PRODUCT LIST — using DummyJSON (very stable API)
# -------------------------------------------------------
@app.get("/products")
async def get_products():
    """
    Fetch products from DummyJSON API to act as an e-commerce catalog.
    Used by:
    - Zoho SalesIQ bot (commands like "show products")
    - Frontend page (optional, if you want to consume it there as well).
    """
    try:
        resp = requests.get("https://dummyjson.com/products?limit=10", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return {
            "products": data.get("products", []),
            "source": "dummyjson",
        }
    except Exception as e:
        # In case of error, return an empty list but keep shape same
        return {
            "products": [],
            "source": "error",
            "detail": str(e),
        }


# -------------------------------------------------------
# ROOT
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "SalesIQ AI Engine Ready",
        "endpoints": ["/chat", "/order", "/products", "/health"],
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
