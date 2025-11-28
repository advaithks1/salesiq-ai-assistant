# server_fixed.py — FINAL HACKATHON BACKEND
import os
import json
import random
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_engine_step5 import AIEngine


# -----------------------------
# Setup
# -----------------------------

app = FastAPI(title="SalesIQ AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AIEngine()


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "engine": True}


# -----------------------------
# CHAT ENDPOINT
# -----------------------------

@app.post("/chat")
async def chat_post(request: Request):

    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    user_id = (
        data.get("user_id")
        or data.get("visitor_id")
        or "visitor"
    )

    msg = (
        data.get("message")
        or data.get("text")
        or data.get("query")
        or ""
    )

    if not msg:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    try:
        result = engine.process(user_id, msg)
    except Exception as e:
        return JSONResponse({"error": "AI Engine Error", "detail": str(e)}, status_code=500)

    return {
        "response": result.get("final_answer"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "metadata": result.get("metadata"),
        "engine_raw": result
    }


# -----------------------------
# ORDER TRACKING ENDPOINT
# -----------------------------

ORDER_STAGES = [
    "Order confirmed",
    "Packing",
    "Ready to ship",
    "Shipped",
    "In transit",
    "Out for delivery",
    "Delivered"
]

@app.get("/order")
async def order_lookup(oid: str):

    if not oid:
        return {"error": "Missing order id"}

    # deterministic
    try:
        seed = int("".join(filter(str.isdigit, oid))) % 9999
    except:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)

    stage = ORDER_STAGES[idx]
    eta = max(0, 5 - idx)

    return {
        "order_id": oid,
        "stage": stage,
        "eta_days": eta,
        "history": ORDER_STAGES[:idx + 1]
    }


# -----------------------------
# RUN LOCAL
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fixed:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
