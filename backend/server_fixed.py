# server_fixed.py — FINAL VERSION (Compatible with Agent Assist PRO)

import os
import json
import random
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_engine_step5 import AIEngine


# -------------------------------------------------------
# App Setup
# -------------------------------------------------------
app = FastAPI(title="SalesIQ AI Engine with Agent Assist PRO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# -------------------------------------------------------
# CHAT ENDPOINT — MAIN BOT LOGIC
# -------------------------------------------------------
@app.post("/chat")
async def chat_post(request: Request):

    try:
        data = await request.json()
    except:
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
        return JSONResponse({"error": "Engine Failure", "detail": str(e)}, status_code=500)

    # API Response
    return {
        "response": result.get("final_answer"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "confidence": result.get("confidence"),
        "metadata": result.get("metadata"),
    }


# -------------------------------------------------------
# ORDER LOOKUP SIMULATION
# -------------------------------------------------------
ORDER_STAGES = [
    "Order confirmed",
    "Packing",
    "Ready to ship",
    "Shipped",
    "In transit",
    "Out for delivery",
    "Delivered"
]

def simulate_order(oid: str):
    try:
        seed = int("".join([c for c in oid if c.isdigit()])) % 9999
    except:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)

    return {
        "order_id": oid,
        "stage": ORDER_STAGES[idx],
        "eta_days": max(0, 5 - idx),
        "history": ORDER_STAGES[:idx + 1]
    }


@app.get("/order")
async def order_lookup(oid: str):

    if not oid:
        return {"error": "Missing order ID"}

    o = simulate_order(oid)
    return o


# -------------------------------------------------------
# ROOT
# -------------------------------------------------------
@app.get("/")
def root():
    return {"status": "SalesIQ AI Engine Ready"}


# -------------------------------------------------------
# Local Dev
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fixed:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
