# server_fixed.py — Full SalesIQ compatible version

import os
import json
from datetime import datetime
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import threading

from ai_engine_step5 import AIEngine

LOG_FILE = "chat_logs.jsonl"
ANALYTICS_PERSIST = "analytics.json"
HOST = "0.0.0.0"
PORT = 8000

app = FastAPI(title="SalesIQ AI Engine (FINAL + SalesIQ Compatible)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Engine
engine = AIEngine()

analytics_lock = threading.Lock()
analytics = {
    "total_chats": 0,
    "intent_counts": {},
    "emotion_counts": {},
    "priority_counts": {"high": 0, "medium": 0, "low": 0},
    "escalations_total": 0,
    "last_updated": None
}


# ============================================
# ✔ HEALTH CHECK ENDPOINT (SalesIQ GET ping)
# ============================================
@app.get("/chat")
async def chat_get():
    return {"status": "ok", "detail": "Use POST to chat"}


# ============================================
# ✔ SALESIQ COMPATIBLE POST ENDPOINT
# ============================================
@app.post("/chat")
async def chat_post(request: Request):
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # ------------------------------
    # Extract user_id safely
    # ------------------------------
    user_id = (
        data.get("user_id")
        or data.get("visitor", {}).get("id")
        or data.get("client_id")
        or "visitor"
    )

    # ------------------------------
    # Extract message safely
    # ------------------------------
    message = (
        data.get("message")
        or data.get("query")
        or data.get("question")
        or data.get("text")
        or ""
    )

    if not message:
        return JSONResponse({"error": "No message received"}, status_code=400)

    # ------------------------------
    # Process with AI engine
    # ------------------------------
    try:
        result = engine.process(user_id, message)
    except Exception as e:
        return JSONResponse({"error": "AI Engine Error", "detail": str(e)}, status_code=500)

    return JSONResponse({
        "response": result.get("final_answer"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "metadata": result.get("metadata"),
        "engine_raw": result
    })


# ============================================
# RESET + METRICS + HEALTH
# ============================================
@app.post("/reset")
async def reset():
    engine.memory.data.clear()
    return {"status": "ok", "message": "Memory cleared"}

@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": True}


# Only for local test
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fixed:app", host=HOST, port=PORT, reload=True)
