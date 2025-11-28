# server_fixed.py — PREMIUM backend (analytics + order + emotion)
import os
import json
import threading
import random
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from ai_engine_step5 import AIEngine

# Paths
ANALYTICS_FILE = "analytics.json"
LOG_FILE = "chat_logs.jsonl"

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

# App
app = FastAPI(title="SalesIQ AI Engine — PREMIUM")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine
engine = AIEngine()

# Analytics
analytics_lock = threading.Lock()

def default_analytics() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "intent_counts": {},
        "emotion_counts": {},
        "priority_counts": {"high": 0, "medium": 0, "low": 0},
        "escalations_total": 0,
        "top_questions": {},
        "last_updated": None,
        "chat_log": []
    }

def load_analytics():
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return default_analytics()

analytics = load_analytics()

def save_analytics():
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics, f, indent=2)
    except Exception as e:
        print("Error saving analytics:", e)

# -----------------------
# Health Check
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": True, "time": datetime.utcnow().isoformat()}

# -----------------------
# Chat GET
# -----------------------
@app.get("/chat")
async def chat_get():
    return {"status": "ok", "detail": "Use POST /chat to interact"}

# -----------------------
# CHAT — FINAL FIXED POST
# Accept JSON OR form-data (SalesIQ fix)
# -----------------------
@app.post("/chat")
async def chat_post(request: Request):
    # 1) Try JSON first
    try:
        data = await request.json()
    except:
        # 2) Try form-data (SalesIQ sends this)
        try:
            form = await request.form()
            data = dict(form)
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON or form data")

    # Extract user_id
    user_id = (
        data.get("user_id")
        or data.get("visitor", {}).get("id")
        or data.get("client_id")
        or "visitor"
    )

    # Extract message
    message = (
        data.get("message")
        or data.get("text")
        or data.get("query")
        or data.get("question")
        or ""
    ).strip()

    if not message:
        return JSONResponse({"error": "No message received"}, status_code=400)

    # Process using AI Engine
    try:
        result = engine.process(user_id, message)
    except Exception as e:
        return JSONResponse(
            {"error": "AI Engine Error", "detail": str(e)},
            status_code=500
        )

    # ================
    # Analytics update
    # ================
    with analytics_lock:
        analytics["total_requests"] += 1
        intent = result.get("intent", "unknown")
        emotion = result.get("emotion", "neutral")
        priority = result.get("priority", "medium")

        analytics["intent_counts"][intent] = analytics["intent_counts"].get(intent, 0) + 1
        analytics["emotion_counts"][emotion] = analytics["emotion_counts"].get(emotion, 0) + 1
        analytics["priority_counts"][priority] = analytics["priority_counts"].get(priority, 0) + 1

        # Escalations total
        analytics["escalations_total"] += engine.memory.data[user_id]["escalations"]

        # Track most common questions
        q = result.get("matched_question") or message
        analytics["top_questions"][q] = analytics["top_questions"].get(q, 0) + 1

        # Log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "message": message,
            "response": result.get("final_answer"),
            "intent": intent,
            "emotion": emotion,
            "priority": priority,
            "metadata": result.get("metadata", {})
        }
        analytics["chat_log"].append(log_entry)
        analytics["chat_log"] = analytics["chat_log"][-500:]  # keep last 500
        analytics["last_updated"] = datetime.utcnow().isoformat()

        save_analytics()

    return {
        "response": result.get("final_answer"),
        "intent": intent,
        "emotion": emotion,
        "metadata": result.get("metadata", {}),
        "engine_raw": result
    }

# -----------------------
# ORDER LOOKUP
# -----------------------
ORDER_STAGES = [
    "Order confirmed", "Packing", "Ready to ship", "Shipped",
    "In transit", "Out for delivery", "Delivered"
]

def simulate_order(oid: str):
    if not oid:
        oid = "0"
    try:
        seed = int("".join(filter(str.isdigit, oid))) % 9999
    except:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    stage_index = random.randint(0, len(ORDER_STAGES) - 1)
    stage = ORDER_STAGES[stage_index]
    eta = max(0, 5 - stage_index)

    return {
        "order_id": oid,
        "stage": stage,
        "eta_days": eta,
        "status": "Delivered" if stage == "Delivered" else "In Progress",
        "history": ORDER_STAGES[:stage_index+1]
    }

@app.get("/order")
async def order_lookup(oid: str):
    if not oid:
        return {"error": "Missing order ID"}
    data = simulate_order(oid)

    with analytics_lock:
        analytics["intent_counts"]["order_lookup"] = analytics["intent_counts"].get("order_lookup", 0) + 1
        analytics["total_requests"] += 1
        analytics["last_updated"] = datetime.utcnow().isoformat()
        save_analytics()

    return data

# -----------------------
# ANALYTICS
# -----------------------
@app.get("/analytics")
async def get_analytics():
    return analytics

@app.get("/analytics/csv")
async def analytics_csv():
    if not analytics.get("chat_log"):
        return {"error": "No data yet"}

    df = pd.DataFrame(analytics["chat_log"])
    csv_path = "analytics_export.csv"
    df.to_csv(csv_path, index=False)

    return FileResponse(csv_path, media_type="text/csv", filename="analytics.csv")

@app.post("/analytics/reset")
async def reset_analytics():
    global analytics
    analytics = default_analytics()
    save_analytics()
    return {"status": "ok", "message": "Analytics reset"}

# -----------------------
# MEMORY CLEAR
# -----------------------
@app.post("/reset")
async def reset_memory():
    engine.memory.data.clear()
    return {"status": "ok", "message": "Memory cleared"}

# -----------------------
# Local Dev Runner
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fixed:app", host=HOST, port=PORT, reload=True)
