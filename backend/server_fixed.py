# server_fixed.py — FINAL FULL ANALYTICS + ORDER API + SALESIQ COMPATIBLE

import os
import json
import threading
import random
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from ai_engine_step5 import AIEngine

# -----------------------------------------------------
# File paths
# -----------------------------------------------------
ANALYTICS_FILE = "analytics.json"
LOG_FILE = "chat_logs.jsonl"

HOST = "0.0.0.0"
PORT = 8000

# -----------------------------------------------------
# Init FastAPI app
# -----------------------------------------------------
app = FastAPI(title="SalesIQ AI Engine — FULL ANALYTICS VERSION")

# -----------------------------------------------------
# CORS (required for SalesIQ + React)
# -----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Load AI Engine
# -----------------------------------------------------
engine = AIEngine()

# -----------------------------------------------------
# Analytics (persistent)
# -----------------------------------------------------
analytics_lock = threading.Lock()

def load_analytics():
    """Load analytics from file or create defaults."""
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r") as f:
                return json.load(f)
        except:
            pass

    # default structure
    return {
        "total_requests": 0,
        "intent_counts": {},
        "emotion_counts": {},
        "priority_counts": {"high": 0, "medium": 0, "low": 0},
        "escalations_total": 0,
        "last_updated": None,
        "chat_log": []  # stores last 150 chats
    }

analytics = load_analytics()

def save_analytics():
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics, f, indent=2)
    except Exception as e:
        print("Analytics save error:", e)

# =====================================================
#                    CHAT ENDPOINTS
# =====================================================

@app.get("/chat")
async def chat_get():
    return {"status": "ok", "detail": "Use POST /chat to send messages"}

@app.post("/chat")
async def chat_post(request: Request):
    """SalesIQ-compatible chat endpoint"""
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

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
        or data.get("question")
        or ""
    )

    if not message:
        return JSONResponse({"error": "No message received"}, status_code=400)

    # Process using AI Engine
    try:
        result = engine.process(user_id, message)
    except Exception as e:
        return JSONResponse({"error": "AI Engine Error", "detail": str(e)}, status_code=500)

    # -------- Update Analytics --------
    with analytics_lock:
        analytics["total_requests"] += 1

        intent = result["intent"]
        emotion = result["emotion"]
        priority = result["priority"]

        analytics["intent_counts"][intent] = analytics["intent_counts"].get(intent, 0) + 1
        analytics["emotion_counts"][emotion] = analytics["emotion_counts"].get(emotion, 0) + 1
        analytics["priority_counts"][priority] += 1

        # Escalations (from engine memory)
        analytics["escalations_total"] += engine.memory.data[user_id]["escalations"]

        # Store log
        analytics["chat_log"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "message": message,
            "response": result["final_answer"],
            "intent": intent,
            "emotion": emotion,
            "priority": priority
        })

        # Only keep last 150 chats
        analytics["chat_log"] = analytics["chat_log"][-150:]

        analytics["last_updated"] = datetime.utcnow().isoformat()
        save_analytics()

    return {
        "response": result["final_answer"],
        "intent": intent,
        "emotion": emotion,
        "metadata": result["metadata"],
        "engine_raw": result
    }

# =====================================================
#                     ORDER SYSTEM
# =====================================================

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
    """Generate consistent fake order data"""
    random.seed(int(oid) % 9999)
    stage_index = random.randint(0, len(ORDER_STAGES) - 1)

    stage = ORDER_STAGES[stage_index]
    eta = max(0, 5 - stage_index)

    return {
        "order_id": oid,
        "stage": stage,
        "eta_days": eta,
        "status": "Delivered" if stage == "Delivered" else "In Progress",
        "history": ORDER_STAGES[: stage_index + 1]
    }

@app.get("/order")
async def order_lookup(oid: str):
    if not oid:
        return {"error": "Missing order ID"}

    data = simulate_order(oid)

    # update analytics
    with analytics_lock:
        analytics["intent_counts"]["order_lookup"] = analytics["intent_counts"].get("order_lookup", 0) + 1
        analytics["total_requests"] += 1
        save_analytics()

    return data

# =====================================================
#                     ANALYTICS
# =====================================================

@app.get("/analytics")
async def get_analytics():
    return analytics

@app.get("/analytics/csv")
async def analytics_csv():
    if not analytics["chat_log"]:
        return {"error": "No data yet"}

    df = pd.DataFrame(analytics["chat_log"])
    csv_path = "analytics_export.csv"
    df.to_csv(csv_path, index=False)

    return FileResponse(csv_path, media_type="text/csv", filename="analytics.csv")

@app.post("/analytics/reset")
async def reset_analytics():
    global analytics
    analytics = load_analytics()
    save_analytics()
    return {"status": "ok", "message": "Analytics reset"}

# =====================================================
#                     HEALTH
# =====================================================

@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": True}

# Local only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fixed:app", host=HOST, port=PORT, reload=True)
