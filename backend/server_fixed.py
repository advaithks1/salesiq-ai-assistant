# server_fixed.py — PREMIUM backend (analytics + order + emotion) (final)
import os
import re
import json
import threading
import random
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from ai_engine_step5 import AIEngine, USE_HF

# -------------------------
# Config / Paths
# -------------------------
ANALYTICS_FILE = "analytics.json"
LOG_FILE = "chat_logs.jsonl"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# -------------------------
# App & CORS
# -------------------------
app = FastAPI(title="SalesIQ AI Engine — PREMIUM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Engine
# -------------------------
engine = AIEngine()

# -------------------------
# Analytics (thread-safe)
# -------------------------
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

def load_analytics() -> Dict[str, Any]:
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_analytics()

analytics = load_analytics()

def save_analytics():
    try:
        with open(ANALYTICS_FILE, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Error saving analytics:", e)

# -------------------------
# Health
# -------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_loaded": True,
        "use_hf": bool(USE_HF),
        "time": datetime.utcnow().isoformat()
    }

# -------------------------
# Chat endpoints
# -------------------------
@app.get("/chat")
async def chat_get():
    return {"status": "ok", "detail": "Use POST /chat to interact"}

@app.post("/chat")
async def chat_post(request: Request):
    try:
        data = await request.json()
    except Exception:
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

    # Process through engine with safe error handling
    try:
        result = engine.process(user_id, message)
    except Exception as e:
        # Log exception server-side and return friendly error for SalesIQ
        print("AI Engine Error:", str(e))
        return JSONResponse({"error": "AI Engine Error", "detail": "Internal processing error"}, status_code=500)

    # Update analytics (thread-safe)
    try:
        with analytics_lock:
            analytics["total_requests"] = analytics.get("total_requests", 0) + 1
            intent = result.get("intent", "unknown")
            emotion = result.get("emotion", "neutral")
            priority = result.get("priority", "medium")

            analytics["intent_counts"][intent] = analytics["intent_counts"].get(intent, 0) + 1
            analytics["emotion_counts"][emotion] = analytics["emotion_counts"].get(emotion, 0) + 1
            analytics["priority_counts"][priority] = analytics["priority_counts"].get(priority, 0) + 1

            # Keep global escalations count
            try:
                analytics["escalations_total"] = analytics.get("escalations_total", 0) + engine.memory.get(user_id)["escalations"]
            except Exception:
                pass

            q = result.get("matched_question") or message
            analytics["top_questions"][q] = analytics["top_questions"].get(q, 0) + 1

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
    except Exception as e:
        print("Analytics update error:", e)

    # Return structured result
    return {
        "response": result.get("final_answer"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "metadata": result.get("metadata", {}),
        "engine_raw": result
    }

# -------------------------
# Order system (simulated)
# -------------------------
def simulate_order(oid: str) -> Dict[str, Any]:
    if not oid:
        oid = "0"
    try:
        seed = int(re.sub(r'\D', '', oid) or 0) % 9999
    except Exception:
        seed = sum(ord(c) for c in oid) % 9999
    random.seed(seed)
    stage_index = random.randint(0, len(ORDER_STAGES) - 1) if 'ORDER_STAGES' in globals() else random.randint(0, 6)
    ORDER_STAGES_LOCAL = [
        "Order confirmed", "Packing", "Ready to ship", "Shipped",
        "In transit", "Out for delivery", "Delivered"
    ]
    stage = ORDER_STAGES_LOCAL[stage_index]
    eta = max(0, 5 - stage_index)
    return {
        "order_id": oid,
        "stage": stage,
        "eta_days": eta,
        "status": "Delivered" if stage == "Delivered" else "In Progress",
        "history": ORDER_STAGES_LOCAL[:stage_index + 1]
    }

@app.get("/order")
async def order_lookup(oid: str = ""):
    if not oid:
        return {"error": "Missing order ID"}
    data = simulate_order(oid)
    # analytics
    try:
        with analytics_lock:
            analytics["intent_counts"]["order_lookup"] = analytics["intent_counts"].get("order_lookup", 0) + 1
            analytics["total_requests"] = analytics.get("total_requests", 0) + 1
            analytics["last_updated"] = datetime.utcnow().isoformat()
            save_analytics()
    except Exception:
        pass
    return data

# -------------------------
# Analytics endpoints
# -------------------------
@app.get("/analytics")
async def get_analytics():
    return analytics

@app.get("/analytics/csv")
async def analytics_csv():
    if not analytics.get("chat_log"):
        return {"error": "No data yet"}
    try:
        df = pd.DataFrame(analytics["chat_log"])
        csv_path = "analytics_export.csv"
        df.to_csv(csv_path, index=False)
        return FileResponse(csv_path, media_type="text/csv", filename="analytics.csv")
    except Exception as e:
        return JSONResponse({"error": "Failed to generate CSV", "detail": str(e)}, status_code=500)

@app.post("/analytics/reset")
async def reset_analytics():
    global analytics
    analytics = default_analytics()
    save_analytics()
    return {"status": "ok", "message": "Analytics reset"}

# -------------------------
# Reset memory endpoint
# -------------------------
@app.post("/reset")
async def reset_memory():
    try:
        engine.memory.clear()
    except Exception:
        pass
    return {"status": "ok", "message": "Memory cleared"}

# -------------------------
# Local dev runner
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, reload=True)
