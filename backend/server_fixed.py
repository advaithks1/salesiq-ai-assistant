# server_fixed.py — FastAPI wrapper (final)
import os
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading

from ai_engine_step5 import AIEngine

LOG_FILE = "chat_logs.jsonl"
ANALYTICS_PERSIST = "analytics.json"
HOST = "0.0.0.0"
PORT = 8000

app = FastAPI(title="SalesIQ AI Engine (final)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    matched_question: Optional[str]
    intent: Optional[str]
    emotion: Optional[str]
    priority: Optional[str]
    missing_info: Optional[str]
    escalations: Optional[int]
    metadata: dict

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
if os.path.exists(ANALYTICS_PERSIST):
    try:
        with open(ANALYTICS_PERSIST, "r", encoding="utf-8") as f:
            analytics.update(json.load(f))
    except Exception:
        pass

log_lock = threading.Lock()
def append_log(record: dict):
    try:
        with log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def persist_analytics():
    with analytics_lock:
        analytics["last_updated"] = datetime.utcnow().isoformat() + "Z"
        try:
            with open(ANALYTICS_PERSIST, "w", encoding="utf-8") as f:
                json.dump(analytics, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

def background_log(user_id: str, req: ChatRequest, resp: dict):
    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "request": req.dict(),
        "response": resp
    }
    append_log(rec)
    persist_analytics()

def update_analytics(engine_result: dict):
    with analytics_lock:
        analytics["total_chats"] += 1
        intent = (engine_result.get("intent") or "unknown").lower()
        emotion = (engine_result.get("emotion") or "unknown").lower()
        priority = (engine_result.get("priority") or "low").lower()
        escal = int(engine_result.get("escalations") or 0)
        analytics["intent_counts"][intent] = analytics["intent_counts"].get(intent, 0) + 1
        analytics["emotion_counts"][emotion] = analytics["emotion_counts"].get(emotion, 0) + 1
        analytics["priority_counts"][priority] = analytics["priority_counts"].get(priority, 0) + 1
        analytics["escalations_total"] += escal

def _safe_num(x, fallback=0.0):
    try:
        return float(x)
    except Exception:
        return float(fallback)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    if not req.message or not req.user_id:
        raise HTTPException(status_code=400, detail="user_id and message required")

    try:
        result = engine.process(req.user_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"engine error: {e}")

    meta = result.get("metadata") or {}

    # Merge: engine-level values take precedence; metadata supplements
    merged_meta = {
        "confidence": _safe_num(result.get("confidence") or meta.get("confidence") or 0.0),
        "similarity": _safe_num(meta.get("similarity") or result.get("similarity") or 0.0),
        "risk": result.get("risk") or meta.get("risk") or "low",
        "autoflow": bool(meta.get("autoflow")) if meta.get("autoflow") is not None else False,
        "field_required": meta.get("field_required") or result.get("missing_info") or None,
        "hint": meta.get("hint") or None,
        "order_id": meta.get("order_id") or result.get("order_id") or None,
        "product_name": meta.get("product_name") or result.get("product_name") or None,
        "plan_type": meta.get("plan_type") or result.get("plan_type") or None,
        "availability": meta.get("availability") or None
    }

    payload = {
        "answer": result.get("final_answer") or result.get("answer") or "",
        "matched_question": result.get("matched_question"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "priority": result.get("priority"),
        "missing_info": result.get("missing_info"),
        "escalations": int(result.get("escalations") or 0),
        "metadata": merged_meta
    }

    update_analytics(payload)
    background_tasks.add_task(background_log, req.user_id, req, payload)
    return payload

@app.post("/reset")
async def reset(user_id: Optional[str] = None):
    try:
        if user_id:
            if hasattr(engine, "memory"):
                engine.memory.data.pop(user_id, None)
            return {"status": "ok", "message": f"memory for {user_id} cleared"}
        else:
            if hasattr(engine, "memory"):
                engine.memory.data.clear()
            with analytics_lock:
                analytics.update({
                    "total_chats": 0,
                    "intent_counts": {},
                    "emotion_counts": {},
                    "priority_counts": {"high": 0, "medium": 0, "low": 0},
                    "escalations_total": 0,
                    "last_updated": None
                })
            persist_analytics()
            return {"status": "ok", "message": "all memory and analytics cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    with analytics_lock:
        return analytics

@app.get("/health")
async def health():
    engine_loaded = bool(getattr(engine, "model", None))
    return {"status": "ok", "engine_loaded": engine_loaded, "time": datetime.utcnow().isoformat() + "Z"}

if __name__ == "__main__":
    import uvicorn
    print("Starting SalesIQ AI Engine server (final)...")
    uvicorn.run("server_fixed:app", host=HOST, port=PORT, reload=True)
