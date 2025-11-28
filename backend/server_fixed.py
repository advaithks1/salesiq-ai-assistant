# server_fixed.py — FINAL HACKATHON BACKEND (FastAPI)
"""
FastAPI server for SalesIQ Hackathon.
- Exposes /chat and /order
- Accepts JSON (and will gracefully return 400 on invalid)
- Returns safe JSON (no emojis, stable metadata)
- Minimal analytics (optional)
- Designed to work with ai_engine_step5.py
"""

import os
import json
import random
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# import local engine (ai_engine_step5.py)
from ai_engine_step5 import AIEngine

# -------- config ----------
PORT = int(os.environ.get("PORT", 8000))
ANALYTICS_FILE = "analytics.json"

# -------- app ----------
app = FastAPI(title="SalesIQ AI Backend (Hackathon)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AIEngine()

# -------- analytics (lightweight) ----------
def _default_analytics() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "intent_counts": {},
        "emotion_counts": {},
        "last_updated": None,
        "chat_log": []
    }

def _load_analytics():
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return _default_analytics()

analytics = _load_analytics()

def _save_analytics():
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics, f, indent=2)
    except Exception as e:
        print("Failed saving analytics:", e)

# -------- health ----------
@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": True, "time": datetime.utcnow().isoformat()}

# -------- chat endpoint ----------
@app.post("/chat")
async def chat_post(request: Request):
    """
    Expects JSON body:
    {
      "user_id": "visitor123",
      "message": "track 101"
    }
    Returns:
    {
      "response": "<final_answer>",
      "intent": "...",
      "emotion": "...",
      "metadata": {...},
      "engine_raw": {...}
    }
    """
    # parse JSON
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON body")
    except Exception:
        # return helpful error for debugging in dev; SalesIQ should always send valid JSON
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_id = data.get("user_id") or data.get("visitor_id") or "visitor"
    message = (data.get("message") or data.get("text") or data.get("query") or "").strip()

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # process via engine
    try:
        result = engine.process(user_id, message)
    except Exception as e:
        # return safe structured error for SalesIQ to handle
        return JSONResponse({"error": "AI Engine Error", "detail": str(e)}, status_code=500)

    # update analytics (best-effort)
    try:
        analytics["total_requests"] = analytics.get("total_requests", 0) + 1
        intent = result.get("intent", "unknown")
        emotion = result.get("emotion", "neutral")
        analytics["intent_counts"][intent] = analytics["intent_counts"].get(intent, 0) + 1
        analytics["emotion_counts"][emotion] = analytics["emotion_counts"].get(emotion, 0) + 1

        # chat log (keep last 500)
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "message": message,
            "response": result.get("final_answer"),
            "intent": intent,
            "emotion": emotion,
            "metadata": result.get("metadata", {})
        }
        analytics["chat_log"].append(log_entry)
        analytics["chat_log"] = analytics["chat_log"][-500:]
        analytics["last_updated"] = datetime.utcnow().isoformat()
        _save_analytics()
    except Exception as _e:
        # don't fail the request because analytics failed
        print("Analytics error:", _e)

    # ensure response values are JSON serializable and safe (avoid non-serializable objects)
    safe_result = {
        "response": result.get("final_answer"),
        "intent": result.get("intent"),
        "emotion": result.get("emotion"),
        "metadata": result.get("metadata", {}),
        "engine_raw": result  # engine returns dicts/strings only
    }
    return safe_result

# -------- order endpoint ----------
ORDER_STAGES = [
    "Order confirmed", "Packing", "Ready to ship", "Shipped",
    "In transit", "Out for delivery", "Delivered"
]

@app.get("/order")
async def order_lookup(oid: str = ""):
    """
    Simple deterministic order simulation for demo.
    /order?oid=101
    """
    if not oid:
        return JSONResponse({"error": "Missing order id"}, status_code=400)

    try:
        seed = int("".join(filter(str.isdigit, oid))) % 9999
    except:
        seed = sum(ord(c) for c in oid) % 9999

    random.seed(seed)
    idx = random.randint(0, len(ORDER_STAGES) - 1)
    stage = ORDER_STAGES[idx]
    eta = max(0, 5 - idx)

    resp = {
        "order_id": oid,
        "stage": stage,
        "eta_days": eta,
        "history": ORDER_STAGES[:idx + 1]
    }

    # minimal analytics update
    try:
        analytics["total_requests"] = analytics.get("total_requests", 0) + 1
        analytics["intent_counts"]["order_lookup"] = analytics["intent_counts"].get("order_lookup", 0) + 1
        analytics["last_updated"] = datetime.utcnow().isoformat()
        _save_analytics()
    except:
        pass

    return resp

# -------- analytics endpoints (optional) ----------
@app.get("/analytics")
async def get_analytics():
    return analytics

@app.post("/analytics/reset")
async def reset_analytics():
    global analytics
    analytics = _default_analytics()
    _save_analytics()
    return {"status": "ok", "message": "analytics reset"}
@app.post("/agent_assist")
async def agent_assist(request: Request):
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    mem = engine.memory.get(user_id)

    # last 3 messages
    ctx = list(mem["context"])[-3:]

    # last known info
    last_intent = mem["last_intent"]
    escalations = mem["escalations"]

    # build summary
    conversation_summary = ""
    for item in ctx:
        conversation_summary += f"{item['speaker'].upper()}: {item['text']}\n"

    # suggestions
    suggestions = []

    if last_intent == "order":
        suggestions.append("Provide detailed delivery timeline")
        suggestions.append("Ask if user wants SMS/Email alerts")
    elif last_intent == "pricing":
        suggestions.append("Explain feature differences")
        suggestions.append("Offer discount or coupon")
    elif last_intent == "support":
        suggestions.append("Request screenshot or error code")
        suggestions.append("Guide through basic troubleshooting")
    else:
        suggestions.append("Ask the user to clarify their issue")

    # recommended agent action
    recommended_action = "Respond politely and ask for more details"

    if last_intent == "support":
        recommended_action = "Collect error details, device info, steps to reproduce"
    if last_intent == "escalate":
        recommended_action = "Take over actively and reassure the user"
    if escalations > 1:
        recommended_action = "Handle carefully — user is frustrated"

    return {
        "summary": conversation_summary,
        "last_intent": last_intent,
        "escalations": escalations,
        "suggestions": suggestions,
        "recommended_action": recommended_action,
        "recent_messages": ctx
    }
@app.get("/agent_assist")
async def agent_assist(uid: str):
    """
    Provide real-time agent assistance based on memory + analytics.
    """
    if uid not in engine.memory.data:
        return {
            "error": "Unknown user",
            "summary": "",
            "suggested_reply": "Hello! How may I help you today?",
            "history": [],
            "emotion": "neutral",
            "priority": "medium",
            "frustration_score": 0.0,
            "last_intent": "unknown"
        }

    mem = engine.memory.data[uid]

    # ---------------------------
    # 1. Conversation History
    # ---------------------------
    history = [x["text"] for x in mem["context"] if x["speaker"] == "user"]

    # ---------------------------
    # 2. Last Intent
    # ---------------------------
    last_intent = mem.get("last_intent", "unknown")

    # ---------------------------
    # 3. Frustration Score
    # ---------------------------
    frustration_words = ["not working", "angry", "hate", "bad", "error", "slow", "wtf", "no help"]
    frustration_score = 0.0
    for h in history[-6:]:     # last 6 messages
        for w in frustration_words:
            if w in h.lower():
                frustration_score += 0.2
    frustration_score = min(frustration_score, 1.0)

    # ---------------------------
    # 4. Emotion Trend
    # ---------------------------
    emotion = "neutral"
    if frustration_score >= 0.6:
        emotion = "angry"
    elif frustration_score >= 0.3:
        emotion = "confused"

    # ---------------------------
    # 5. Priority Level
    # ---------------------------
    if emotion == "angry":
        priority = "high"
    elif emotion == "confused":
        priority = "medium"
    else:
        priority = "low"

    # ---------------------------
    # 6. Conversation Summary
    # ---------------------------
    if len(history) == 0:
        summary = "User just started the chat."
    else:
        summary = (
            f"User asked about: {history[-1]}. "
            f"Recent intents show interest in {last_intent}. "
            f"Overall emotion seems {emotion}."
        )

    # ---------------------------
    # 7. Suggested Replies
    # ---------------------------
    suggestions = {
        "order": "I can help track your order. Could you share the Order ID?",
        "pricing": "We offer Basic, Pro, and Enterprise plans. Would you like details?",
        "support": "Can you share more information or a screenshot of the issue?",
        "login": "Please try resetting your password using the Forgot Password option.",
        "escalate": "Connecting you to a human agent now.",
        "unknown": "Can you please clarify what you need help with?",
    }
    suggested_reply = suggestions.get(last_intent, "How may I assist you?")

    # final result
    return {
        "user_id": uid,
        "summary": summary,
        "last_intent": last_intent,
        "emotion": emotion,
        "priority": priority,
        "frustration_score": round(frustration_score, 2),
        "suggested_reply": suggested_reply,
        "history": history[-10:]  # last 10 messages
    }


# -------- run locally ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fixed:app", host="0.0.0.0", port=PORT, reload=True)
