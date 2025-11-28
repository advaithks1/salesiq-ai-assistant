import React, { useState } from "react";
import "./index.css";

const BACKEND_URL = "https://salesiq-ai-assistant-8m8r.onrender.com";

export default function App() {
  const [orderId, setOrderId] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [aiMsg, setAiMsg] = useState("");
  const [aiData, setAiData] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);

  const [showAnalytics, setShowAnalytics] = useState(false);
  const [analytics, setAnalytics] = useState(null);

  // -----------------------------------------------------
  // 📦 ORDER TRACKING
  // -----------------------------------------------------
  const trackOrder = async () => {
    if (!orderId.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/order?oid=${orderId}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setResult({ error: "Server unavailable" });
    }
    setLoading(false);
  };

  // -----------------------------------------------------
  // 🤖 AI ENGINE
  // -----------------------------------------------------
  const askAI = async () => {
    if (!aiMsg.trim()) return;
    setAiLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "demo_user", message: aiMsg }),
      });
      const data = await res.json();
      setAiData(data);
    } catch (e) {
      setAiData({ error: "AI Engine unavailable" });
    }
    setAiLoading(false);
  };

  // -----------------------------------------------------
  // 📊 ANALYTICS
  // -----------------------------------------------------
  const loadAnalytics = async () => {
    setShowAnalytics(true);
    try {
      const res = await fetch(`${BACKEND_URL}/analytics`);
      const data = await res.json();
      setAnalytics(data);
    } catch (e) {
      setAnalytics({ error: "Analytics unavailable" });
    }
  };

  // -----------------------------------------------------
  // UI STARTS HERE
  // -----------------------------------------------------
  return (
    <div className="app-container">
      <h1 className="app-title">🚀 Smart Support & Order Tracking Dashboard</h1>
      <p className="subtitle">Powered by AI Engine + Order Simulator + Analytics</p>

      <div className="main-grid">
        {/* ------------------------------------
            📦 ORDER TRACKING SECTION
          ------------------------------------ */}
        <div className="card">
          <h2>📦 Smart Order Tracker</h2>
          <input
            className="input"
            placeholder="Enter Order ID"
            value={orderId}
            onChange={(e) => setOrderId(e.target.value)}
          />
          <button className="btn" onClick={trackOrder}>
            {loading ? "Loading..." : "Track Order"}
          </button>

          {result && (
            <div className="result-box">
              {result.error ? (
                <p className="error">{result.error}</p>
              ) : (
                <>
                  <p><b>Order ID:</b> {result.order_id}</p>
                  <p><b>Status:</b> {result.stage}</p>
                  <p><b>ETA:</b> {result.eta_days} day(s)</p>

                  <h4>📜 History</h4>
                  <ul className="timeline">
                    {result.history.map((h, i) => (
                      <li key={i}>{h}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}
        </div>

        {/* ------------------------------------
            🤖 AI PANEL
          ------------------------------------ */}
        <div className="card">
          <h2>🤖 AI Assistant</h2>

          <input
            className="input"
            placeholder="Ask anything…"
            value={aiMsg}
            onChange={(e) => setAiMsg(e.target.value)}
          />
          <button className="btn" onClick={askAI}>
            {aiLoading ? "Thinking..." : "Ask AI"}
          </button>

          {aiData && (
            <div className="result-box">
              <p><b>AI:</b> {aiData.response}</p>
              <hr />
              <p><b>Intent:</b> {aiData.intent}</p>
              <p><b>Emotion:</b> {aiData.emotion}</p>
              <p><b>Priority:</b> {aiData.engine_raw.priority}</p>
            </div>
          )}
        </div>
      </div>

      {/* ------------------------------------
          📊 ANALYTICS SECTION
        ------------------------------------ */}
      <div className="analytics-container">
        <button className="btn-analytics" onClick={loadAnalytics}>
          📊 Show Analytics
        </button>

        {showAnalytics && analytics && (
          <div className="analytics-box">
            <h2>📈 AI Analytics</h2>
            <p>Total Requests: {analytics.total_requests}</p>

            <h4>Intent Breakdown</h4>
            <ul>
              {Object.entries(analytics.intent_counts).map(([k, v]) => (
                <li key={k}>
                  {k}: <b>{v}</b>
                </li>
              ))}
            </ul>

            <h4>Emotion Stats</h4>
            <ul>
              {Object.entries(analytics.emotion_counts).map(([k, v]) => (
                <li key={k}>
                  {k}: <b>{v}</b>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
