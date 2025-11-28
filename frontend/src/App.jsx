// src/App.jsx
import React, { useEffect, useState } from "react";
import "./index.css";

const BACKEND_URL = "https://salesiq-ai-assistant-8m8r.onrender.com";

function Badge({ children, color = "#2563eb" }) {
  return (
    <span className="badge" style={{ background: color }}>
      {children}
    </span>
  );
}

function SmallStat({ title, value }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-title">{title}</div>
    </div>
  );
}

export default function App() {
  // Order
  const [orderId, setOrderId] = useState("101");
  const [order, setOrder] = useState(null);
  const [loadingOrder, setLoadingOrder] = useState(false);

  // AI
  const [aiInput, setAiInput] = useState("");
  const [aiRes, setAiRes] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);

  // Analytics
  const [analytics, setAnalytics] = useState(null);
  const [showAnalytics, setShowAnalytics] = useState(false);

  // UI microstate
  const [toast, setToast] = useState(null);

  useEffect(() => {
    // initial analytics fetch (light)
    fetchAnalyticsSafe();
  }, []);

  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(null), 3000);
      return () => clearTimeout(t);
    }
  }, [toast]);

  async function fetchOrder(oid) {
    if (!oid) {
      setToast("Enter an order id");
      return;
    }
    setLoadingOrder(true);
    try {
      const r = await fetch(`${BACKEND_URL}/order?oid=${encodeURIComponent(oid)}`);
      const j = await r.json();
      setOrder(j);
      setToast("Order loaded");
    } catch (e) {
      setToast("Failed to reach backend");
    } finally {
      setLoadingOrder(false);
    }
  }

  async function askAi() {
    if (!aiInput || aiInput.trim().length < 1) {
      setToast("Type a question for the AI");
      return;
    }
    setAiLoading(true);
    try {
      const r = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "demo_user", message: aiInput }),
      });
      const j = await r.json();
      setAiRes(j);
      setAiInput("");
      setToast("AI replied");
      // optionally fetch analytics
      fetchAnalyticsSafe();
    } catch (e) {
      setToast("AI unavailable");
    } finally {
      setAiLoading(false);
    }
  }

  async function fetchAnalyticsSafe() {
    try {
      const r = await fetch(`${BACKEND_URL}/analytics`);
      if (!r.ok) return;
      const j = await r.json();
      setAnalytics(j);
    } catch (e) {
      // ignore
    }
  }

  // small helpers for UI
  const stageIndex = (s) => {
    const list = [
      "Order confirmed",
      "Packing",
      "Ready to ship",
      "Shipped",
      "In transit",
      "Out for delivery",
      "Delivered",
    ];
    return Math.max(0, list.indexOf(s));
  };

  const progressPercent = (stage) => {
    const idx = stageIndex(stage);
    if (idx < 0) return 0;
    const percent = Math.round(((idx + 1) / 7) * 100);
    return percent;
  };

  const emotionColor = (e) => {
    if (!e) return "#6b7280";
    if (e === "angry") return "#ef4444";
    if (e === "sad") return "#6366f1";
    if (e === "happy") return "#10b981";
    if (e === "confused") return "#f59e0b";
    return "#6b7280";
  };

  // tiny polished demo action
  const handleQuickTrack = (id) => {
    setOrderId(id);
    fetchOrder(id);
  };

  return (
    <div className="premium-root">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">AI</div>
          <div className="brand-title">Smart Support & Order Tracker</div>
        </div>
        <div className="top-actions">
          <SmallStat title="Requests" value={analytics?.total_requests ?? "—"} />
          <SmallStat title="Escalations" value={analytics?.escalations_total ?? "—"} />
        </div>
      </header>

      <main className="grid">
        {/* Left column: Order tracking */}
        <section className="panel">
          <div className="panel-head">
            <h2>📦 Order Tracker</h2>
            <div className="muted">Fast lookup • Demo mode</div>
          </div>

          <div className="control-row">
            <input
              className="input"
              placeholder="Enter order id (e.g. 101)"
              value={orderId}
              onChange={(e) => setOrderId(e.target.value)}
            />
            <button className="btn" onClick={() => fetchOrder(orderId)}>
              {loadingOrder ? "Loading…" : "Track"}
            </button>
          </div>

          <div className="quick-row">
            <button className="chip" onClick={() => handleQuickTrack("101")}>Try 101</button>
            <button className="chip" onClick={() => handleQuickTrack("105")}>Try 105</button>
            <button className="chip" onClick={() => handleQuickTrack("103")}>Try 103</button>
          </div>

          {order ? (
            <div className="order-card">
              <div className="order-top">
                <div className="order-left">
                  <div className="order-id">Order #{order.order_id}</div>
                  <div className="order-stage" style={{ color: emotionColor(aiRes?.emotion) }}>
                    {order.stage}
                  </div>
                  <div className="muted">ETA: {order.eta_days} day(s)</div>
                </div>
                <div className="order-right">
                  <Badge color="#e0f2fe">{order.status ?? order.stage}</Badge>
                </div>
              </div>

              <div className="progress">
                <div className="progress-bar" style={{ width: `${progressPercent(order.stage)}%` }} />
                <div className="progress-label">{progressPercent(order.stage)}%</div>
              </div>

              <ol className="timeline">
                {order.history.map((h, i) => {
                  const done = stageIndex(order.stage) >= i;
                  return (
                    <li key={i} className={done ? "done" : ""}>
                      <span className="dot" />
                      <div className="event">{h}</div>
                    </li>
                  );
                })}
              </ol>
            </div>
          ) : (
            <div className="placeholder">No order loaded. Try the sample buttons.</div>
          )}
        </section>

        {/* Right column: AI and analytics */}
        <aside className="panel side">
          <div className="panel-head">
            <h2>🤖 AI Assistant</h2>
            <div className="muted">Natural language, emotion & intent</div>
          </div>

          <div className="ai-box">
            <textarea
              className="ai-input"
              placeholder="Ask the assistant (e.g., track 101)"
              value={aiInput}
              onChange={(e) => setAiInput(e.target.value)}
            />
            <div className="control-row">
              <button className="btn ghost" onClick={() => { setAiInput("track " + (orderId || "101")); askAi(); }}>
                Quick: Track
              </button>
              <button className="btn" onClick={askAi}>{aiLoading ? "Thinking…" : "Ask AI"}</button>
            </div>

            <div className="ai-result">
              {aiRes ? (
                <>
                  <div className="ai-row">
                    <div className="ai-text">{aiRes.response}</div>
                  </div>

                  <div className="ai-meta">
                    <div className="meta-item">
                      <div className="meta-title">Intent</div>
                      <div className="meta-value">{aiRes.intent}</div>
                    </div>
                    <div className="meta-item">
                      <div className="meta-title">Emotion</div>
                      <div className="meta-value" style={{ color: emotionColor(aiRes.emotion) }}>{aiRes.emotion ?? "neutral"}</div>
                    </div>
                    <div className="meta-item">
                      <div className="meta-title">Priority</div>
                      <div className="meta-value">{aiRes.engine_raw?.priority ?? "—"}</div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="muted">Ask anything — AI will analyze intent & emotion.</div>
              )}
            </div>
          </div>

          <div className="analytics-compact">
            <div className="analytics-head">
              <h3>📊 Analytics</h3>
              <button className="link-btn" onClick={() => { setShowAnalytics(!showAnalytics); if (!analytics) fetchAnalyticsSafe(); }}>
                {showAnalytics ? "Hide" : "Show"}
              </button>
            </div>

            {showAnalytics && analytics ? (
              <div className="analytics-inner">
                <div className="mini-grid">
                  <div>
                    <SmallStat title="Total" value={analytics.total_requests ?? 0} />
                  </div>
                  <div>
                    <SmallStat title="Escalations" value={analytics.escalations_total ?? 0} />
                  </div>
                </div>

                <div className="chart">
                  <div className="chart-title">Intent</div>
                  <div className="bars">
                    {Object.entries(analytics.intent_counts || {}).slice(0,6).map(([k, v]) => {
                      const max = Math.max(...Object.values(analytics.intent_counts || {unknown:1}));
                      const width = Math.round((v / (max || 1)) * 100);
                      return (
                        <div key={k} className="bar-row">
                          <div className="bar-label">{k}</div>
                          <div className="bar-track"><div className="bar-fill" style={{width: `${width}%`}} /></div>
                          <div className="bar-value">{v}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="chart">
                  <div className="chart-title">Emotion</div>
                  <div className="bars">
                    {Object.entries(analytics.emotion_counts || {}).map(([k, v]) => {
                      const max = Math.max(...Object.values(analytics.emotion_counts || {neutral:1}));
                      const width = Math.round((v / (max || 1)) * 100);
                      return (
                        <div key={k} className="bar-row">
                          <div className="bar-label">{k}</div>
                          <div className="bar-track"><div className="bar-fill" style={{width: `${width}%`, background: emotionColor(k)}} /></div>
                          <div className="bar-value">{v}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="muted" style={{ marginTop: 10 }}>Analytics snapshot — click Show.</div>
            )}
          </div>
        </aside>
      </main>

      <footer className="footer">
        <div>Powered by your Team OHM</div>
              </footer>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
