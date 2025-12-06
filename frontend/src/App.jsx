import React, { useEffect, useState } from "react";
import "./index.css";

// Backend shared by website + chatbot
const BACKEND_URL =
  import.meta.env?.VITE_API_BASE_URL ||
  "https://salesiq-ai-assistant-8m8r.onrender.com";

function App() {
  // ----------------------------
  // HEALTH
  // ----------------------------
  const [health, setHealth] = useState({ status: "checking..." });

  // ----------------------------
  // PRODUCTS (from /products)
  // ----------------------------
  const [products, setProducts] = useState([]);
  const [productsLoading, setProductsLoading] = useState(true);
  const [productsError, setProductsError] = useState("");

  // ----------------------------
  // ORDER TRACKER (from /order)
  // ----------------------------
  const [orderId, setOrderId] = useState("");
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState("");
  const [orderData, setOrderData] = useState(null);

  // ----------------------------
  // CHAT / AGENT ASSIST DEMO (from /chat)
  // ----------------------------
  const [chatInput, setChatInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [botResponse, setBotResponse] = useState("");
  const [intent, setIntent] = useState("");
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [agentAssist, setAgentAssist] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [chatError, setChatError] = useState("");

  // Fetch backend health + products on load
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`);
        const data = await res.json();
        setHealth(data);
      } catch {
        setHealth({ status: "error", time: "-", service: "unreachable" });
      }
    };

    const fetchProducts = async () => {
      try {
        setProductsLoading(true);
        setProductsError("");
        const res = await fetch(`${BACKEND_URL}/products`);
        const data = await res.json();
        if (Array.isArray(data.products)) {
          setProducts(data.products);
        } else {
          setProductsError("Product list not available.");
        }
      } catch {
        setProductsError("Failed to load products.");
      } finally {
        setProductsLoading(false);
      }
    };

    fetchHealth();
    fetchProducts();
  }, []);

  // Handle order tracking
  const handleTrackOrder = async (e) => {
    e.preventDefault();
    const clean = (orderId || "").replace(/[^0-9]/g, "");

    if (!clean) {
      setOrderError("Please enter a numeric Order ID (e.g., 101).");
      setOrderData(null);
      return;
    }

    setOrderLoading(true);
    setOrderError("");
    setOrderData(null);

    try {
      const res = await fetch(`${BACKEND_URL}/order?oid=${clean}`);
      const data = await res.json();
      if (data.error) {
        setOrderError(data.error.toString());
      } else {
        setOrderData(data);
      }
    } catch {
      setOrderError("Failed to fetch order status.");
    } finally {
      setOrderLoading(false);
    }
  };

  // Handle chat → /chat
  const handleSendChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    setIsSending(true);
    setChatError("");
    setBotResponse("");
    setAgentAssist(null);
    setMetadata(null);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "demo-web-user",
          message: chatInput.trim(),
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || err.error || "Backend error");
      }

      const data = await res.json();
      setBotResponse(data.response || "");
      setIntent(data.intent || "");
      setEmotion(data.emotion || "");
      setConfidence(
        typeof data.confidence === "number" ? data.confidence : null
      );
      setAgentAssist(data.agent_assist || null);
      setMetadata(data.metadata || null);
    } catch (err) {
      setChatError(err.message || "Something went wrong");
    } finally {
      setIsSending(false);
    }
  };

  const handleQuickMessage = (msg) => {
    setChatInput(msg);
  };

  return (
    <div className="app-root">
      {/* NAVBAR */}
      <header className="nav">
        <div className="nav-left">
          <span className="brand">Smart AI Store</span>
          <span className="brand-sub">
            Powered by Smart AI Assistant (Zoho SalesIQ)
          </span>
        </div>
        <div className="nav-right">
          <a href="#top">Dashboard</a>
          <a href="#products">Products</a>
          <a href="#order-tracker">Order Tracker</a>
          <a href="#agent-assist">Bot Tester</a>
          <a href="#evaluators">For Evaluators</a>
        </div>
      </header>

      <main className="main">
        {/* HERO + HEALTH */}
        <section id="top" className="hero">
          <div className="hero-text">
            <div className="pill">CliqTrix&apos;26 – Smart AI Assistant</div>

            <h1>AI-Powered E-Commerce Support & Order Tracking</h1>

            <p>
              This single-page demo connects a custom AI backend, an
              e-commerce-style catalog, and a Zoho SalesIQ chatbot. The same
              IDs and endpoints are used by both this page and the chatbot.
            </p>

            <p className="hero-note">
              <span className="dot" /> The chatbot at the bottom-right can{" "}
              <b>track orders</b>, <b>manage cart & orders</b>,{" "}
              <b>show products</b>, and <b>analyze sentiment</b> using the same
              backend as this website.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>1. Browse Products</h3>
                <p>Products below are loaded live from the backend.</p>
              </div>
              <div className="hero-card">
                <h3>2. Track Orders</h3>
                <p>
                  Use the website order tracker or type <code>track 101</code> in
                  chat.
                </p>
              </div>
              <div className="hero-card">
                <h3>3. E-Commerce Bot</h3>
                <p>
                  In chat, try <code>add 101 to cart</code>,{" "}
                  <code>show my cart</code>, <code>checkout</code>,{" "}
                  <code>my orders</code>, etc.
                </p>
              </div>
            </div>
          </div>

          <div className="hero-side-card">
            <h3>Backend Status</h3>
            <p className={`health-status health-${health.status || "unknown"}`}>
              Status: <b>{health.status}</b>
            </p>
            <p>Service: {health.service || "-"}</p>
            <p>
              Last check:{" "}
              {health.time ? new Date(health.time).toLocaleString() : "-"}
            </p>
            <p className="health-note">
              API Base: <code>{BACKEND_URL}</code>
            </p>
          </div>
        </section>

        {/* PRODUCTS – dynamic */}
        <section id="products" className="section">
          <div className="section-header">
            <h2>Featured Products (Dynamic Catalog)</h2>
            <p>
              These products are loaded from <code>/products</code>. The chatbot
              uses the same API when you type <code>show products</code>.
            </p>
          </div>

          {productsLoading && <p>Loading products from backend...</p>}

          {productsError && !productsLoading && (
            <p className="error-text">{productsError}</p>
          )}

          {!productsLoading && !productsError && (
            <div className="product-grid">
              {products.slice(0, 8).map((p) => (
                <div key={p.id} className="product-card">
                  <div className="product-tag">From API</div>
                  <div className="product-body">
                    <h3>{p.title}</h3>
                    <p className="price">₹{p.price}</p>

                    <p className="order-label">Sample Product / Order ID</p>
                    <p className="order-id">{p.id}</p>

                    <p className="product-hint">
                      In the bot, try: <code>details {p.id}</code>,{" "}
                      <code>add {p.id} to cart</code>, or <code>track {p.id}</code>.
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ORDER TRACKER */}
        <section id="order-tracker" className="section section-dark">
          <div className="section-header">
            <h2>Live Order Tracker (Same Backend as the Bot)</h2>
            <p>
              This form calls the <code>/order?oid=&lt;id&gt;</code> endpoint.
              The chatbot uses the same API when you type <code>track 101</code>.
            </p>
          </div>

          <div className="order-tracker">
            <form onSubmit={handleTrackOrder} className="order-form">
              <label>
                Enter Sample Order ID
                <input
                  type="text"
                  value={orderId}
                  onChange={(e) => setOrderId(e.target.value)}
                  placeholder="e.g. 101"
                />
              </label>
              <button
                type="submit"
                className="primary-btn"
                disabled={orderLoading}
              >
                {orderLoading ? "Checking..." : "Track Order"}
              </button>
            </form>

            {orderError && <div className="error-box">⚠ {orderError}</div>}

            {orderData && !orderError && (
              <div className="order-result">
                <h3>📦 Order Tracking</h3>
                <p>
                  <b>Order ID:</b> {orderData.order_id}
                  <br />
                  <b>Status:</b> {orderData.stage}
                  <br />
                  <b>ETA:</b> {orderData.eta_days} day(s)
                  <br />
                  <b>Source:</b> {orderData.source}
                </p>

                {orderData.history && (
                  <>
                    <h4>🕒 History</h4>
                    <ul>
                      {orderData.history.map((h, idx) => (
                        <li key={idx}>{h}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            )}
          </div>
        </section>

        {/* BOT & AGENT ASSIST DEMO */}
        <section id="agent-assist" className="section">
          <div className="section-header">
            <h2>Live Bot & Agent Assist Demo</h2>
            <p>
              This panel sends your message to the same AI engine used by Zoho
              SalesIQ and visualises the Agent Assist PRO metadata.
            </p>
          </div>

          <div className="agent-grid">
            {/* Left: Chat */}
            <div className="agent-panel">
              <h3>Test a Message</h3>

              <form className="chat-form" onSubmit={handleSendChat}>
                <label className="chat-label">
                  Message to Smart AI Assistant
                  <textarea
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder='Try: "add 101 to cart", "show my cart", "checkout", "my orders", "cancel order 123456", "reorder 123456", "details 101", "search earbuds"'
                    rows={3}
                  />
                </label>

                <div className="chat-actions">
                  <div className="quick-buttons">
                    <button
                      type="button"
                      onClick={() => handleQuickMessage("show products")}
                    >
                      show products
                    </button>
                    <button
                      type="button"
                      onClick={() => handleQuickMessage("add 101 to cart")}
                    >
                      add 101 to cart
                    </button>
                    <button
                      type="button"
                      onClick={() => handleQuickMessage("show my cart")}
                    >
                      show my cart
                    </button>
                    <button
                      type="button"
                      onClick={() => handleQuickMessage("checkout")}
                    >
                      checkout
                    </button>
                    <button
                      type="button"
                      onClick={() => handleQuickMessage("my orders")}
                    >
                      my orders
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        handleQuickMessage("cancel order 123456")
                      }
                    >
                      cancel order
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        handleQuickMessage("reorder 123456")
                      }
                    >
                      reorder
                    </button>
                    <button
                      type="button"
                      onClick={() => handleQuickMessage("details 101")}
                    >
                      details 101
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        handleQuickMessage("search earbuds")
                      }
                    >
                      search earbuds
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        handleQuickMessage("this is bad, I am angry")
                      }
                    >
                      angry (sentiment)
                    </button>
                  </div>

                  <button
                    type="submit"
                    className="primary-btn"
                    disabled={isSending}
                  >
                    {isSending ? "Sending..." : "Send to Backend"}
                  </button>
                </div>
              </form>

              {chatError && <div className="error-box">⚠ {chatError}</div>}

              {botResponse && (
                <div className="bot-response">
                  <h4>Bot Response</h4>
                  <p>{botResponse}</p>

                  <div className="inline-meta">
                    {intent && (
                      <span>
                        Intent: <b>{intent}</b>
                      </span>
                    )}
                    {emotion && (
                      <span>
                        Emotion: <b>{emotion}</b>
                      </span>
                    )}
                    {confidence !== null && (
                      <span>
                        Confidence:{" "}
                        <b>{(confidence * 100).toFixed(0)}%</b>
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Right: Agent Assist */}
            <div className="agent-panel">
              <h3>Operator View – Agent Assist PRO</h3>

              {!agentAssist && !metadata && (
                <p className="muted">
                  Send a test message to see frustration score, risk level,
                  summary and suggested reply that an operator would see inside
                  SalesIQ.
                </p>
              )}

              {agentAssist && (
                <div className="agent-assist-card">
                  <div className="aa-row">
                    <span className="aa-label">Frustration Score</span>
                    <span className="aa-value">
                      {agentAssist.frustration ?? 0}
                    </span>
                  </div>
                  <div className="aa-row">
                    <span className="aa-label">Risk Level</span>
                    <span className="aa-value">
                      {agentAssist.risk || "low"}
                    </span>
                  </div>
                  <div className="aa-row">
                    <span className="aa-label">Suggestion</span>
                    <span className="aa-value">
                      {agentAssist.suggestion || "-"}
                    </span>
                  </div>
                  <div className="aa-row">
                    <span className="aa-label">Summary</span>
                    <span className="aa-value aa-summary">
                      {agentAssist.summary || "-"}
                    </span>
                  </div>

                  {agentAssist.recent_messages?.length > 0 && (
                    <div className="aa-recent">
                      <span className="aa-label">Recent Messages</span>
                      <ul>
                        {agentAssist.recent_messages.map((m, i) => (
                          <li key={i}>{m}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {metadata && (
                <div className="meta-raw">
                  <h4>Raw Metadata (for evaluators)</h4>
                  <pre>{JSON.stringify(metadata, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* FEEDBACK MAPPING */}
        <section id="evaluators" className="section">
          <div className="section-header">
            <h2>CliqTrix&apos;26 – Feedback Mapping</h2>
            <p>
              This implementation directly addresses the feedback shared by the
              CliqTrix team.
            </p>
          </div>

          <div className="feedback-grid">
            <div className="feedback-card">
              <h3>1️⃣ Bot + E-Commerce Integration</h3>
              <p>
                Both the website and the Zoho SalesIQ bot are connected to a
                shared e-commerce backend via REST APIs.
              </p>
              <ul>
                <li>
                  <code>/products</code> and <code>/order</code> power both this
                  page and the chatbot.
                </li>
                <li>
                  Bot flows like <b>track 101</b> use the same backend endpoints.
                </li>
                <li>
                  Orders created via the AI engine can be tracked with order IDs.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>2️⃣ Product Browsing & Order Flows Inside the Bot</h3>
              <p>
                The AI engine supports a mini e-commerce flow:
                product search, details, cart, checkout and orders, all via chat.
              </p>
              <ul>
                <li>
                  Browsing: <code>show products</code>, <code>search earbuds</code>,
                  <code>details 101</code>.
                </li>
                <li>
                  Cart & checkout: <code>add 101 to cart</code>,{" "}
                  <code>show my cart</code>, <code>checkout</code>.
                </li>
                <li>
                  Orders: <code>my orders</code>,{" "}
                  <code>cancel order 123456</code>, <code>reorder 123456</code>.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>3️⃣ Improved Sentiment Analysis & Agent Assist</h3>
              <p>
                The AI engine uses lexicon + negation-based sentiment and
                multi-message frustration scoring to drive Agent Assist.
              </p>
              <ul>
                <li>
                  Classifies user emotion as <b>happy</b>, <b>neutral</b> or{" "}
                  <b>angry</b>.
                </li>
                <li>
                  Tracks frustration across the recent message history.
                </li>
                <li>
                  Exposes summary, frustration and risk via{" "}
                  <code>agent_assist</code> object used by operators in SalesIQ.
                </li>
              </ul>
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <span>
          © 2025 Smart AI Assistant • Team OHM • Built for CliqTrix&apos;26
        </span>
      </footer>
    </div>
  );
}

export default App;
