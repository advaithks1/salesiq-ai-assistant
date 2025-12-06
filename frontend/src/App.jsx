import React, { useEffect, useState } from "react";
import "./index.css";

// Backend shared by website + Zoho SalesIQ bot
const BACKEND_URL =
  import.meta.env?.VITE_API_BASE_URL ||
  "https://salesiq-ai-assistant-8m8r.onrender.com";

function App() {
  // HEALTH
  const [health, setHealth] = useState({ status: "checking..." });

  // PRODUCTS
  const [products, setProducts] = useState([]);
  const [productsLoading, setProductsLoading] = useState(true);
  const [productsError, setProductsError] = useState("");

  // ORDER TRACKER
  const [orderId, setOrderId] = useState("");
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState("");
  const [orderData, setOrderData] = useState(null);

  // On load: health + products
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

  // Order tracker handler
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
          <a href="#howto">How to Test</a>
          <a href="#evaluators">For Evaluators</a>
        </div>
      </header>

      <main className="main">
        {/* HERO */}
        <section id="top" className="hero">
          <div className="hero-text">
            <div className="pill">CliqTrix&apos;26 – Smart AI Assistant</div>

            <h1>AI-Powered E-Commerce Support & Order Tracking</h1>

            <p>
              This single-page demo connects a custom AI backend, an
              e-commerce-style catalog and a Zoho SalesIQ chatbot. The same
              product IDs and order IDs are used by both this page and the
              chatbot.
            </p>

            <p className="hero-note">
              <span className="dot" /> All conversations happen in the{" "}
              <b>Zoho SalesIQ chat bubble</b> at the bottom-right. The bot uses
              the same backend as this page for order tracking, product browsing
              and sentiment analysis.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>1. Browse Products</h3>
                <p>Products below are loaded live from the backend.</p>
              </div>
              <div className="hero-card">
                <h3>2. Track Orders</h3>
                <p>
                  Use the <b>Order Tracker</b> on this page or type{" "}
                  <code>track 101</code> in the chatbot.
                </p>
              </div>
              <div className="hero-card">
                <h3>3. Chat with the Bot</h3>
                <p>
                  Open the SalesIQ bubble and try messages like{" "}
                  <code>show products</code>, <code>add 101 to cart</code>,{" "}
                  <code>show my cart</code>, <code>checkout</code>,{" "}
                  <code>my orders</code>.
                </p>
              </div>
            </div>
          </div>

          {/* Backend status */}
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
              uses the same API when you type <code>show products</code> in the
              Zoho SalesIQ widget.
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
                      In the chatbot, you can try:{" "}
                      <code>details {p.id}</code>,{" "}
                      <code>add {p.id} to cart</code>,{" "}
                      <code>track {p.id}</code>.
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
            <h2>Live Order Tracker (Shared with Bot)</h2>
            <p>
              This form calls the <code>/order?oid=&lt;id&gt;</code> endpoint in
              the backend. The Zoho SalesIQ bot calls the same API when a user
              types <code>track 101</code>.
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

        {/* HOW TO TEST (FOCUS ON ZOHO BOT) */}
        <section id="howto" className="section">
          <div className="section-header">
            <h2>How to Test the Smart AI Assistant (Zoho SalesIQ)</h2>
            <p>
              All bot interactions happen in the Zoho SalesIQ widget at the
              bottom-right of this page.
            </p>
          </div>

          <div className="usage-grid">
            <div className="usage-card">
              <h3>🧾 Order Tracking</h3>
              <ul>
                <li>Open the Zoho SalesIQ chat bubble.</li>
                <li>
                  Type <code>track 101</code>, <code>track 102</code>, etc.
                </li>
                <li>
                  The bot calls the same <code>/order</code> API used by this
                  page and shows stage, ETA and history.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>🛍️ E-Commerce Flow in Chat</h3>
              <ul>
                <li>
                  Browse products: <code>show products</code>,{" "}
                  <code>search earbuds</code>, <code>details 101</code>.
                </li>
                <li>
                  Manage cart: <code>add 101 to cart</code>,{" "}
                  <code>show my cart</code>.
                </li>
                <li>
                  Place & manage orders: <code>checkout</code>,{" "}
                  <code>my orders</code>, <code>cancel order 500123</code>,{" "}
                  <code>reorder 500123</code>.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>😊 Sentiment & Agent Assist</h3>
              <ul>
                <li>
                  Send: <code>thanks, this is great</code> (happy).
                </li>
                <li>
                  Then: <code>this is bad, I am angry</code> (frustrated).
                </li>
                <li>
                  The backend updates emotion, frustration score and risk level,
                  shown to operators through Agent Assist in the SalesIQ
                  dashboard.
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* FEEDBACK MAPPING */}
        <section id="evaluators" className="section">
          <div className="section-header">
            <h2>CliqTrix&apos;26 – Feedback Mapping</h2>
            <p>
              This implementation directly addresses the three feedback points
              shared by the CliqTrix team.
            </p>
          </div>

          <div className="feedback-grid">
            <div className="feedback-card">
              <h3>1️⃣ Bot + E-Commerce Integration</h3>
              <p>
                The Zoho SalesIQ chatbot and this website both connect to the
                same custom backend.
              </p>
              <ul>
                <li>
                  <code>/order</code> and <code>/products</code> are used by
                  both the chat bot (via Deluge) and this page.
                </li>
                <li>
                  Order IDs and product IDs are consistent across website and
                  chat.
                </li>
                <li>
                  Demonstrates real API-based integration, not static replies.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>2️⃣ Product Browsing Inside the Bot</h3>
              <p>
                Users can explore the same catalog shown above directly via
                chat.
              </p>
              <ul>
                <li>
                  Product listing via <code>show products</code>.
                </li>
                <li>
                  Product details via <code>details 101</code>.
                </li>
                <li>
                  Full e-commerce flow in chat: cart, checkout, orders.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>3️⃣ Improved Sentiment Analysis</h3>
              <p>
                The AI engine powering the Zoho SalesIQ bot includes refined
                sentiment rules and frustration scoring for Agent Assist.
              </p>
              <ul>
                <li>
                  Classifies messages as <b>happy</b>, <b>neutral</b> or{" "}
                  <b>angry</b> using lexicon + negation.
                </li>
                <li>
                  Tracks frustration across multiple messages per visitor.
                </li>
                <li>
                  Exposes summary, frustration and risk to agents via metadata
                  and the <code>agent_assist</code> object.
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
