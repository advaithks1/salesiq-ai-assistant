import React, { useEffect, useState } from "react";
import "./index.css";

// Same backend the bot uses
const BACKEND_URL = "https://salesiq-ai-assistant-8m8r.onrender.com";

function App() {
  // ----------------------------
  // PRODUCTS (dynamic from /products)
  // ----------------------------
  const [products, setProducts] = useState([]);
  const [productsLoading, setProductsLoading] = useState(true);
  const [productsError, setProductsError] = useState("");

  // ----------------------------
  // LIVE ORDER TRACKER (from /order)
  // ----------------------------
  const [orderId, setOrderId] = useState("");
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState("");
  const [orderData, setOrderData] = useState(null);

  // Fetch products from backend on load
  useEffect(() => {
    const fetchProducts = async () => {
      try {
        setProductsLoading(true);
        setProductsError("");

        const res = await fetch(`${BACKEND_URL}/products`);
        const data = await res.json();

        // API shape: { products: [...], source: "dummyjson" }
        if (Array.isArray(data.products)) {
          setProducts(data.products);
        } else {
          setProductsError("Product list not available.");
        }
      } catch (err) {
        setProductsError("Failed to load products.");
      } finally {
        setProductsLoading(false);
      }
    };

    fetchProducts();
  }, []);

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
    } catch (err) {
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
              <span className="dot" /> The chatbot at the bottom-right can{" "}
              <b>track orders</b>, <b>show products</b> and{" "}
              <b>analyze sentiment</b> — using the same backend that powers this
              web page.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>1. Browse Products</h3>
                <p>
                  Scroll down to view products loaded dynamically from the
                  backend.
                </p>
              </div>
              <div className="hero-card">
                <h3>2. Track Orders</h3>
                <p>
                  Use the <b>Order Tracker</b> section or the chatbot to track
                  any sample order ID (e.g. <code>101</code>).
                </p>
              </div>
              <div className="hero-card">
                <h3>3. Test the Bot</h3>
                <p>
                  Open the Zoho SalesIQ chat bubble and try{" "}
                  <code>track 101</code> or <code>show products</code> — it hits
                  the same backend as this page.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* PRODUCTS – dynamic from backend */}
        <section id="products" className="section">
          <div className="section-header">
            <h2>Featured Products (Dynamic Catalog)</h2>
            <p>
              These products are loaded from the backend via{" "}
              <code>/products</code>. The chatbot uses the same API when you
              type <code>show products</code> in chat.
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

                    <p className="order-label">Sample Order ID</p>
                    <p className="order-id">{p.id}</p>

                    <p className="product-hint">
                      In the bot, type: <code>track {p.id}</code>
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* LIVE ORDER TRACKER – website -> /order */}
        <section id="order-tracker" className="section section-dark">
          <div className="section-header">
            <h2>Live Order Tracker (Same Backend as the Bot)</h2>
            <p>
              This form calls the <code>/order?oid=&lt;id&gt;</code> endpoint in
              the backend. The chatbot uses the exact same API when you type{" "}
              <code>track 101</code>.
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
              <button type="submit" className="primary-btn" disabled={orderLoading}>
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

        {/* HOW TO TEST BOT */}
        <section id="howto" className="section">
          <div className="section-header">
            <h2>How to Test the Smart AI Assistant</h2>
            <p>
              Use the Zoho SalesIQ widget (bottom-right) and the sections above
              to see how the website and bot share the same backend.
            </p>
          </div>

          <div className="usage-grid">
            <div className="usage-card">
              <h3>🧾 Order Tracking</h3>
              <ul>
                <li>
                  On the website, use the <b>Live Order Tracker</b> with sample
                  IDs like <code>101</code>, <code>102</code>, etc.
                </li>
                <li>
                  In chat, type <code>track 101</code>. The bot calls the same{" "}
                  <code>/order</code> API.
                </li>
                <li>
                  Both show stage, ETA and order history from the backend.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>🛍️ Product Browsing</h3>
              <ul>
                <li>
                  This page loads products from <code>/products</code> API.
                </li>
                <li>
                  In chat, type <code>show products</code> or{" "}
                  <code>browse products</code> to browse the same data.
                </li>
                <li>
                  Demonstrates shared e-commerce layer between web & chatbot.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>😊 Sentiment & Agent Assist</h3>
              <ul>
                <li>
                  In chat, send happy text:{" "}
                  <code>thanks, this is great</code>.
                </li>
                <li>
                  Then send frustrated text:{" "}
                  <code>this is bad, I am angry</code>.
                </li>
                <li>
                  Backend sentiment engine updates frustration / risk, which the
                  operator sees via Agent Assist in SalesIQ.
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* ARCHITECTURE / FEEDBACK MAPPING */}
        <section id="evaluators" className="section">
          <div className="section-header">
            <h2>CliqTrix&apos;26 – Feedback Mapping</h2>
            <p>
              This implementation directly addresses the three key feedback
              points shared by the CliqTrix team.
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
                  <code>/order</code> and <code>/products</code> are consumed by
                  this page.
                </li>
                <li>
                  The chatbot calls the same APIs via Deluge integration.
                </li>
                <li>
                  Demonstrates a real, API-based integration instead of static
                  replies.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>2️⃣ Product Browsing Inside the Bot</h3>
              <p>
                Products shown above are loaded dynamically from the backend and
                can also be explored via chat commands like{" "}
                <code>show products</code>.
              </p>
              <ul>
                <li>
                  Single source of truth: <code>/products</code> API.
                </li>
                <li>
                  Shared IDs between website cards and chatbot prompts.
                </li>
                <li>
                  Clear, consistent e-commerce experience across channels.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>3️⃣ Improved Sentiment Analysis</h3>
              <p>
                The AI engine behind the bot includes a refined sentiment &
                frustration scoring system that powers Agent Assist for
                operators.
              </p>
              <ul>
                <li>
                  Classifies user emotion as <b>happy</b>, <b>neutral</b> or{" "}
                  <b>angry</b> using lexicon + negation.
                </li>
                <li>
                  Aggregates frustration across multiple user messages.
                </li>
                <li>
                  Exposes summary, frustration and risk via metadata to the
                  SalesIQ operator panel.
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
