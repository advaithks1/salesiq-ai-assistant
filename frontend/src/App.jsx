import React, { useEffect, useState } from "react";
import "./index.css";

// Backend shared by website + Zoho SalesIQ bot
const BACKEND_URL =
  import.meta.env?.VITE_API_BASE_URL ||
  "https://salesiq-ai-assistant-8m8r.onrender.com";

// Shared demo user id (matches Deluge visitor_id)
const DEMO_USER_ID = "demo-user";

// Static products used by FRONTEND + BOT (same IDs as AI engine)
const STATIC_PRODUCTS = [
  {
    id: 101,
    title: "Wireless Earbuds",
    price: 1299,
    tag: "Best Seller",
  },
  {
    id: 102,
    title: "Smartwatch",
    price: 2499,
    tag: "Trending",
  },
  {
    id: 103,
    title: "Bluetooth Speaker",
    price: 1999,
    tag: "New",
  },
  {
    id: 104,
    title: "Laptop Stand",
    price: 799,
    tag: "Popular",
  },
  // If you want a 5th product, add one more object here and in backend/AI DB.
];

function App() {
  // HEALTH
  const [health, setHealth] = useState({ status: "checking..." });

  // CART (synced with backend / AI engine)
  const [cart, setCart] = useState([]);
  const [cartLoading, setCartLoading] = useState(false);
  const [cartError, setCartError] = useState("");

  // ORDER TRACKER
  const [orderId, setOrderId] = useState("");
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState("");
  const [orderData, setOrderData] = useState(null);

  // On load: health + cart (products are static, no fetch)
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

    const fetchCart = async () => {
      try {
        setCartLoading(true);
        setCartError("");
        const res = await fetch(
          `${BACKEND_URL}/cart?user_id=${DEMO_USER_ID}`
        );
        const data = await res.json();
        setCart(Array.isArray(data.items) ? data.items : []);
      } catch {
        setCartError("Failed to load cart.");
      } finally {
        setCartLoading(false);
      }
    };

    fetchHealth();
    fetchCart();
  }, []);

  // --------- CART HELPERS (BACKEND-SYNCED) ----------

  const syncCartFromBackend = async () => {
    try {
      setCartLoading(true);
      setCartError("");
      const res = await fetch(
        `${BACKEND_URL}/cart?user_id=${DEMO_USER_ID}`
      );
      const data = await res.json();
      setCart(Array.isArray(data.items) ? data.items : []);
    } catch {
      setCartError("Failed to sync cart.");
    } finally {
      setCartLoading(false);
    }
  };

  const handleAddToCart = async (product) => {
    try {
      setCartLoading(true);
      setCartError("");
      const res = await fetch(`${BACKEND_URL}/cart/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: DEMO_USER_ID,
          product_id: product.id,
        }),
      });
      const data = await res.json();
      setCart(Array.isArray(data.items) ? data.items : []);
    } catch {
      setCartError("Failed to add item to cart.");
    } finally {
      setCartLoading(false);
    }
  };

  const handleRemoveFromCart = async (id) => {
    try {
      setCartLoading(true);
      setCartError("");
      const res = await fetch(`${BACKEND_URL}/cart/remove`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: DEMO_USER_ID,
          product_id: id,
        }),
      });
      const data = await res.json();
      setCart(Array.isArray(data.items) ? data.items : []);
    } catch {
      setCartError("Failed to remove item from cart.");
    } finally {
      setCartLoading(false);
    }
  };

  const cartCount = cart.reduce((sum, item) => sum + (item.qty || 0), 0);
  const cartTotal = cart.reduce(
    (sum, item) => sum + (Number(item.price) || 0) * (item.qty || 0),
    0
  );

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
          <a href="#cart">Cart ({cartCount})</a>
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
              This single-page demo connects a custom AI backend, a
              sample e-commerce catalog and a Zoho SalesIQ chatbot. The same
              product IDs and order IDs are used by both this page and the
              chatbot.
            </p>

            <p className="hero-note">
              <span className="dot" /> All conversations happen in the{" "}
              <b>Zoho SalesIQ chat bubble</b> at the bottom-right. The bot uses
              the same backend as this page for order tracking, shared cart and
              sentiment analysis.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>1. Browse Products</h3>
                <p>Products below are static but share IDs with the chatbot.</p>
              </div>
              <div className="hero-card">
                <h3>2. Track Orders</h3>
                <p>
                  Use the <b>Order Tracker</b> on this page or type{" "}
                  <code>track 101</code> in the chatbot.
                </p>
              </div>
              <div className="hero-card">
                <h3>3. Shared Cart</h3>
                <p>
                  Add items here or via chat (<code>add 101 to cart</code>) and
                  sync the cart between both.
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

        {/* PRODUCTS – STATIC (but aligned with bot) */}
        <section id="products" className="section">
          <div className="section-header">
            <h2>Sample Products (Shared with Chatbot)</h2>
            <p>
              These products are defined once and used both by this page and the
              chatbot. IDs match the cart and order flows like{" "}
              <code>add 101 to cart</code>, <code>details 101</code>,{" "}
              <code>track 101</code>.
            </p>
          </div>

          <div className="product-grid">
            {STATIC_PRODUCTS.map((p) => (
              <div key={p.id} className="product-card">
                <div className="product-tag">
                  {p.tag ? p.tag : "Sample"}
                </div>
                <div className="product-body">
                  <h3>{p.title}</h3>
                  <p className="price">₹{p.price}</p>

                  <p className="order-label">Sample Product / Order ID</p>
                  <p className="order-id">{p.id}</p>

                  <p className="product-hint">
                    In the chatbot, you can try:{" "}
                    <code>details {p.id}</code>,{" "}
                    <code>add {p.id} to cart</code>,{" "}
                    <code>remove {p.id} from cart</code>,{" "}
                    <code>track {p.id}</code>.
                  </p>

                  <button
                    type="button"
                    className="primary-btn"
                    onClick={() => handleAddToCart(p)}
                  >
                    Add to Cart (Shared with Bot)
                  </button>
                </div>
              </div>
            ))}
          </div>
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

        {/* CART SECTION (SYNCED WITH BOT) */}
        <section id="cart" className="section">
          <div className="section-header">
            <h2>Shared Cart (Chatbot + Frontend)</h2>
            <p>
              This cart is backed by the same memory used by the Zoho SalesIQ
              bot. Add items in chat using{" "}
              <code>add 101 to cart</code> / <code>remove 101 from cart</code>{" "}
              and click <b>Sync from Chatbot</b> to see them here.
            </p>
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <button
              type="button"
              className="secondary-btn"
              onClick={syncCartFromBackend}
              disabled={cartLoading}
            >
              {cartLoading ? "Syncing..." : "Sync from Chatbot"}
            </button>
          </div>

          {cartError && <p className="error-text">{cartError}</p>}

          {cart.length === 0 && !cartLoading ? (
            <p>Your shared cart is empty. Use the buttons above or the chatbot to add items.</p>
          ) : (
            <div className="cart-box">
              {cartLoading && <p>Updating cart...</p>}
              {!cartLoading && cart.length > 0 && (
                <table className="cart-table">
                  <thead>
                    <tr>
                      <th>Product</th>
                      <th>Qty</th>
                      <th>Price</th>
                      <th>Subtotal</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {cart.map((item) => (
                      <tr key={item.id}>
                        <td>{item.title}</td>
                        <td>{item.qty}</td>
                        <td>₹{item.price}</td>
                        <td>₹{(Number(item.price) || 0) * item.qty}</td>
                        <td>
                          <button
                            type="button"
                            className="secondary-btn"
                            onClick={() => handleRemoveFromCart(item.id)}
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr>
                      <td colSpan={3} style={{ textAlign: "right" }}>
                        <b>Total:</b>
                      </td>
                      <td colSpan={2}>
                        <b>₹{cartTotal}</b>
                      </td>
                    </tr>
                  </tfoot>
                </table>
              )}
            </div>
          )}
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
                  <code>details 101</code>.
                </li>
                <li>
                  Manage cart: <code>add 101 to cart</code>,{" "}
                  <code>remove 101 from cart</code>,{" "}
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
                  Shared endpoints for <code>/order</code> and <code>/cart</code>.
                </li>
                <li>
                  Order IDs and product IDs are consistent across website and
                  chat.
                </li>
                <li>
                  Demonstrates real API-based integration for tracking and cart,
                  not static replies.
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
