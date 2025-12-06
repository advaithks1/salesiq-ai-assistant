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
    tag: "New Arrival",
  },
  {
    id: 104,
    title: "Laptop Stand",
    price: 799,
    tag: "Popular",
  },
];

function App() {
  // HEALTH
  const [health, setHealth] = useState({ status: "checking..." });

  // CART (synced with backend / AI engine)
  const [cart, setCart] = useState([]);
  const [cartLoading, setCartLoading] = useState(false);
  const [cartError, setCartError] = useState("");

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

  return (
    <div className="app-root">
      {/* NAVBAR */}
      <header className="nav">
        <div className="nav-left">
          <span className="brand">Smart AI Store</span>
          <span className="brand-sub">
            Powered by Smart AI Assistant • Zoho SalesIQ + FastAPI
          </span>
        </div>
        <div className="nav-right">
          <a href="#top">Overview</a>
          <a href="#products">Products</a>
          <a href="#cart">Shared Cart</a>
          <a href="#howto">How It Works</a>
          <a href="#evaluators">For Evaluators</a>
        </div>
      </header>

      <main className="main">
        {/* HERO */}
        <section id="top" className="hero">
          <div className="hero-text">
            <div className="pill">CliqTrix&apos;26 • Team OHM</div>

            <h1>Smart AI Assistant for E-Commerce & Support</h1>

            <p>
              This demo shows how a Zoho SalesIQ chatbot, a custom FastAPI
              backend and a simple storefront work together as a single
              experience. The same product IDs, order IDs and cart are shared
              between the chat widget and this page.
            </p>

            <p className="hero-note">
              <span className="dot" /> Open the <b>Zoho SalesIQ chat bubble</b>{" "}
              (bottom-right) and type <code>hi</code> to see what the bot can
              do. You can then come back here and sync the cart to see the
              same items on the website.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>Unified Experience</h3>
                <p>
                  The bot and this page call the same backend APIs for products,
                  orders and cart operations.
                </p>
              </div>
              <div className="hero-card">
                <h3>Shared Cart</h3>
                <p>
                  Add items in chat using <code>add 101 to cart</code> or from
                  this page, then sync to keep them in sync.
                </p>
              </div>
              <div className="hero-card">
                <h3>Operator Insights</h3>
                <p>
                  Every message is analysed for intent, emotion, frustration and
                  risk, and exposed as Agent Assist inside SalesIQ.
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
            <h2>Sample Product Catalog</h2>
            <p>
              These products are defined once in the backend and reused by both
              the chatbot and the website. The IDs shown here are the same IDs
              you use in the chat (for example, <code>details 101</code> or{" "}
              <code>add 101 to cart</code>).
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

                  <p className="order-label">Product / Order ID</p>
                  <p className="order-id">{p.id}</p>

                  <p className="product-hint">
                    In the chatbot, try:{" "}
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
                    Add to Cart (shared with bot)
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* CART SECTION (SYNCED WITH BOT) */}
        <section id="cart" className="section">
          <div className="section-header">
            <h2>Shared Cart (Chatbot + Website)</h2>
            <p>
              This cart is backed by the same in-memory store that the Zoho
              SalesIQ bot uses. If you add or remove items in chat, you can
              refresh the view here with a single click.
            </p>
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <button
              type="button"
              className="secondary-btn"
              onClick={syncCartFromBackend}
              disabled={cartLoading}
            >
              {cartLoading ? "Syncing with chatbot..." : "Sync cart from chatbot"}
            </button>
            <span style={{ marginLeft: "0.75rem", opacity: 0.8 }}>
              Items in cart: <b>{cartCount}</b>
            </span>
          </div>

          {cartError && <p className="error-text">{cartError}</p>}

          {cart.length === 0 && !cartLoading ? (
            <p>
              The shared cart is currently empty. Add products from this page or
              by using commands like <code>add 101 to cart</code> in the chat,
              then click <b>Sync cart from chatbot</b>.
            </p>
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

        {/* HOW TO TEST (FOCUS ON REAL FLOWS) */}
        <section id="howto" className="section">
          <div className="section-header">
            <h2>How to Experience the Assistant</h2>
            <p>
              The idea is to show an end-to-end journey: browse products, talk
              to the bot, change the cart and see how everything stays in sync.
            </p>
          </div>

          <div className="usage-grid">
            <div className="usage-card">
              <h3>1. Start a Conversation</h3>
              <ul>
                <li>Open the Zoho SalesIQ chat bubble on this page.</li>
                <li>Type <code>hi</code> to see the menu of capabilities.</li>
                <li>
                  Try examples like <code>show products</code>,{" "}
                  <code>details 101</code>, <code>refund status</code>.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>2. Work with Orders & Cart</h3>
              <ul>
                <li>
                  In chat: <code>add 101 to cart</code>,{" "}
                  <code>show my cart</code>, <code>checkout</code>.
                </li>
                <li>
                  On the website: add items from the product cards, then open
                  the <b>Shared Cart</b> section.
                </li>
                <li>
                  Click <b>Sync cart from chatbot</b> and see the same cart
                  items reflected here.
                </li>
                <li>
                  In chat, you can also use <code>track 101</code> to see
                  simulated order tracking.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>3. See Sentiment & Agent Assist</h3>
              <ul>
                <li>Send a positive message, e.g. “thanks, this is great”.</li>
                <li>
                  Then send a negative one, e.g. “this is bad, I am angry”.
                </li>
                <li>
                  In the SalesIQ operator view, the Agent Assist panel shows the
                  detected intent, emotion, frustration score, risk level,
                  recent summary and suggested reply.
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* FEEDBACK MAPPING */}
        <section id="evaluators" className="section">
          <div className="section-header">
            <h2>CliqTrix&apos;26 – Feedback Implementation Summary</h2>
            <p>
              The changes in this version are based on the review comments
              shared after the first round. Below is how each point was
              addressed in a concrete way.
            </p>
          </div>

          <div className="feedback-grid">
            <div className="feedback-card">
              <h3>1️⃣ Bot integrated with e-commerce backend</h3>
              <p>
                The chatbot no longer works with static replies. It uses a
                dedicated FastAPI backend for core operations:
              </p>
              <ul>
                <li>
                  <code>/order</code> for deterministic order tracking
                  simulation (also used by chat with <code>track 101</code>).
                </li>
                <li>
                  <code>/products</code> and a shared{" "}
                  <code>PRODUCT_DB</code> for consistent IDs and names.
                </li>
                <li>
                  <code>/cart</code>, <code>/cart/add</code>,{" "}
                  <code>/cart/remove</code> for a shared cart between chat and
                  frontend.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>2️⃣ Product discovery inside the chatbot</h3>
              <p>
                Users can work with the same catalog entirely from the chat
                interface:
              </p>
              <ul>
                <li>
                  <code>show products</code> lists the catalog.
                </li>
                <li>
                  <code>details 101</code> returns a structured product card.
                </li>
                <li>
                  <code>add 101 to cart</code>, <code>remove 101 from cart</code>,{" "}
                  <code>show my cart</code> and <code>checkout</code> complete
                  the in-chat e-commerce flow.
                </li>
                <li>
                  The same IDs are visible on this page so the relationship is
                  clear during the demo.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>3️⃣ Refined sentiment analysis & Agent Assist</h3>
              <p>
                The AI engine now includes a small but effective sentiment and
                risk layer to assist human operators:
              </p>
              <ul>
                <li>
                  Lexicon-based sentiment with basic negation handling, mapping
                  messages to <b>happy</b>, <b>neutral</b> or <b>angry</b>.
                </li>
                <li>
                  Frustration score accumulated over recent messages for each
                  visitor.
                </li>
                <li>
                  Risk level, conversation summary, last three messages and a
                  suggested reply exposed via metadata and rendered as an
                  Agent Assist panel in the SalesIQ operator view.
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
