import React from "react";
import "./index.css";

const PRODUCTS = [
  { id: "101", name: "Wireless Earbuds", price: "₹1,299", tag: "Best Seller" },
  { id: "102", name: "Smartwatch", price: "₹2,499", tag: "Trending" },
  { id: "103", name: "Bluetooth Speaker", price: "₹1,999", tag: "New" },
  { id: "104", name: "Laptop Stand", price: "₹799", tag: "Popular" },
];

function App() {
  return (
    <div className="app-root">
      {/* NAVBAR */}
      <header className="nav">
        <div className="nav-left">
          <span className="brand">Smart AI Store</span>
          <span className="brand-sub">Powered by Smart AI Assistant (Zoho SalesIQ)</span>
        </div>
        <div className="nav-right">
          <a href="#top">Dashboard</a>
          <a href="#products">Products</a>
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
              This single-page demo connects a custom AI backend, an e-commerce style
              catalog and a Zoho SalesIQ chatbot. The same product IDs and order IDs
              are used by both this page and the chatbot.
            </p>

            <p className="hero-note">
              <span className="dot" /> The chatbot at the bottom-right can{" "}
              <b>track orders</b>, <b>show products</b> and <b>analyze sentiment</b>.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>1. Browse Products</h3>
                <p>Scroll down to view the featured products and sample order IDs.</p>
              </div>
              <div className="hero-card">
                <h3>2. Open the Bot</h3>
                <p>
                  Click the Zoho SalesIQ chat bubble at the bottom-right to open the{" "}
                  <b>Smart AI Assistant</b>.
                </p>
              </div>
              <div className="hero-card">
                <h3>3. Track & Explore</h3>
                <p>
                  In chat, try messages like <code>track 101</code> or{" "}
                  <code>show products</code>.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* PRODUCTS */}
        <section id="products" className="section">
          <div className="section-header">
            <h2>Featured Products (Demo Catalog)</h2>
            <p>
              These products are part of the e-commerce flow and are also accessible
              from the chatbot using the same IDs.
            </p>
          </div>

          <div className="product-grid">
            {PRODUCTS.map((p) => (
              <div key={p.id} className="product-card">
                <div className="product-tag">{p.tag}</div>
                <div className="product-body">
                  <h3>{p.name}</h3>
                  <p className="price">{p.price}</p>

                  <p className="order-label">Sample Order ID</p>
                  <p className="order-id">{p.id}</p>

                  <p className="product-hint">
                    In the bot, type: <code>track {p.id}</code>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* HOW TO TEST BOT */}
        <section id="howto" className="section section-dark">
          <div className="section-header">
            <h2>How to Test the Smart AI Assistant</h2>
            <p>
              Use the Zoho SalesIQ widget (bottom-right) and try the following
              messages to see the integrated behaviour.
            </p>
          </div>

          <div className="usage-grid">
            <div className="usage-card">
              <h3>🧾 Order Tracking</h3>
              <ul>
                <li>
                  Try <code>track 101</code>, <code>track 102</code>, etc.
                </li>
                <li>
                  The bot calls the backend <code>/order?oid=&lt;id&gt;</code> endpoint.
                </li>
                <li>
                  It returns stage, ETA and order history as a rich tracking message.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>🛍️ Product Browsing</h3>
              <ul>
                <li>
                  Type <code>show products</code> or <code>browse products</code>.
                </li>
                <li>
                  The bot fetches product data from the e-commerce API layer.
                </li>
                <li>
                  Users can explore product names and prices directly in the chat.
                </li>
              </ul>
            </div>

            <div className="usage-card">
              <h3>😊 Sentiment & Agent Assist</h3>
              <ul>
                <li>
                  Send happy text: <code>thanks, this is great</code>.
                </li>
                <li>
                  Send frustrated text: <code>this is bad, I am angry</code>.
                </li>
                <li>
                  The backend classifies emotion and exposes frustration / risk +
                  suggestions for the operator via Agent Assist.
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
              The sections below explain how this implementation addresses each of
              the feedback points provided by the CliqTrix team.
            </p>
          </div>

          <div className="feedback-grid">
            <div className="feedback-card">
              <h3>1️⃣ Bot + E-Commerce Integration</h3>
              <p>
                The Smart AI Assistant is connected to an e-commerce style backend
                via REST APIs. The <code>/order</code> and <code>/products</code>{" "}
                endpoints are used both by this page and the chatbot.
              </p>
              <ul>
                <li>Order IDs on this page are the same as in the bot flow.</li>
                <li>SalesIQ Deluge script invokes the backend for real responses.</li>
                <li>Order tracking messages are rendered directly in chat.</li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>2️⃣ Product Browsing Inside the Bot</h3>
              <p>
                Users can explore the same catalog displayed above via chat commands
                such as <code>show products</code>.
              </p>
              <ul>
                <li>Backend exposes a <code>/products</code> API.</li>
                <li>
                  The bot lists product name + price using the API response.
                </li>
                <li>
                  This demonstrates an integrated e-commerce browsing experience.
                </li>
              </ul>
            </div>

            <div className="feedback-card">
              <h3>3️⃣ Improved Sentiment Analysis</h3>
              <p>
                The AI engine includes refined sentiment rules and a frustration
                scoring system that powers an Agent Assist view for operators.
              </p>
              <ul>
                <li>
                  Classifies messages as <b>happy</b>, <b>neutral</b> or{" "}
                  <b>angry</b>.
                </li>
                <li>
                  Tracks frustration across multiple messages (not just one message).
                </li>
                <li>
                  Exposes summary, frustration and risk level to agents via metadata.
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
