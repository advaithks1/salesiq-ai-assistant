import React from "react";
import "./App.css";

const PRODUCTS = [
  { id: "101", name: "Wireless Earbuds", price: "₹1,299", tag: "Best Seller" },
  { id: "102", name: "Smartwatch", price: "₹2,499", tag: "Trending" },
  { id: "103", name: "Bluetooth Speaker", price: "₹1,999", tag: "New" },
  { id: "104", name: "Laptop Stand", price: "₹799", tag: "Popular" },
];

function App() {
  return (
    <div className="app-root">
      {/* Top navigation */}
      <header className="nav">
        <div className="nav-left">
          <span className="brand">Smart AI Store</span>
          <span className="brand-sub">Powered by Smart AI Assistant</span>
        </div>
        <div className="nav-right">
          <span>Dashboard</span>
          <span>Products</span>
          <span>Support</span>
        </div>
      </header>

      {/* Hero / intro section */}
      <main className="main">
        <section className="hero">
          <div className="hero-text">
            <h1>Smart AI Assistant + E-Commerce Order Tracking</h1>
            <p>
              This demo connects a custom AI backend, an e-commerce style
              product catalog, and a Zoho SalesIQ chatbot into one experience.
            </p>
            <p className="hero-note">
              ✅ The same data (products & order IDs) are used by both this page
              and the SalesIQ bot.
            </p>

            <div className="hero-grid">
              <div className="hero-card">
                <h3>1. Browse Products</h3>
                <p>Check out the featured products and sample order IDs below.</p>
              </div>
              <div className="hero-card">
                <h3>2. Chat with the Bot</h3>
                <p>
                  Click the chat bubble at the bottom-right to open the{" "}
                  <b>Smart AI Assistant</b>.
                </p>
              </div>
              <div className="hero-card">
                <h3>3. Track & Explore</h3>
                <p>
                  Ask the bot to <code>track 101</code> or <code>show products</code>.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Product section */}
        <section className="section">
          <div className="section-header">
            <h2>Featured Products (Demo Catalog)</h2>
            <p>
              These products are part of the e-commerce flow and are also
              accessible via the chatbot.
            </p>
          </div>

          <div className="product-grid">
            {PRODUCTS.map((p) => (
              <div key={p.id} className="product-card">
                <div className="product-tag">{p.tag}</div>
                <div className="product-body">
                  <h3>{p.name}</h3>
                  <p className="price">{p.price}</p>
                  <p className="order-label">Sample Order ID:</p>
                  <p className="order-id">{p.id}</p>
                  <p className="product-hint">
                    In the bot, try: <code>track {p.id}</code>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Bot usage guide */}
        <section className="section section-dark">
          <div className="section-header">
            <h2>How to Test the Smart AI Assistant</h2>
            <p>
              Use the Zoho SalesIQ widget (bottom-right) and try the following
              messages.
            </p>
          </div>

          <div className="usage-grid">
            <div className="usage-card">
              <h3>🧾 Order Tracking</h3>
              <ul>
                <li>
                  <code>track 101</code>, <code>track 102</code> etc.
                </li>
                <li>
                  Bot calls the backend <b>/order?oid=&lt;id&gt;</b> endpoint.
                </li>
                <li>
                  Shows status, ETA and history inside the chat.
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
                  Bot fetches product data from the e-commerce API.
                </li>
                <li>
                  Returns a list of products with name & price.
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
                  Backend classifies emotion and powers Agent Assist metadata
                  (frustration, risk, suggestions).
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Technical overview */}
        <section className="section">
          <div className="section-header">
            <h2>Architecture Overview</h2>
            <p>This is the flow used to satisfy the CliqTrix feedback points.</p>
          </div>

          <div className="arch-grid">
            <div className="arch-card">
              <h3>🧠 AI Engine</h3>
              <p>
                FastAPI backend with intent detection, sentiment analysis, order
                simulation / lookup and memory. Exposed via <code>/chat</code>{" "}
                and <code>/order</code> endpoints.
              </p>
            </div>
            <div className="arch-card">
              <h3>💬 Zoho SalesIQ Bot</h3>
              <p>
                Uses a Deluge script to call the backend, render order tracking
                cards, product lists and Agent Assist insights for operators.
              </p>
            </div>
            <div className="arch-card">
              <h3>🛒 E-Commerce Layer</h3>
              <p>
                A simple e-commerce style catalog and order IDs shared between
                this page and the chatbot, demonstrating integration with an
                external platform / API.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <span>
          © 2025 Smart AI Assistant • Team OHM • Built for CliqTrix&apos;26
        </span>
      </footer>
    </div>
  );
}

export default App;
