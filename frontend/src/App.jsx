import React, { useEffect, useState } from "react";
import "./index.css";

// Backend shared by website + Zoho SalesIQ bot
const BACKEND_URL =
  import.meta.env?.VITE_API_BASE_URL ||
  "https://salesiq-ai-assistant-8m8r.onrender.com";

// Shared demo user id (matches Deluge visitor_id)
const DEMO_USER_ID = "demo-user";

function App() {
  // HEALTH
  const [health, setHealth] = useState({ status: "checking..." });

  // PRODUCTS
  const [products, setProducts] = useState([]);
  const [productsLoading, setProductsLoading] = useState(true);
  const [productsError, setProductsError] = useState("");

  // CART (synced with backend / AI engine)
  const [cart, setCart] = useState([]);
  const [cartLoading, setCartLoading] = useState(false);
  const [cartError, setCartError] = useState("");

  // ORDER TRACKER
  const [orderId, setOrderId] = useState("");
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState("");
  const [orderData, setOrderData] = useState(null);

  // On load: health + products + cart
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
    fetchProducts();
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
        {/* ... your existing hero, order tracker, howto, evaluators sections stay same ... */}

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
                  <div className="product-tag">
                    {p.tag ? p.tag : "From API"}
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
                      Add to Cart (Sync with Bot)
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* CART SECTION (SYNCED WITH BOT) */}
        <section id="cart" className="section">
          <div className="section-header">
            <h2>Shared Cart (Chatbot + Frontend)</h2>
            <p>
              This cart is backed by the same memory used by the Zoho SalesIQ
              bot. Add items in chat using{" "}
              <code>add 101 to cart</code> and click{" "}
              <b>Sync from Chatbot</b> to see them here.
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

        {/* (Keep your existing Order Tracker, How to Test, Evaluators sections as they are) */}
      </main>

      {/* FOOTER stays same */}
    </div>
  );
}

export default App;
