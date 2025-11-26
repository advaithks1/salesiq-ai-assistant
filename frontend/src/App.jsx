// App.jsx — SalesIQ AI Assistant — Hackathon Edition (Full + History Restored)
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FiSend, FiPaperclip, FiUser, FiRefreshCw } from "react-icons/fi";
import { FaBolt } from "react-icons/fa";

const API_BASE = "https://salesiq-ai-assistant-8m8r.onrender.com";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [selectedMsg, setSelectedMsg] = useState(null);

  const [stats, setStats] = useState({
    total: 0,
    intents: {},
    emotions: {},
    priority: { high: 0, medium: 0, low: 0 },
    escalations: 0,
  });

  const scrollRef = useRef();

  const QUICK_REPLIES = [
    "Track my order",
    "Refund status",
    "Escalate to agent",
    "Pricing details",
    "Product availability",
  ];

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  const makeId = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`;

  const safeNumber = (v, fallback = 0) => {
    const n = Number(v);
    return Number.isNaN(n) ? fallback : n;
  };

  const sendMessage = async (text) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    const userMsg = {
      id: makeId(),
      sender: "user",
      text: trimmed,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setTyping(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        user_id: "demo_user",
        message: trimmed,
      });

      const bot = res?.data || {};
      const meta = bot.metadata || {};

      const botMsg = {
        id: makeId(),
        sender: "bot",
        text: bot.answer || "—",
        intent: bot.intent || "unknown",
        emotion: bot.emotion || "neutral",
        priority: bot.priority || "low",
        escalations: safeNumber(bot.escalations || 0),
        matched: bot.matched_question || bot.matched || null,
        missing_info: bot.missing_info || null,

        raw: {
          confidence: safeNumber(meta.confidence, 0),
          similarity: safeNumber(meta.similarity, 0),
          risk: meta.risk || "low",
          autoflow: meta.autoflow || false,
          field_required: meta.field_required || null,
          hint: meta.hint || null,
        },
      };

      setTimeout(() => {
        setMessages((prev) => [...prev, botMsg]);
        setTyping(false);
        setSelectedMsg(botMsg);

        setStats((prev) => {
          const intents = { ...prev.intents };
          intents[botMsg.intent] = (intents[botMsg.intent] || 0) + 1;

          const emotions = { ...prev.emotions };
          emotions[botMsg.emotion] = (emotions[botMsg.emotion] || 0) + 1;

          const priority = { ...prev.priority };
          priority[botMsg.priority] = (priority[botMsg.priority] || 0) + 1;

          return {
            ...prev,
            total: prev.total + 1,
            intents,
            emotions,
            priority,
            escalations: prev.escalations + botMsg.escalations,
          };
        });
      }, 300);
    } catch (e) {
      const errMsg = {
        id: makeId(),
        sender: "bot",
        text: "⚠️ Server unreachable.",
        intent: "error",
        emotion: "sad",
        raw: { confidence: 0, similarity: 0, risk: "high" },
      };
      setMessages((prev) => [...prev, errMsg]);
      setTyping(false);
    }
  };

  const getEmoji = (e) => {
    switch (e) {
      case "happy":
        return "😊";
      case "sad":
        return "😢";
      case "angry":
        return "😡";
      case "confused":
        return "😕";
      default:
        return "😐";
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="h-screen flex bg-gray-50 font-sans text-sm">

      {/* LEFT SIDEBAR */}
      <aside className="w-80 p-4 border-r bg-white flex flex-col gap-4">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-xl">
              <FaBolt />
            </div>
            <div>
              <div className="text-lg font-bold">SalesIQ AI</div>
              <div className="text-xs text-gray-500">Realtime assistant dashboard</div>
            </div>
          </div>

          <button
            onClick={() => window.location.reload()}
            className="p-2 rounded bg-gray-100 hover:bg-gray-200"
          >
            <FiRefreshCw />
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-white rounded-xl shadow border text-center">
            <div className="text-xs text-gray-500">Total</div>
            <div className="text-lg font-bold">{stats.total}</div>
          </div>
          <div className="p-3 bg-white rounded-xl shadow border text-center">
            <div className="text-xs text-gray-500">Escalations</div>
            <div className="text-lg font-bold">{stats.escalations}</div>
          </div>
        </div>

        {/* Intents */}
        <div>
          <div className="text-xs text-gray-500 mb-1">Top Intents</div>
          {!Object.keys(stats.intents).length && (
            <div className="text-gray-400">No data yet</div>
          )}
          {Object.entries(stats.intents).map(([k, v]) => (
            <div key={k} className="flex justify-between text-sm">
              <span>{k}</span>
              <span>{v}</span>
            </div>
          ))}
        </div>

        {/* HISTORY (RESTORED) */}
        <div className="mt-4">
          <div className="text-xs text-gray-500 mb-2">Recent Messages</div>

          <div className="flex flex-col gap-2 max-h-56 overflow-auto">

            {messages
              .slice()
              .reverse()
              .slice(0, 8)
              .map((m) => (
                <div
                  key={m.id}
                  className="p-2 border rounded hover:bg-gray-50 cursor-pointer"
                  onClick={() => setSelectedMsg(m)}
                >
                  <div className="text-xs text-gray-600 truncate">
                    {m.sender === "user" ? "User" : "Bot"}: {m.text}
                  </div>

                  {m.sender === "bot" && (
                    <div className="text-xs text-gray-400 mt-1">
                      {m.intent} • {m.emotion} •{" "}
                      {safeNumber(m.raw?.confidence).toFixed(2)}
                    </div>
                  )}
                </div>
              ))}

            {!messages.length && (
              <div className="text-gray-400">No messages yet</div>
            )}
          </div>
        </div>

        {/* Tips */}
        <div className="mt-auto text-xs text-gray-500">
          <div className="mb-1">Demo Tips:</div>
          <ul className="list-disc pl-5 space-y-1">
            <li>Demo Autoflow: “Track my order”</li>
            <li>Demo KG: Ask related questions</li>
            <li>Click messages to see metadata</li>
          </ul>
        </div>
      </aside>

      {/* MAIN CHAT */}
      <main className="flex-1 p-6 flex flex-col">
        <div className="text-2xl font-bold mb-2">Support Chat</div>
        <div className="text-xs text-gray-500 mb-4">Live assistant — demo mode</div>

        {/* Chat window */}
        <div className="flex-1 overflow-auto border rounded p-4 bg-white">
          <div className="flex flex-col gap-4">

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`max-w-xl ${
                  msg.sender === "user" ? "self-end text-right" : "self-start"
                }`}
                onClick={() => setSelectedMsg(msg)}
              >
                <div
                  className={`inline-block px-4 py-2 rounded-lg shadow-sm ${
                    msg.sender === "user" ? "bg-blue-100" : "bg-gray-100"
                  }`}
                >
                  {msg.text}
                </div>

                {/* BOT METADATA */}
                {msg.sender === "bot" && (
                  <>
                    <div className="text-xs text-gray-600 mt-1">
                      {getEmoji(msg.emotion)} {msg.emotion} | {" "}
                      Intent: {msg.intent} | {" "}
                      Confidence: {safeNumber(msg.raw?.confidence).toFixed(2)} | {" "}
                      Similarity: {safeNumber(msg.raw?.similarity).toFixed(2)} | {" "}
                      Risk: {msg.raw?.risk} | {" "}
                      Priority: {msg.priority}
                    </div>

                    {/* Autoflow / KG UI */}
                    {msg.raw && (
                      <div className="meta-box">
                        {msg.raw.autoflow && (
                          <div className="tag autoflow">
                            🔄 Autoflow Active — Missing:{" "}
                            {msg.missing_info || msg.raw.field_required}
                          </div>
                        )}

                        {msg.raw.field_required && (
                          <div className="tag required">
                            📝 Required: {msg.raw.field_required}
                          </div>
                        )}

                        {msg.raw.hint && (
                          <div className="tag hint">
                            💡 Hint: {msg.raw.hint}
                          </div>
                        )}

                        <div className="tag score">
                          Similarity Score: {safeNumber(msg.raw.similarity).toFixed(2)}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}

            {typing && (
              <div className="self-start">
                <div className="inline-block px-4 py-2 bg-gray-200 rounded animate-pulse">
                  Typing…
                </div>
              </div>
            )}

            <div ref={scrollRef}></div>
          </div>
        </div>

        {/* QUICK REPLIES */}
        <div className="mt-3 flex gap-2 flex-wrap">
          {QUICK_REPLIES.map((q) => (
            <button
              key={q}
              onClick={() => sendMessage(q)}
              className="px-3 py-1 rounded-full bg-green-500 text-white text-xs hover:bg-green-600 shadow"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="mt-4 bg-white p-3 rounded shadow flex items-center gap-3">
          <button className="p-2 rounded bg-gray-100">
            <FiPaperclip />
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Type your message..."
            className="flex-1 border rounded p-2 h-12 resize-none"
          />

          <button
            onClick={() => sendMessage(input)}
            className="px-4 py-2 bg-blue-600 text-white rounded flex items-center gap-2 hover:bg-blue-700"
          >
            <FiSend /> Send
          </button>
        </div>
      </main>

      {/* RIGHT PANEL */}
      <aside className="w-96 p-4 border-l bg-white flex flex-col gap-4">
        <div className="text-sm font-semibold">Message Insights</div>

        {!selectedMsg && <div className="text-gray-400">Click a bot message.</div>}

        {selectedMsg && (
          <div className="space-y-3">
            <div className="p-3 bg-gray-50 rounded border">
              <div className="text-xs text-gray-500">Text</div>
              <div className="mt-1">{selectedMsg.text}</div>
            </div>

            {selectedMsg.sender === "bot" && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 border rounded">
                    <div className="text-xs text-gray-500">Intent</div>
                    <div>{selectedMsg.intent}</div>
                  </div>
                  <div className="p-2 border rounded">
                    <div className="text-xs text-gray-500">Emotion</div>
                    <div>{selectedMsg.emotion}</div>
                  </div>
                  <div className="p-2 border rounded">
                    <div className="text-xs text-gray-500">Confidence</div>
                    <div>
                      {safeNumber(selectedMsg.raw?.confidence).toFixed(2)}
                    </div>
                  </div>
                  <div className="p-2 border rounded">
                    <div className="text-xs text-gray-500">Similarity</div>
                    <div>
                      {safeNumber(selectedMsg.raw?.similarity).toFixed(2)}
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-white rounded border">
                  <div className="text-xs text-gray-500 mb-1">Autoflow</div>
                  <div>{selectedMsg.raw?.autoflow ? "Active" : "No"}</div>

                  <div className="text-xs text-gray-500 mt-3">Required Field</div>
                  <div>{selectedMsg.raw?.field_required || "—"}</div>

                  <div className="text-xs text-gray-500 mt-3">KG Hint</div>
                  <div>{selectedMsg.raw?.hint || "—"}</div>

                  <div className="text-xs text-gray-500 mt-3">KB Match</div>
                  <div>{selectedMsg.matched || "—"}</div>
                </div>
              </>
            )}

            <button
              onClick={() => setSelectedMsg(null)}
              className="w-full py-2 border rounded"
            >
              Close
            </button>
          </div>
        )}
      </aside>
    </div>
  );
}
