import React, { useState, useRef, useEffect } from "react";
import "./ChatBox.css";

interface Message {
  sender: "user" | "bot";
  text?: string;
  file?: File;
}

const ChatBox: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    setMessages([...messages, { sender: "user", text: input }]);
    setInput("");

    // Fake bot reply
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Bot response..." },
      ]);
    }, 600);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setMessages([...messages, { sender: "user", file }]);
    setShowUpload(false);
  };

  return (
    <div className="chat-container">
      {/* Messages */}
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            {msg.text && <span>{msg.text}</span>}
            {msg.file && <div className="file-message">📎 {msg.file.name}</div>}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Upload panel */}
      <div className={`upload-panel ${showUpload ? "" : "hidden"}`}>
        <label>
          <input
            type="file"
            style={{ display: "none" }}
            onChange={handleFileUpload}
          />
          <button className="send-btn">Upload File</button>
        </label>
      </div>

      {/* Input Footer */}
      <div className="chat-input">
        <button
          className="upload-toggle"
          onClick={() => setShowUpload((prev) => !prev)}
        >
          +
        </button>
        <input
          className="message-box"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type a message..."
        />
        <button className="send-btn" onClick={handleSend}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatBox;
