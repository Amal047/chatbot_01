import React from "react";
import ChatBox from "./ChatBox";
import "./App.css";

const App: React.FC = () => {
  return (
    <div className="app">
      {/* Header */}
      <header className="header">AI Chatbot</header>

      {/* Chat area */}
      <main className="chat-area">
        <ChatBox />
      </main>
    </div>
  );
};

export default App;
