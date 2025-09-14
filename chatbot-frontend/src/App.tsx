import React from "react";
import ChatBox from "./ChatBox";
import FileUploader from "./FileUploader";
import "./App.css"; // We'll put full screen styles here

const App: React.FC = () => {
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>AI Chatbot</h1>
      </header>

      {/* Chat area */}
      <main className="app-main">
        <ChatBox />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <FileUploader />
      </footer>
    </div>
  );
};

export default App;
