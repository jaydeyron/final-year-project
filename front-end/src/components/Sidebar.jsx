// src/components/Sidebar.js
import React from "react";
import "./Sidebar.css";

function Sidebar({ isVisible, onSelectChart }) {
  return (
    <div className={`sidebar ${isVisible ? "visible" : ""}`}>
      <h2>SYMBOLS</h2>
      <ul>
        <li onClick={() => onSelectChart("BSE:SENSEX")}>SENSEX</li>
        <li onClick={() => onSelectChart("BSE:BANK")}>BANKEX</li>
        <li onClick={() => onSelectChart("BSE:HDFCBANK")}>HDFC BANK</li>
        <li onClick={() => onSelectChart("BSE:TCS")}>TCS</li>
      </ul>
    </div>
  );
}

export default Sidebar;
