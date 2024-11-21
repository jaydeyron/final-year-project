// src/components/Sidebar.js
import React from "react";
import "./Sidebar.css";

function Sidebar() {
  return (
    <div className="sidebar">
      <h2>INDICES</h2>
      <ul>
        <li>BANK NIFTY</li>
        <li>NIFTY 50</li>
        <li>SENSEX</li>
      </ul>
    </div>
  );
}

export default Sidebar;
