// src/components/Header.js
import React from "react";
import "./Header.css";

function Header() {
  return (
    <header className="header">
      <div className="logo">SPORS</div>
      <nav className="nav-links">
        <a href="#">Dashboard</a>
        <a href="#">History</a>
        <a href="#">Settings</a>
      </nav>
    </header>
  );
}

export default Header;
