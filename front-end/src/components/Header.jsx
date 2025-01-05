import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.css';
import logo from '../assets/placeholder2.svg'; // Adjust the path if necessary

function Header() {
  return (
    <header className="header">
      <div className="header-left">
        <img src={logo} alt="Logo" className="logo" />
        <span className="project-name">SPORS</span>
      </div>
      <nav className="header-right">
        <Link to="/home">Home</Link>
        <Link to="/settings">Settings</Link>
      </nav>
    </header>
  );
}

export default Header;