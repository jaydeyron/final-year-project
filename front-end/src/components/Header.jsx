import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.css';
import logo from '../assets/placeholder2.svg';

function Header() {
  return (
    <header className="header">
      <div className="header-left">
        <span className="project-name">SPORS</span>
      </div>
      <div className="header-middle">
        <Link to="/symbol?name=SENSEX" className="symbol">SENSEX</Link>
        <Link to="/symbol?name=TCS" className="symbol">TCS</Link>
        <Link to="/symbol?name=HDFCBANK" className="symbol">HDFC BANK</Link>
      </div>
      <nav className="header-right">
        <Link to="/home">Home</Link>
        <Link to="/settings">Settings</Link>
      </nav>
    </header>
  );
}

export default Header;