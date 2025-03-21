import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { niftyStocks } from '../data/niftyStocks';
import '../styles/Header.css';

function Header() {
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [filteredStocks, setFilteredStocks] = useState([]);
  const navigate = useNavigate();
  const searchRef = useRef(null);
  const [isExiting, setIsExiting] = useState(false);
  const [inputRef, setInputRef] = useState(null); // Add this state

  useEffect(() => {
    if (showDropdown && !searchTerm) {
      setFilteredStocks(niftyStocks);
    } else if (searchTerm) {
      const filtered = niftyStocks.filter(stock =>
        stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredStocks(filtered);
    }
  }, [searchTerm, showDropdown]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.body.addEventListener('mousedown', handleClickOutside);
    return () => document.body.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleStockSelect = (symbol) => {
    navigate(`/symbol?name=${encodeURIComponent(symbol)}`);
    setSearchTerm('');
    setShowDropdown(false);
    inputRef?.blur(); // Remove focus from input
  };

  const handleDropdownClose = () => {
    setIsExiting(true);
    setTimeout(() => {
      setShowDropdown(false);
      setIsExiting(false);
    }, 300); // Match the transition duration
  };

  return (
    <header className="header">
      <div className="header-left">
        <span className="project-name">SPORS</span>
      </div>
      <div className="header-middle">
        <div className="symbols-group" ref={searchRef}>
          <div className="search-container">
            <input
              ref={ref => setInputRef(ref)}  // Add this ref
              type="text"
              placeholder="Search symbols..."
              className="search-input"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onFocus={() => {
                setShowDropdown(true);
                setFilteredStocks(niftyStocks);
              }}
              onBlur={handleDropdownClose}
            />
            {showDropdown && filteredStocks.length > 0 && (
              <div
                className={`search-dropdown ${isExiting ? 'exiting' : ''}`}
                onMouseDown={(e) => {
                  // Prevent onBlur from firing when clicking dropdown items
                  e.preventDefault();
                }}
              >
                {filteredStocks.map((stock) => (
                  <div
                    key={stock.symbol}
                    className="search-item"
                    onMouseDown={() => handleStockSelect(stock.symbol)}
                  >
                    {stock.name} ({stock.symbol})
                  </div>
                ))}
              </div>
            )}
          </div>
          <Link to="/symbol?name=SENSEX" className="symbol">SENSEX</Link>
          <Link to="/symbol?name=TCS" className="symbol">TCS</Link>
        </div>
      </div>
      <nav className="header-right">
        <Link to="/home">Home</Link>
        <Link to="/about">About</Link>
      </nav>
    </header>
  );
}

export default Header;