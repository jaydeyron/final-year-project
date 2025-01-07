import React, { useState } from 'react';
import '../styles/Menu.css';

const Menu = ({ symbols, indices, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const handleSelect = (item) => {
    onSelect(item);
    toggleMenu();
  };

  return (
    <div className={`menu-container ${isOpen ? 'open' : ''}`}>
      <button className="menu-toggle" onClick={toggleMenu}>
        {isOpen ? 'Close Menu' : 'Open Menu'}
      </button>
      {isOpen && (
        <div className="menu-overlay">
          <div className="menu-content">
            <h3>Select Symbols</h3>
            <ul>
              {symbols.map((symbol) => (
                <li key={symbol} onClick={() => handleSelect(symbol)}>
                  {symbol}
                </li>
              ))}
            </ul>
            <h3>Select Indices</h3>
            <ul>
              {indices.map((index) => (
                <li key={index} onClick={() => handleSelect(index)}>
                  {index}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default Menu;