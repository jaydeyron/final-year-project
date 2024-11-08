// src/App.jsx
import React, { useState } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';

function App() {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const indices = ['Nifty 50', 'Bank Nifty', 'Sensex']; // Sample indices

  return (
    <div className="app">
      <Header />
      <div className="main-content">
        <Sidebar indices={indices} onSelectIndex={setSelectedIndex} />
        <Dashboard selectedIndex={selectedIndex} />
      </div>
    </div>
  );
}

export default App;
