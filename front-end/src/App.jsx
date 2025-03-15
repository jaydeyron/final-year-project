import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header';
import Divider from './components/Divider';
import Home from './pages/Home';
import Settings from './pages/Settings';
import Symbol from './pages/Symbol';
import About from './pages/About';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Divider />
        <Routes>
          <Route path="/" element={<About />} />
          <Route path="/home" element={<Home />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/symbol" element={<Symbol />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;