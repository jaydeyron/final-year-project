import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Divider from './components/Divider';
import Home from './pages/Home';
import Settings from './pages/Settings';
import Symbol from './pages/Symbol';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Divider />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/home" element={<Home />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/symbol" element={<Symbol />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;