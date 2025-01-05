// src/App.js
import React, { useState } from "react";
import Header from "./components/Header";
import MainContent from "./components/MainContent";
import Sidebar from "./components/Sidebar";
import "./App.css";

function App() {
  const [isSidebarVisible, setSidebarVisible] = useState(false);
  const [selectedChart, setSelectedChart] = useState(null);

  const toggleSidebar = () => {
    setSidebarVisible(!isSidebarVisible);
  };

  const handleSelectChart = (chart) => {
    setSelectedChart(chart);
    setSidebarVisible(false); // Collapse the sidebar
  };

  return (
    <div className="app-container">
      <Header />
      <button onClick={toggleSidebar} className="toggle-sidebar-button">
        <span className="material-icons">menu</span>
      </button>
      <Sidebar isVisible={isSidebarVisible} onSelectChart={handleSelectChart} />
      <div className="content-container">
        <MainContent selectedChart={selectedChart} />
      </div>
    </div>
  );
}

export default App;
