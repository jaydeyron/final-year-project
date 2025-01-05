import React from "react";
import TradingViewChart from "./TradingViewChart";
import StockMarketWidget from "./StockMarketWidget";
import "./MainContent.css";

function MainContent({ selectedChart }) {
  return (
    <div className="main-content">
      {selectedChart ? (
        <div className="chart-container">
          <h2>{selectedChart} Chart</h2>
          <TradingViewChart symbol={selectedChart} />
        </div>
      ) : (
        <div className="placeholder">
          <StockMarketWidget />
        </div>
      )}
    </div>
  );
}

export default MainContent;
