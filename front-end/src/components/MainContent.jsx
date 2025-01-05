import React from "react";
import TradingViewChart from "./TradingViewChart";
import TradingViewStockMarket from "./TradingViewStockMarket";
import TradingViewSymbolInfo from "./TradingViewSymbolInfo";
import TradingViewCompanyProfile from "./TradingViewCompanyProfile";
import TradingViewFundamentalData from "./TradingViewFundamentalData";
import TradingViewTopStories from "./TradingViewTopStories";
import TradingViewTechnicalAnalysis from "./TradingViewTechnicalAnalysis";
import "./MainContent.css";

function MainContent({ selectedChart }) {
  return (
    <div className="main-content">
      {selectedChart ? (
        <>
          <div className="symbol-info">
            <TradingViewSymbolInfo symbol={selectedChart} />
          </div>
          <div className="chart-profile">
            <div className="company-profile">
              <TradingViewCompanyProfile symbol={selectedChart} />
            </div>
            <div className="chart-container">
              {/* <h2>{selectedChart} Chart</h2> */}
              <TradingViewChart symbol={selectedChart} />
            </div>
          </div>
          <div className="fundamental-data">
              <TradingViewFundamentalData symbol={selectedChart} />
          </div>
          <div className="row">
            {/* <div className="top-stories">
              <TradingViewTopStories symbol={selectedChart} />
            </div> */}
            {/* <div className="technical-analysis">
              <TradingViewTechnicalAnalysis symbol={selectedChart} />
            </div> */}
          </div>
        </>
      ) : (
        <div className="placeholder">
          <TradingViewStockMarket />
        </div>
      )}
    </div>
  );
}

export default MainContent;
