import React from 'react';

const TradingViewTechnicalAnalysis = ({ symbol }) => {
  return (
    <div className="tradingview-widget-container">
      <div className="tradingview-widget-container__widget"></div>
      <script
        type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js"
        async
      >
        {`
          {
            "interval": "1m",
            "width": 425,
            "isTransparent": true,
            "height": 450,
            "symbol": "${symbol}",
            "showIntervalTabs": true,
            "displayMode": "single",
            "locale": "en",
            "colorTheme": "dark"
          }
        `}
      </script>
    </div>
  );
};

export default TradingViewTechnicalAnalysis;