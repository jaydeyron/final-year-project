import React, { useEffect, useRef } from 'react';

const TradingViewFundamentalData = ({ symbol }) => {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-financials.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "isTransparent": true,
        "largeChartUrl": "",
        "displayMode": "regular",
        "width": "100%",
        "height": "100%",
        "colorTheme": "dark",
        "symbol": "${symbol}",
        "locale": "en"
      }
    `;

    containerElement.appendChild(script);
  }, [symbol]);

  return (
    <div className="tradingview-widget-container" ref={container} style={{ width: '100%', height: '100%' }}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  );
};

export default TradingViewFundamentalData;