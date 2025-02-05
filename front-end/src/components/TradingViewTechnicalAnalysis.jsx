import React, { useEffect, useRef } from 'react';

function TradingViewTechnicalAnalysis({ symbol }) {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = '';

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "interval": "1D",
        "width": "100%",
        "isTransparent": true,
        "height": "500",
        "symbol": "${symbol}",
        "showIntervalTabs": true,
        "locale": "en",
        "colorTheme": "dark",
        "backgroundColor": "transparent"
      }`;

    containerElement.appendChild(script);

    return () => {
      containerElement.innerHTML = '';
    };
  }, [symbol]);

  return (
    <div className="tradingview-widget-container" ref={container}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  );
}

export default TradingViewTechnicalAnalysis;