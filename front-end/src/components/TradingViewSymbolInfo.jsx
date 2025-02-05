import React, { useEffect, useRef } from 'react';

const TradingViewSymbolInfo = ({ symbol }) => {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "symbol": "${symbol}",
        "width": "100%",
        "locale": "en",
        "colorTheme": "dark",
        "isTransparent": true,
        "backgroundColor": "transparent"
      }
    `;

    containerElement.appendChild(script);
  }, [symbol]);

  return (
    <div className="tradingview-widget-container" ref={container} style={{ width: '100%', height: 'auto' }}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  );
};

export default TradingViewSymbolInfo;