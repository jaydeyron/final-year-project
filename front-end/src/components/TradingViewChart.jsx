import React, { useEffect, useRef, memo } from 'react';

function TradingViewChart({ symbol }) {
  const container = useRef();

  useEffect(
    () => {
      const containerElement = container.current;
      containerElement.innerHTML = ''; // Clear previous widget

      const script = document.createElement("script");
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.type = "text/javascript";
      script.async = true;
      script.innerHTML = `
        {
          "width": "100%",
          "height": "700",
          "symbol": "${symbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "backgroundColor": "rgba(0, 0, 0, 1)",
          "gridColor": "rgba(66, 66, 66, 1)",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "allow_symbol_change": false,
          "calendar": false,
          "support_host": "https://www.tradingview.com"
        }`;
      containerElement.appendChild(script);
    },
    [symbol] // Re-run effect when symbol changes
  );

  return (
    <div className="tradingview-widget-container" ref={container} style={{ height: "100%", width: "100%" }}>
      <div className="tradingview-widget-container__widget" style={{ height: "calc(100% - 32px)", width: "100%" }}></div>
    </div>
  );
}

export default memo(TradingViewChart);