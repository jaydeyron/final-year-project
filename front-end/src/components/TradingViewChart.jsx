import React, { useEffect, useRef, memo } from 'react';

function TradingViewChart({ symbol }) {
  const container = useRef();

  useEffect(() => {
    // Clean up previous widget
    if (container.current) {
      container.current.innerHTML = '';
    }

    // Create widget container
    const widgetContainer = document.createElement('div');
    widgetContainer.className = 'tradingview-widget-container__widget';
    container.current.appendChild(widgetContainer);

    // Create and add script
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.type = 'text/javascript';
    script.async = true;
    script.innerHTML = `
      {
        "autosize": false,
        "symbol": "${symbol}",
        "interval": "D",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "width": "100%",
        "height": "800",
        "backgroundColor": "rgba(0, 0, 0, 1)",
        "gridLineColor": "rgba(0, 0, 0, 0)",
        "allow_symbol_change": true,
        "calendar": false,
        "hide_volume": false,
        "support_host": "https://www.tradingview.com"
      }`;
    container.current.appendChild(script);

    // Cleanup function
    return () => {
      if (container.current) {
        container.current.innerHTML = '';
      }
    };
  }, [symbol]); // Dependencies array includes symbol to reinitialize on symbol change

  return (
    <div className="tradingview-widget-container" ref={container} style={{ height: "800px" }} />
  );
}

export default memo(TradingViewChart);