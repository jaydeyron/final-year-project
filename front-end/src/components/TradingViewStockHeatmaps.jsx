import React, { useEffect, useRef } from 'react';

function TradingViewStockHeatmaps() {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "exchanges": [],
        "dataSource": "SENSEX",
        "grouping": "sector",
        "blockSize": "market_cap_basic",
        "blockColor": "change",
        "isTransparent": true,
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": false,
        "isDataSetEnabled": true,
        "isZoomEnabled": false,
        "hasSymbolTooltip": true,
        "isMonoSize": false,
        "width": "60%",
        "height": "550"
      }
    `;

    containerElement.appendChild(script);
  }, []);

  return (
    <div className="tradingview-widget-container" ref={container}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  );
}

export default TradingViewStockHeatmaps;