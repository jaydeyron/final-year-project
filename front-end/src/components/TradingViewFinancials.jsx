import React, { useEffect, useRef } from 'react';

const TradingViewFinancials = ({ symbol }) => {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = '';

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-financials.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "symbol": "${symbol}",
        "width": "100%",
        "height": "800",
        "colorTheme": "dark",
        "isTransparent": true,
        "backgroundColor": "transparent",
        "displayMode": "regular",
        "largeChartUrl": "",
        "showVolume": true,
        "locale": "en"
      }
    `;

    containerElement.appendChild(script);
  }, [symbol]);

  return (
    <div className="tradingview-widget-container" ref={container} style={{ width: '100%', height: '800px', minHeight: '800px' }}>
      <div className="tradingview-widget-container__widget" style={{ height: '100%' }}></div>
    </div>
  );
};

export default TradingViewFinancials;