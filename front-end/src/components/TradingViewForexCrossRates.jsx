import React, { useEffect, useRef } from 'react';

function TradingViewForexCrossRates() {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-forex-cross-rates.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "width": "100%",
        "height": "550",
        "currencies": [
          "EUR",
          "USD",
          "INR",
          "JPY",
          "GBP",
          "AUD",
          "CAD"
        ],
        "isTransparent": true,
        "colorTheme": "dark",
        "locale": "en",
        "backgroundColor": "#000000"
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

export default TradingViewForexCrossRates;