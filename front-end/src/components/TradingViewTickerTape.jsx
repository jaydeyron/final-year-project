import React, { useEffect, useRef } from 'react';

function TradingViewTickerTape() {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "symbols": [
          { "proName": "FOREXCOM:SPXUSD", "title": "S&P 500 Index" },
          { "proName": "FX_IDC:EURUSD", "title": "EUR to USD" },
          { "proName": "BITSTAMP:BTCUSD", "title": "Bitcoin" },
          { "proName": "BITSTAMP:ETHUSD", "title": "Ethereum" },
          { "description": "USD to INR", "proName": "FX_IDC:USDINR" },
          { "description": "EUR to INR", "proName": "FX_IDC:EURINR" },
          { "description": "ADANI Enterprises", "proName": "BSE:ADANIENT" },
          { "description": "TATA Consultancy Services", "proName": "BSE:TCS" },
          { "description": "Japanese Index", "proName": "TVC:NI225" }
        ],
        "showSymbolLogo": true,
        "isTransparent": true,
        "displayMode": "adaptive",
        "colorTheme": "dark",
        "locale": "en"
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

export default TradingViewTickerTape;