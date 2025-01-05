import React, { useEffect, useRef } from 'react';

const TradingViewCompanyProfile = ({ symbol }) => {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "width": "100%",
        "height": "auto",
        "isTransparent": true,
        "colorTheme": "dark",
        "symbol": "${symbol}",
        "locale": "en"
      }
    `;

    containerElement.appendChild(script);
  }, [symbol]);

  return (
    <div className="tradingview-widget-container" ref={container} style={{ width: '400px', height: '550px' }}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  );
};

export default TradingViewCompanyProfile;