import React, { useEffect, useRef } from 'react';

const TradingViewTopStories = ({ symbol }) => {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-timeline.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "feedMode": "${symbol}",
        "isTransparent": true,
        "displayMode": "regular",
        "width": 400,
        "height": 550,
        "colorTheme": "dark",
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
}

export default TradingViewTopStories;