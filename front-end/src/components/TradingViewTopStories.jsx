import React, { useEffect, useRef } from 'react';

function TradingViewTopStories({ symbol }) {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = '';

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-timeline.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "feedMode": "symbol",
        "symbol": "${symbol}",
        "colorTheme": "dark",
        "isTransparent": true,
        "displayMode": "regular",
        "width": "100%",
        "height": "500",
        "locale": "en"
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

export default TradingViewTopStories;