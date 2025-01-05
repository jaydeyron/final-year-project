import React, { useEffect, useRef } from 'react';

const TradingViewEconomicCalendar = () => {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-events.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "colorTheme": "dark",
        "isTransparent": true,
        "width": "400",
        "height": "550",
        "locale": "en",
        "importanceFilter": "0,1",
        "countryFilter": "ar,au,br,ca,cn,fr,de,in,id,it,jp,kr,mx,ru,sa,za,tr,gb,us,eu"
      }
    `;

    containerElement.appendChild(script);
  }, []);

  return (
    <div className="tradingview-widget-container" ref={container} style={{ width: '400px', height: '550px' }}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  );
};

export default TradingViewEconomicCalendar;