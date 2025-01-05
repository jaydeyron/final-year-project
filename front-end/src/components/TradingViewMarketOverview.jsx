import React, { useEffect, useRef } from 'react';

function TradingViewMarketOverview() {
  const container = useRef();

  useEffect(() => {
    const containerElement = container.current;
    containerElement.innerHTML = ''; // Clear previous widget

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "colorTheme": "dark",
        "dateRange": "12M",
        "showChart": true,
        "locale": "en",
        "width": "100%",
        "height": "500",
        "largeChartUrl": "",
        "isTransparent": true,
        "showSymbolLogo": true,
        "showFloatingTooltip": false,
        "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
        "plotLineColorFalling": "rgba(41, 98, 255, 1)",
        "gridLineColor": "rgba(42, 46, 57, 0)",
        "scaleFontColor": "rgba(209, 212, 220, 1)",
        "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
        "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
        "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
        "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
        "symbolActiveColor": "rgba(41, 98, 255, 0.12)",
        "tabs": [
          {
            "title": "Indices",
            "symbols": [
              { "s": "BSE:SENSEX", "d": "S&P SENSEX index" },
              { "s": "FOREXCOM:SPXUSD", "d": "S&P 500 Index" },
              { "s": "FOREXCOM:NSXUSD", "d": "US 100 Cash CFD" },
              { "s": "FOREXCOM:DJI", "d": "Dow Jones Industrial" },
              { "s": "INDEX:NKY", "d": "Japan 225" },
              { "s": "INDEX:DEU40", "d": "DAX Index" },
              { "s": "FOREXCOM:UKXGBP", "d": "FTSE 100 Index" }
            ],
            "originalTitle": "Indices"
          },
          {
            "title": "Futures",
            "symbols": [
              { "s": "CME:6E1!", "d": "Euro" },
              { "s": "COMEX:GC1!", "d": "Gold" },
              { "s": "NYMEX:NG1!", "d": "Gas" },
              { "s": "CBOT:ZC1!", "d": "Corn" },
              { "s": "BSE:BSX1!", "d": "SENSEX FUT" },
              { "s": "CME_MINI:ES1!", "d": "S&P 500" },
              { "s": "NYMEX:CL1!", "d": "WTI Crude Oil" }
            ],
            "originalTitle": "Futures"
          },
          {
            "title": "Bonds",
            "symbols": [
              { "s": "CBOT:ZB1!", "d": "T-Bond" },
              { "s": "CBOT:UB1!", "d": "Ultra T-Bond" },
              { "s": "EUREX:FGBL1!", "d": "Euro Bund" },
              { "s": "EUREX:FBTP1!", "d": "Euro BTP" },
              { "s": "EUREX:FGBM1!", "d": "Euro BOBL" },
              { "s": "BSE:SGBDE31III", "d": "Gold" }
            ],
            "originalTitle": "Bonds"
          },
          {
            "title": "Forex",
            "symbols": [
              { "s": "FX:EURUSD", "d": "EUR to USD" },
              { "s": "FX:GBPUSD", "d": "GBP to USD" },
              { "s": "FX:USDJPY", "d": "USD to JPY" },
              { "s": "FX:USDCHF", "d": "USD to CHF" },
              { "s": "FX:AUDUSD", "d": "AUD to USD" },
              { "s": "FX_IDC:USDINR", "d": "USD to INR" },
              { "s": "FX:USDCAD", "d": "USD to CAD" }
            ],
            "originalTitle": "Forex"
          }
        ]
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

export default TradingViewMarketOverview;