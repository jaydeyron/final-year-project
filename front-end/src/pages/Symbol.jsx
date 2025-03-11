import React from 'react';
import { useSearchParams } from 'react-router-dom';
import TradingViewSymbolInfo from '../components/TradingViewSymbolInfo';
import TradingViewChart from '../components/TradingViewChart';
import TradingViewCompanyProfile from '../components/TradingViewCompanyProfile';
import TradingViewFinancials from '../components/TradingViewFinancials';
import TradingViewTechnicalAnalysis from '../components/TradingViewTechnicalAnalysis';
import TradingViewTopStories from '../components/TradingViewTopStories';
import '../styles/Symbol.css';

function Symbol() {
  const [searchParams] = useSearchParams();
  const symbolName = searchParams.get('name');
  const fullSymbol = symbolName === 'SENSEX' ? symbolName : `BSE:${symbolName}`;

  return (
    <div className="symbol-page-container">
      <TradingViewSymbolInfo symbol={fullSymbol} />
      <TradingViewChart symbol={fullSymbol} />
      <TradingViewCompanyProfile symbol={fullSymbol} />
      <TradingViewFinancials symbol={fullSymbol} />
      <div className="side-by-side-container">
        <div className="side-by-side-item">
          <TradingViewTechnicalAnalysis symbol={fullSymbol} />
        </div>
        <div className="side-by-side-item">
          <TradingViewTopStories symbol={fullSymbol} />
        </div>
      </div>
    </div>
  );
}

export default Symbol;