import React from 'react';
import { useSearchParams } from 'react-router-dom';
import TradingViewSymbolInfo from '../components/TradingViewSymbolInfo';
import TradingViewChart from '../components/TradingViewChart';
import TradingViewCompanyProfile from '../components/TradingViewCompanyProfile';
import TradingViewFinancials from '../components/TradingViewFinancials';
import TradingViewTechnicalAnalysis from '../components/TradingViewTechnicalAnalysis';
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
      <TradingViewTechnicalAnalysis symbol={fullSymbol} />
    </div>
  );
}

export default Symbol;