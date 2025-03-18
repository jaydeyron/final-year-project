import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { niftyStocks } from '../data/niftyStocks';
import TradingViewSymbolInfo from '../components/TradingViewSymbolInfo';
import TradingViewChart from '../components/TradingViewChart';
import TradingViewCompanyProfile from '../components/TradingViewCompanyProfile';
import TradingViewFinancials from '../components/TradingViewFinancials';
import TradingViewTechnicalAnalysis from '../components/TradingViewTechnicalAnalysis';
import TFTModel from '../components/TFTModel';
import '../styles/Symbol.css';

function Symbol() {
  const [searchParams] = useSearchParams();
  const symbolName = searchParams.get('name');
  
  // Find the stock in niftyStocks to get its TradingView symbol
  const stock = niftyStocks.find(s => s.symbol === symbolName);
  const tradingViewSymbol = stock ? stock.tradingViewSymbol : (symbolName === 'SENSEX' ? symbolName : `BSE:${symbolName}`);

  return (
    <div className="symbol-page-container">
      <TradingViewSymbolInfo symbol={tradingViewSymbol} />
      <TradingViewChart symbol={tradingViewSymbol} />
      <TradingViewCompanyProfile symbol={tradingViewSymbol} />
      <TradingViewFinancials symbol={tradingViewSymbol} />
      <TradingViewTechnicalAnalysis symbol={tradingViewSymbol} />
      <TFTModel symbol={symbolName} /> 
    </div>
  );
}

export default Symbol;