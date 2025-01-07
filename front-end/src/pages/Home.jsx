import React from 'react';
import TradingViewStockMarket from '../components/TradingViewStockMarket';
import TradingViewTopStories from '../components/TradingViewTopStories';
import TradingViewMarketOverview from '../components/TradingViewMarketOverview';
import TradingViewTickerTape from '../components/TradingViewTickerTape';
import TradingViewCryptocurrencyMarket from '../components/TradingViewCryptocurrencyMarket';
import TradingViewEconomicCalendar from '../components/TradingViewEconomicCalendar';
import TradingViewStockHeatmaps from '../components/TradingViewStockHeatmaps';
import TradingViewForexCrossRates from '../components/TradingViewForexCrossRates';
import '../styles/Home.css';

function Home() {
  return (
    <div>
      <TradingViewTickerTape />
      <h2>Market Summary</h2>
      <div className='flex-container'>
        <TradingViewMarketOverview />
        <div style={{ flex: '0 0 30%' }}>
          <TradingViewStockMarket />
        </div>
      </div>
      <div className='flex-container'>
        <TradingViewEconomicCalendar />
        <TradingViewStockHeatmaps />
        <TradingViewTopStories symbol="all_symbols" />
      </div>
      <div className='flex-container'>
        <TradingViewCryptocurrencyMarket />      
        <TradingViewForexCrossRates />
      </div>
    </div>
  );
}

export default Home;