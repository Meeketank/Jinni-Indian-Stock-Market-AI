"""
Stock Recommender System - AI-powered stock selection from top NSE/BSE stocks
Scans multiple stocks and provides ranked BUY recommendations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import concurrent.futures
from analytics.trading_signals import TradingSignals
from analytics.technical_indicators import TechnicalIndicators

class StockRecommender:
    """
    Intelligent stock recommendation system that scans top stocks
    and provides ranked opportunities based on AI predictions
    """
    
    # Top NSE stocks to scan
    NSE_TOP_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'AXISBANK.NS',
        'LT.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS',
        'WIPRO.NS', 'M&M.NS', 'TECHM.NS', 'POWERGRID.NS', 'NTPC.NS',
        'TATASTEEL.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'COALINDIA.NS', 'TATAMOTORS.NS'
    ]
    
    def __init__(self):
        self.trading_signals = TradingSignals()
        self.technical_indicators = TechnicalIndicators()
        
    def _fetch_stock_data(self, symbol: str, period: str = '3mo') -> Tuple[str, pd.DataFrame, bool]:
        """
        Fetch historical data for a single stock
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if len(data) < 50:  # Need minimum data
                return symbol, None, False
                
            return symbol, data, True
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return symbol, None, False
    
    def _calculate_simple_prediction(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Quick prediction using momentum and trend analysis
        """
        current_price = data['Close'].iloc[-1]
        
        # Calculate multiple timeframe trends
        sma_5 = data['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        # Momentum calculation
        returns = data['Close'].pct_change()
        momentum_5d = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]
        momentum_20d = (current_price - data['Close'].iloc[-21]) / data['Close'].iloc[-21]
        
        # Volume trend
        volume_ratio = data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:].mean()
        
        # Prediction logic
        prediction_factors = []
        
        # Trend alignment
        if current_price > sma_5 > sma_20 > sma_50:
            prediction_factors.append(0.05)  # Strong uptrend
        elif current_price > sma_20 > sma_50:
            prediction_factors.append(0.03)  # Moderate uptrend
        elif current_price > sma_50:
            prediction_factors.append(0.01)  # Weak uptrend
        else:
            prediction_factors.append(-0.02)  # Downtrend
        
        # Momentum contribution
        if momentum_5d > 0.02 and momentum_20d > 0:
            prediction_factors.append(0.03)
        elif momentum_5d > 0:
            prediction_factors.append(0.01)
        else:
            prediction_factors.append(-0.01)
        
        # Volume confirmation
        if volume_ratio > 1.2:
            prediction_factors.append(0.02)  # High volume support
        elif volume_ratio > 1.0:
            prediction_factors.append(0.01)
        
        # Calculate predicted change
        predicted_change = sum(prediction_factors)
        predicted_price = current_price * (1 + predicted_change)
        
        # Confidence calculation
        volatility = returns.std()
        if volatility < 0.015:
            confidence = 0.85
        elif volatility < 0.025:
            confidence = 0.75
        else:
            confidence = 0.65
        
        # Adjust confidence based on trend clarity
        if len([f for f in prediction_factors if f > 0]) >= 2:
            confidence += 0.05
        
        return round(predicted_price, 2), min(confidence, 0.90)
    
    def _analyze_single_stock(self, symbol: str) -> Dict:
        """
        Complete analysis of a single stock
        """
        # Fetch data
        symbol, data, success = self._fetch_stock_data(symbol)
        
        if not success or data is None:
            return None
        
        try:
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Calculate technical indicators
            indicators = self.technical_indicators.calculate_all(data)
            
            # Get prediction
            predicted_price, confidence = self._calculate_simple_prediction(data)
            
            # Generate trading signal
            signal = self.trading_signals.generate_trading_signal(
                current_price=current_price,
                predicted_price=predicted_price,
                prediction_confidence=confidence,
                technical_indicators=indicators,
                historical_data=data,
                recommendation='HOLD'  # Initial, will be overridden
            )
            
            # Extract stock name
            stock_name = symbol.replace('.NS', '').replace('.BO', '')
            
            result = {
                'symbol': symbol,
                'name': stock_name,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'potential_gain_pct': signal.get('potential_gain_pct', 0),
                'action': signal['action'],
                'confidence': signal['confidence'],
                'entry_price': signal.get('entry_price', current_price),
                'target_1': signal.get('target_1', 0),
                'target_2': signal.get('target_2', 0),
                'target_3': signal.get('target_3', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
                'reasoning': signal.get('reasoning', ''),
                'rsi': indicators.get('RSI', 50),
                'macd_signal': indicators.get('MACD_Signal', 'NEUTRAL'),
                'volume_trend': 'High' if data['Volume'].iloc[-1] > data['Volume'].iloc[-20:].mean() else 'Normal'
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def get_top_recommendations(self, 
                               count: int = 10,
                               min_gain: float = 2.0,
                               action_filter: List[str] = ['BUY', 'STRONG BUY']) -> List[Dict]:
        """
        Scan all stocks and return top recommendations
        This is what user will see - TOP 10 stocks to BUY NOW
        """
        print("🔍 Scanning top NSE stocks for opportunities...")
        
        recommendations = []
        
        # Parallel processing for faster scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(self._analyze_single_stock, symbol): symbol 
                              for symbol in self.NSE_TOP_STOCKS}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                result = future.result()
                if result is not None:
                    recommendations.append(result)
        
        # Filter by action (only BUY signals)
        if action_filter:
            recommendations = [r for r in recommendations if r['action'] in action_filter]
        
        # Filter by minimum gain
        recommendations = [r for r in recommendations if r['potential_gain_pct'] >= min_gain]
        
        # Sort by potential gain (descending)
        recommendations.sort(key=lambda x: (
            x['potential_gain_pct'], 
            x['risk_reward_ratio'],
            self._confidence_score(x['confidence'])
        ), reverse=True)
        
        # Return top N
        top_recommendations = recommendations[:count]
        
        print(f"✅ Found {len(top_recommendations)} high-potential opportunities")
        
        return top_recommendations
    
    def _confidence_score(self, confidence_str: str) -> float:
        """
        Convert confidence string to numeric score
        """
        confidence_map = {
            'HIGH': 1.0,
            'MEDIUM-HIGH': 0.8,
            'MEDIUM': 0.6,
            'LOW-MEDIUM': 0.4,
            'LOW': 0.2
        }
        return confidence_map.get(confidence_str, 0.5)
    
    def get_sector_recommendations(self, sector: str, count: int = 5) -> List[Dict]:
        """
        Get recommendations filtered by sector
        """
        sector_stocks = {
            'BANKING': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
            'AUTO': ['MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS'],
            'ENERGY': ['RELIANCE.NS', 'ONGC.NS', 'COALINDIA.NS', 'NTPC.NS', 'POWERGRID.NS']
        }
        
        stocks_to_scan = sector_stocks.get(sector.upper(), [])
        
        if not stocks_to_scan:
            return []
        
        recommendations = []
        for symbol in stocks_to_scan:
            result = self._analyze_single_stock(symbol)
            if result and result['action'] in ['BUY', 'STRONG BUY']:
                recommendations.append(result)
        
        recommendations.sort(key=lambda x: x['potential_gain_pct'], reverse=True)
        return recommendations[:count]
    
    def get_quick_summary(self) -> Dict:
        """
        Quick market overview with top 3 picks
        """
        top_3 = self.get_top_recommendations(count=3, min_gain=1.0)
        
        if not top_3:
            return {
                'status': 'No strong opportunities found',
                'top_picks': [],
                'market_sentiment': 'NEUTRAL'
            }
        
        avg_gain = np.mean([r['potential_gain_pct'] for r in top_3])
        
        sentiment = 'BULLISH' if avg_gain > 5 else 'NEUTRAL' if avg_gain > 2 else 'CAUTIOUS'
        
        return {
            'status': 'Active opportunities available',
            'top_picks': top_3,
            'market_sentiment': sentiment,
            'average_potential_gain': round(avg_gain, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
