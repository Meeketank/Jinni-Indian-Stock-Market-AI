"""
Trading Signals Module - Generates precise entry/exit points with targets
Fixes contradiction between predictions and recommendations
Provides BUY/SELL/STOP_LOSS/TARGET prices for actionable trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class TradingSignals:
    """
    Advanced trading signal generator with multi-level target calculation
    Provides precise entry, exit, stop loss, and target prices
    """
    
    def __init__(self, risk_reward_ratio: float = 2.0):
        self.risk_reward_ratio = risk_reward_ratio
        self.min_profit_threshold = 0.02  # Minimum 2% profit for BUY signal
        self.stop_loss_percentage = 0.05  # 5% stop loss
        
    def calculate_support_resistance(self, historical_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate dynamic support and resistance levels
        """
        if len(historical_data) < 20:
            # Fallback to simple calculation
            support = historical_data['Low'].min()
            resistance = historical_data['High'].max()
        else:
            # Use recent highs/lows with weighting
            recent_data = historical_data.tail(20)
            support = recent_data['Low'].rolling(window=5).min().iloc[-1]
            resistance = recent_data['High'].rolling(window=5).max().iloc[-1]
            
        return support, resistance
    
    def calculate_volatility_adjusted_targets(self, 
                                             current_price: float,
                                             volatility: float) -> Dict[str, float]:
        """
        Calculate targets based on volatility for better accuracy
        """
        # Adjust target multipliers based on volatility
        if volatility > 0.03:  # High volatility
            target_multipliers = [1.03, 1.06, 1.10]
        elif volatility > 0.015:  # Medium volatility
            target_multipliers = [1.02, 1.05, 1.08]
        else:  # Low volatility
            target_multipliers = [1.015, 1.03, 1.05]
            
        targets = {
            'target_1': round(current_price * target_multipliers[0], 2),
            'target_2': round(current_price * target_multipliers[1], 2),
            'target_3': round(current_price * target_multipliers[2], 2)
        }
        
        return targets
    
    def generate_trading_signal(self,
                               current_price: float,
                               predicted_price: float,
                               prediction_confidence: float,
                               technical_indicators: Dict,
                               historical_data: pd.DataFrame,
                               recommendation: str) -> Dict:
        """
        Generate comprehensive trading signal with entry/exit/targets
        This FIXES the contradiction issue by aligning with prediction
        """
        
        # Calculate predicted price change percentage
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate volatility from historical data
        returns = historical_data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Get support and resistance
        support, resistance = self.calculate_support_resistance(historical_data)
        
        # Determine action based on prediction AND technical confluence
        rsi = technical_indicators.get('RSI', 50)
        macd_signal = technical_indicators.get('MACD_Signal', 'NEUTRAL')
        
        # CORE FIX: Align action with prediction, not just RSI
        action = 'HOLD'
        confidence_level = 'MEDIUM'
        
        # Strong BUY conditions
        if price_change_pct > 5.0 and prediction_confidence > 0.85:
            action = 'STRONG BUY'
            confidence_level = 'HIGH'
        elif price_change_pct > 2.0 and prediction_confidence > 0.75:
            action = 'BUY'
            confidence_level = 'MEDIUM-HIGH'
        # Moderate BUY conditions
        elif price_change_pct > 0 and price_change_pct <= 2.0:
            if macd_signal == 'Bullish' or rsi < 40:
                action = 'BUY'
                confidence_level = 'MEDIUM'
            else:
                action = 'HOLD'
                confidence_level = 'LOW-MEDIUM'
        # SELL conditions
        elif price_change_pct < -5.0 and prediction_confidence > 0.85:
            action = 'STRONG SELL'
            confidence_level = 'HIGH'
        elif price_change_pct < -2.0 and prediction_confidence > 0.75:
            action = 'SELL'
            confidence_level = 'MEDIUM-HIGH'
        elif price_change_pct < 0 and price_change_pct >= -2.0:
            if macd_signal == 'Bearish' or rsi > 70:
                action = 'SELL'
                confidence_level = 'MEDIUM'
            else:
                action = 'HOLD'
                confidence_level = 'LOW-MEDIUM'
        else:
            action = 'HOLD'
            confidence_level = 'LOW'
        
        # Calculate precise entry/exit prices
        if action in ['BUY', 'STRONG BUY']:
            # Entry at slightly below current for better price
            entry_price = round(current_price * 0.995, 2)  # 0.5% below
            
            # Stop loss calculation
            stop_loss = round(entry_price * (1 - self.stop_loss_percentage), 2)
            
            # Calculate targets based on volatility
            targets = self.calculate_volatility_adjusted_targets(entry_price, volatility)
            
            # Calculate risk-reward
            risk = entry_price - stop_loss
            reward = targets['target_1'] - entry_price
            risk_reward = round(reward / risk, 2) if risk > 0 else 0
            
            # Position sizing based on risk
            max_risk_per_trade = 0.02  # 2% of capital
            position_size_multiplier = round(max_risk_per_trade / self.stop_loss_percentage, 2)
            
            signal = {
                'action': action,
                'confidence': confidence_level,
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'target_1': targets['target_1'],
                'target_2': targets['target_2'],
                'target_3': targets['target_3'],
                'predicted_price': round(predicted_price, 2),
                'potential_gain_pct': round(price_change_pct, 2),
                'risk_reward_ratio': risk_reward,
                'position_size_pct': position_size_multiplier * 100,
                'support_level': round(support, 2),
                'resistance_level': round(resistance, 2),
                'volatility': round(volatility * 100, 2),
                'holding_period': self._estimate_holding_period(price_change_pct, volatility),
                'reasoning': self._generate_reasoning(action, price_change_pct, rsi, macd_signal, confidence_level)
            }
            
        elif action in ['SELL', 'STRONG SELL']:
            # Exit at slightly above current for better price
            exit_price = round(current_price * 1.005, 2)  # 0.5% above
            
            # For short positions
            stop_loss = round(exit_price * (1 + self.stop_loss_percentage), 2)
            
            # Short targets (prices going down)
            targets = {
                'target_1': round(current_price * 0.97, 2),
                'target_2': round(current_price * 0.94, 2),
                'target_3': round(current_price * 0.90, 2)
            }
            
            signal = {
                'action': action,
                'confidence': confidence_level,
                'exit_price': exit_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'target_1': targets['target_1'],
                'target_2': targets['target_2'],
                'target_3': targets['target_3'],
                'predicted_price': round(predicted_price, 2),
                'potential_loss_pct': round(price_change_pct, 2),
                'support_level': round(support, 2),
                'resistance_level': round(resistance, 2),
                'volatility': round(volatility * 100, 2),
                'reasoning': self._generate_reasoning(action, price_change_pct, rsi, macd_signal, confidence_level)
            }
            
        else:  # HOLD
            signal = {
                'action': 'HOLD',
                'confidence': confidence_level,
                'current_price': current_price,
                'predicted_price': round(predicted_price, 2),
                'potential_change_pct': round(price_change_pct, 2),
                'support_level': round(support, 2),
                'resistance_level': round(resistance, 2),
                'reasoning': self._generate_reasoning('HOLD', price_change_pct, rsi, macd_signal, confidence_level),
                'watch_levels': {
                    'buy_below': round(support * 1.01, 2),
                    'sell_above': round(resistance * 0.99, 2)
                }
            }
        
        return signal
    
    def _estimate_holding_period(self, price_change_pct: float, volatility: float) -> str:
        """
        Estimate optimal holding period based on expected price movement
        """
        if abs(price_change_pct) > 10:
            return "7-14 days (Medium-term swing)"
        elif abs(price_change_pct) > 5:
            return "3-7 days (Short-term swing)"
        elif abs(price_change_pct) > 2:
            return "1-3 days (Intraday to short-term)"
        else:
            return "Intraday (Same day)"
    
    def _generate_reasoning(self, action: str, price_change_pct: float, 
                           rsi: float, macd_signal: str, confidence: str) -> str:
        """
        Generate human-readable reasoning for the trading signal
        """
        reasons = []
        
        if action in ['BUY', 'STRONG BUY']:
            reasons.append(f"Predicted price increase of {price_change_pct:.2f}%")
            
            if rsi < 40:
                reasons.append(f"RSI at {rsi:.2f} indicates oversold conditions")
            elif rsi < 50:
                reasons.append(f"RSI at {rsi:.2f} shows buying opportunity")
            
            if macd_signal == 'Bullish':
                reasons.append("MACD showing bullish momentum")
            
            if action == 'STRONG BUY':
                reasons.append("High confidence prediction with strong technical confirmation")
                
        elif action in ['SELL', 'STRONG SELL']:
            reasons.append(f"Predicted price decrease of {abs(price_change_pct):.2f}%")
            
            if rsi > 70:
                reasons.append(f"RSI at {rsi:.2f} indicates overbought conditions")
            elif rsi > 60:
                reasons.append(f"RSI at {rsi:.2f} shows selling opportunity")
            
            if macd_signal == 'Bearish':
                reasons.append("MACD showing bearish momentum")
            
            if action == 'STRONG SELL':
                reasons.append("High confidence prediction with strong technical confirmation")
                
        else:  # HOLD
            if abs(price_change_pct) < 2:
                reasons.append(f"Minimal predicted movement ({price_change_pct:.2f}%)")
            
            if 45 <= rsi <= 55:
                reasons.append("RSI neutral, no clear direction")
            
            if macd_signal == 'NEUTRAL':
                reasons.append("MACD shows no clear trend")
            
            reasons.append("Wait for clearer signals or better entry/exit levels")
        
        return ". ".join(reasons) + "."
    
    def batch_generate_signals(self, stocks_data: List[Dict]) -> List[Dict]:
        """
        Generate trading signals for multiple stocks
        Used by stock recommender to rank opportunities
        """
        signals = []
        
        for stock_data in stocks_data:
            try:
                signal = self.generate_trading_signal(
                    current_price=stock_data['current_price'],
                    predicted_price=stock_data['predicted_price'],
                    prediction_confidence=stock_data['confidence'],
                    technical_indicators=stock_data['technical_indicators'],
                    historical_data=stock_data['historical_data'],
                    recommendation=stock_data.get('recommendation', 'HOLD')
                )
                signal['symbol'] = stock_data['symbol']
                signal['name'] = stock_data.get('name', '')
                signals.append(signal)
            except Exception as e:
                print(f"Error generating signal for {stock_data.get('symbol', 'Unknown')}: {e}")
                continue
        
        return signals
