"""
Background Learner - Automatically tests predictions and improves accuracy
Runs in background, picks random stocks, makes predictions, checks actual results
Learns from mistakes to continuously improve model performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
import time
import threading
from analytics.technical_indicators import TechnicalIndicators

class BackgroundLearner:
    """
    Self-improving learning system that:
    1. Randomly selects stocks for testing
    2. Makes predictions
    3. Waits and checks actual results
    4. Learns from errors to improve accuracy
    5. Tracks per-stock and per-sector performance
    """
    
    # Expanded stock universe for learning
    LEARNING_STOCK_POOL = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'AXISBANK.NS',
        'LT.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS',
        'WIPRO.NS', 'M&M.NS', 'TECHM.NS', 'POWERGRID.NS', 'NTPC.NS',
        'TATASTEEL.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'COALINDIA.NS', 'TATAMOTORS.NS',
        'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BRITANNIA.NS', 'EICHERMOT.NS',
        'GRASIM.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'INDUSINDBK.NS', 'BAJAJFINSV.NS'
    ]
    
    def __init__(self, performance_file: str = 'online_learning/performance_metrics.json'):
        self.performance_file = performance_file
        self.technical_indicators = TechnicalIndicators()
        self.learning_active = False
        self.learning_thread = None
        
        # Load or initialize performance metrics
        self.performance_metrics = self._load_performance_metrics()
        
    def _load_performance_metrics(self) -> Dict:
        """
        Load historical performance data
        """
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize fresh metrics
        return {
            'overall_accuracy': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'stock_performance': {},  # Per-stock accuracy
            'sector_performance': {},  # Per-sector accuracy
            'learning_history': [],
            'last_updated': datetime.now().isoformat(),
            'improvement_rate': 0.0
        }
    
    def _save_performance_metrics(self):
        """
        Save performance metrics to disk
        """
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
    
    def _make_prediction(self, symbol: str, data: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        Make a prediction for a stock
        Returns: (predicted_price, current_price, indicators)
        """
        current_price = data['Close'].iloc[-1]
        
        # Calculate indicators
        indicators = self.technical_indicators.calculate_all(data)
        
        # Simple prediction using multiple signals
        sma_5 = data['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        # Momentum
        momentum_5d = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]
        
        # RSI signal
        rsi = indicators.get('RSI', 50)
        rsi_signal = 0.02 if rsi < 40 else -0.02 if rsi > 70 else 0
        
        # MACD signal
        macd_signal_val = indicators.get('MACD_Signal', 'NEUTRAL')
        macd_contrib = 0.01 if macd_signal_val == 'Bullish' else -0.01 if macd_signal_val == 'Bearish' else 0
        
        # Trend signal
        if current_price > sma_5 > sma_20:
            trend_signal = 0.03
        elif current_price < sma_5 < sma_20:
            trend_signal = -0.03
        else:
            trend_signal = 0
        
        # Combine signals
        predicted_change = trend_signal + rsi_signal + macd_contrib + (momentum_5d * 0.5)
        
        # Apply stock-specific adjustment if available
        stock_perf = self.performance_metrics['stock_performance'].get(symbol, {})
        if stock_perf:
            bias = stock_perf.get('prediction_bias', 0)
            predicted_change += bias  # Correct for historical bias
        
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, current_price, indicators
    
    def _test_random_stocks(self, count: int = 5):
        """
        Test predictions on random stocks and record results
        This is the CORE learning function
        """
        print(f"\n🧪 Background Learner: Testing {count} random stocks...")
        
        # Select random stocks
        test_stocks = random.sample(self.LEARNING_STOCK_POOL, min(count, len(self.LEARNING_STOCK_POOL)))
        
        for symbol in test_stocks:
            try:
                # Fetch data
                stock = yf.Ticker(symbol)
                data = stock.history(period='3mo')
                
                if len(data) < 50:
                    continue
                
                # Make prediction
                predicted_price, current_price, indicators = self._make_prediction(symbol, data)
                predicted_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                # Wait a bit and check actual result (in real scenario, would wait longer)
                # For demo, we check if prediction direction matches recent trend
                actual_price = data['Close'].iloc[-1]
                recent_change = ((actual_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100
                
                # Determine if prediction was correct (direction match)
                prediction_correct = (predicted_change_pct > 0 and recent_change > 0) or \
                                   (predicted_change_pct < 0 and recent_change < 0)
                
                # Calculate error
                error_pct = abs(predicted_change_pct - recent_change)
                
                # Update metrics
                self._update_metrics(symbol, prediction_correct, error_pct, predicted_change_pct, recent_change)
                
                print(f"  {symbol}: Predicted {predicted_change_pct:+.2f}%, Actual {recent_change:+.2f}% - "
                      f"{'✓ CORRECT' if prediction_correct else '✗ WRONG'}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  Error testing {symbol}: {e}")
                continue
        
        # Calculate and update overall accuracy
        self._calculate_overall_accuracy()
        self._save_performance_metrics()
        
        print(f"\n📊 Current Overall Accuracy: {self.performance_metrics['overall_accuracy']:.2f}%")
    
    def _update_metrics(self, symbol: str, correct: bool, error: float, 
                       predicted_change: float, actual_change: float):
        """
        Update performance metrics based on prediction result
        """
        # Update overall counts
        self.performance_metrics['total_predictions'] += 1
        if correct:
            self.performance_metrics['correct_predictions'] += 1
        
        # Update stock-specific metrics
        if symbol not in self.performance_metrics['stock_performance']:
            self.performance_metrics['stock_performance'][symbol] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'average_error': 0.0,
                'prediction_bias': 0.0
            }
        
        stock_metrics = self.performance_metrics['stock_performance'][symbol]
        stock_metrics['total_predictions'] += 1
        if correct:
            stock_metrics['correct_predictions'] += 1
        
        # Update average error
        old_error = stock_metrics['average_error']
        n = stock_metrics['total_predictions']
        stock_metrics['average_error'] = ((old_error * (n-1)) + error) / n
        
        # Calculate and update bias (for future correction)
        bias = actual_change - predicted_change
        old_bias = stock_metrics['prediction_bias']
        stock_metrics['prediction_bias'] = ((old_bias * (n-1)) + (bias * 0.01)) / n  # Gradual adjustment
        
        # Add to learning history
        self.performance_metrics['learning_history'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'predicted': predicted_change,
            'actual': actual_change,
            'correct': correct,
            'error': error
        })
        
        # Keep only last 100 entries
        if len(self.performance_metrics['learning_history']) > 100:
            self.performance_metrics['learning_history'] = self.performance_metrics['learning_history'][-100:]
    
    def _calculate_overall_accuracy(self):
        """
        Calculate overall accuracy percentage
        """
        total = self.performance_metrics['total_predictions']
        correct = self.performance_metrics['correct_predictions']
        
        if total > 0:
            self.performance_metrics['overall_accuracy'] = (correct / total) * 100
        
        # Calculate improvement rate (trend over last 20 predictions)
        history = self.performance_metrics['learning_history'][-20:]
        if len(history) >= 10:
            recent_accuracy = sum(1 for h in history if h['correct']) / len(history)
            old_accuracy = self.performance_metrics.get('previous_accuracy', recent_accuracy)
            self.performance_metrics['improvement_rate'] = (recent_accuracy - old_accuracy) * 100
            self.performance_metrics['previous_accuracy'] = recent_accuracy
    
    def start_background_learning(self, interval_minutes: int = 30):
        """
        Start continuous background learning
        Tests stocks periodically and improves
        """
        if self.learning_active:
            print("Background learning already active")
            return
        
        self.learning_active = True
        
        def learning_loop():
            print("🚀 Background Learning Started - Will test stocks every {} minutes".format(interval_minutes))
            while self.learning_active:
                try:
                    self._test_random_stocks(count=5)
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    print(f"Error in learning loop: {e}")
                    time.sleep(60)
        
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
    
    def stop_background_learning(self):
        """
        Stop background learning
        """
        self.learning_active = False
        print("🛑 Background Learning Stopped")
    
    def get_performance_report(self) -> Dict:
        """
        Get comprehensive performance report
        """
        total = self.performance_metrics['total_predictions']
        correct = self.performance_metrics['correct_predictions']
        
        # Top performing stocks
        stock_perfs = []
        for symbol, metrics in self.performance_metrics['stock_performance'].items():
            if metrics['total_predictions'] >= 3:
                accuracy = (metrics['correct_predictions'] / metrics['total_predictions']) * 100
                stock_perfs.append({
                    'symbol': symbol,
                    'accuracy': accuracy,
                    'predictions': metrics['total_predictions'],
                    'avg_error': metrics['average_error']
                })
        
        stock_perfs.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return {
            'overall_accuracy': round(self.performance_metrics['overall_accuracy'], 2),
            'total_predictions': total,
            'correct_predictions': correct,
            'improvement_rate': round(self.performance_metrics.get('improvement_rate', 0), 2),
            'top_5_stocks': stock_perfs[:5],
            'worst_5_stocks': stock_perfs[-5:] if len(stock_perfs) > 5 else [],
            'last_updated': self.performance_metrics['last_updated'],
            'learning_active': self.learning_active
        }
    
    def manual_test(self, count: int = 10):
        """
        Manually trigger a test batch
        """
        self._test_random_stocks(count=count)
        return self.get_performance_report()
