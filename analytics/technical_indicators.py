import numpy as np
import pandas as pd
from typing import Dict, List

class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis"""
    
    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)"""
        if len(data) < period + 1:
            return 50.0
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(data) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_fast = TechnicalIndicators.calculate_ema(data, fast)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_values = []
        for i in range(slow, len(data)):
            fast_ema = TechnicalIndicators.calculate_ema(data[:i+1], fast)
            slow_ema = TechnicalIndicators.calculate_ema(data[:i+1], slow)
            macd_values.append(fast_ema - slow_ema)
        
        if len(macd_values) >= signal:
            signal_line = TechnicalIndicators.calculate_ema(np.array(macd_values), signal)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_ema(data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average (EMA)"""
        if len(data) < period:
            return np.mean(data)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def calculate_sma(data: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average (SMA)"""
        if len(data) < period:
            return np.mean(data)
        return np.mean(data[-period:])
    
    @staticmethod
    def calculate_bollinger_bands(data: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(data) < period:
            avg = np.mean(data)
            return {'upper': avg, 'middle': avg, 'lower': avg}
        
        sma = TechnicalIndicators.calculate_sma(data, period)
        std = np.std(data[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': (upper_band - lower_band) / sma * 100
        }
    
    @staticmethod
    def calculate_stochastic(data_high: np.ndarray, data_low: np.ndarray, 
                            data_close: np.ndarray, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        if len(data_close) < period:
            return {'k': 50.0, 'd': 50.0}
        
        lowest_low = np.min(data_low[-period:])
        highest_high = np.max(data_high[-period:])
        
        if highest_high == lowest_low:
            k_value = 50.0
        else:
            k_value = ((data_close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (3-period SMA of %K)
        k_values = []
        for i in range(max(0, len(data_close) - period), len(data_close)):
            ll = np.min(data_low[max(0, i-period+1):i+1])
            hh = np.max(data_high[max(0, i-period+1):i+1])
            if hh != ll:
                k_values.append(((data_close[i] - ll) / (hh - ll)) * 100)
        
        d_value = np.mean(k_values[-3:]) if len(k_values) >= 3 else k_value
        
        return {'k': k_value, 'd': d_value}
    
    @staticmethod
    def calculate_atr(data_high: np.ndarray, data_low: np.ndarray, 
                     data_close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        if len(data_close) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(data_close)):
            high_low = data_high[i] - data_low[i]
            high_close = abs(data_high[i] - data_close[i-1])
            low_close = abs(data_low[i] - data_close[i-1])
            true_ranges.append(max(high_low, high_close, low_close))
        
        if len(true_ranges) < period:
            return np.mean(true_ranges)
        
        return np.mean(true_ranges[-period:])
    
    @staticmethod
    def calculate_adx(data_high: np.ndarray, data_low: np.ndarray, 
                     data_close: np.ndarray, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        if len(data_close) < period + 1:
            return 0.0
        
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(data_high)):
            high_diff = data_high[i] - data_high[i-1]
            low_diff = data_low[i-1] - data_low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)
            
            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)
        
        # Calculate ATR
        atr = TechnicalIndicators.calculate_atr(data_high, data_low, data_close, period)
        
        if atr == 0:
            return 0.0
        
        # Calculate +DI and -DI
        plus_di = (np.mean(plus_dm[-period:]) / atr) * 100
        minus_di = (np.mean(minus_dm[-period:]) / atr) * 100
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        
        return dx
    
    @staticmethod
    def calculate_obv(data_close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate On-Balance Volume (OBV)"""
        if len(data_close) < 2 or len(volume) < 2:
            return 0.0
        
        obv = 0
        for i in range(1, len(data_close)):
            if data_close[i] > data_close[i-1]:
                obv += volume[i]
            elif data_close[i] < data_close[i-1]:
                obv -= volume[i]
        
        return obv
    
    @staticmethod
    def get_all_indicators(data_close: np.ndarray, data_high: np.ndarray = None,
                          data_low: np.ndarray = None, volume: np.ndarray = None) -> Dict:
        """Calculate all technical indicators at once"""
        indicators = {}
        
        # Price-based indicators
        indicators['rsi'] = TechnicalIndicators.calculate_rsi(data_close)
        indicators['macd'] = TechnicalIndicators.calculate_macd(data_close)
        indicators['ema_12'] = TechnicalIndicators.calculate_ema(data_close, 12)
        indicators['ema_26'] = TechnicalIndicators.calculate_ema(data_close, 26)
        indicators['sma_20'] = TechnicalIndicators.calculate_sma(data_close, 20)
        indicators['sma_50'] = TechnicalIndicators.calculate_sma(data_close, 50)
        indicators['bollinger'] = TechnicalIndicators.calculate_bollinger_bands(data_close)
        
        # Indicators requiring high/low data
        if data_high is not None and data_low is not None:
            indicators['stochastic'] = TechnicalIndicators.calculate_stochastic(
                data_high, data_low, data_close
            )
            indicators['atr'] = TechnicalIndicators.calculate_atr(
                data_high, data_low, data_close
            )
            indicators['adx'] = TechnicalIndicators.calculate_adx(
                data_high, data_low, data_close
            )
        
        # Volume-based indicators
        if volume is not None:
            indicators['obv'] = TechnicalIndicators.calculate_obv(data_close, volume)
        
        return indicators
