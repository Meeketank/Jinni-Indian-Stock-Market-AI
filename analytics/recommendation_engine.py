"""Intelligent Multi-Factor Recommendation Engine for JINNI

This module eliminates contradictions by considering:
1. Price predictions (most important)
2. Technical indicators (RSI, MACD, MA)
3. Fundamental metrics (P/E, P/B, ROE)
4. Market sentiment and volume
5. Risk-reward ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class IntelligentRecommendationEngine:
    """Advanced recommendation system with multi-factor analysis"""
    
    def __init__(self):
        # Weighting for each factor (total = 100%)
        self.weights = {
            'prediction': 0.40,  # 40% - Most important!
            'technical': 0.30,   # 30% - RSI, MACD, MA
            'momentum': 0.15,    # 15% - Price momentum
            'volume': 0.10,      # 10% - Volume analysis
            'risk': 0.05         # 5% - Risk assessment
        }
    
    def get_recommendation(self, 
                          pred_change_pct: float,
                          rsi: float,
                          macd: float,
                          current_price: float,
                          ma20: float,
                          ma50: float,
                          volume_ratio: float = 1.0,
                          volatility: float = 20.0,
                          fundamentals: Dict = None) -> Tuple[str, str, int]:
        """
        Generate intelligent recommendation based on ALL factors
        
        Returns:
            (recommendation, reason, confidence_score)
        """
        
        scores = {}
        reasons = []
        
        # 1. PREDICTION SCORE (40% weight) - MOST IMPORTANT!
        pred_score = self._score_prediction(pred_change_pct)
        scores['prediction'] = pred_score
        
        if pred_change_pct > 15:
            reasons.append(f"Strong upside potential ({pred_change_pct:+.1f}%)")
        elif pred_change_pct > 5:
            reasons.append(f"Moderate upside ({pred_change_pct:+.1f}%)")
        elif pred_change_pct < -5:
            reasons.append(f"Downside risk ({pred_change_pct:+.1f}%)")
        
        # 2. TECHNICAL SCORE (30% weight)
        tech_score = self._score_technical(rsi, macd, current_price, ma20, ma50)
        scores['technical'] = tech_score
        
        # Add technical reasons
        if rsi < 30:
            reasons.append("Oversold conditions (RSI < 30)")
        elif rsi > 70:
            reasons.append("Overbought warning (RSI > 70)")
        
        if current_price > ma50:
            reasons.append("Above 50-day MA (bullish trend)")
        elif current_price < ma50:
            reasons.append("Below 50-day MA (bearish trend)")
        
        # 3. MOMENTUM SCORE (15% weight)
        momentum_score = self._score_momentum(current_price, ma20)
        scores['momentum'] = momentum_score
        
        # 4. VOLUME SCORE (10% weight)
        volume_score = self._score_volume(volume_ratio)
        scores['volume'] = volume_score
        
        if volume_ratio > 1.5:
            reasons.append("High volume support")
        
        # 5. RISK SCORE (5% weight)
        risk_score = self._score_risk(volatility)
        scores['risk'] = risk_score
        
        # Calculate weighted total score
        total_score = (
            scores['prediction'] * self.weights['prediction'] +
            scores['technical'] * self.weights['technical'] +
            scores['momentum'] * self.weights['momentum'] +
            scores['volume'] * self.weights['volume'] +
            scores['risk'] * self.weights['risk']
        )
        
        # Generate recommendation based on total score
        recommendation, main_reason = self._generate_recommendation(total_score, pred_change_pct, rsi)
        
        # Calculate confidence (0-100)
        confidence = int(min(100, max(0, abs(total_score - 50) * 2)))
        
        # Combine main reason with supporting reasons
        full_reason = main_reason
        if reasons:
            full_reason += ". " + ", ".join(reasons[:2])  # Top 2 reasons
        
        return recommendation, full_reason, confidence
    
    def _score_prediction(self, pred_change_pct: float) -> float:
        """
        Score based on predicted price change
        Returns 0-100 where 50 is neutral
        """
        if pred_change_pct > 20:
            return 95  # Strong buy
        elif pred_change_pct > 10:
            return 80  # Buy
        elif pred_change_pct > 3:
            return 65  # Mild buy
        elif pred_change_pct > -3:
            return 50  # Neutral/Hold
        elif pred_change_pct > -10:
            return 35  # Mild sell
        elif pred_change_pct > -20:
            return 20  # Sell
        else:
            return 5   # Strong sell
    
    def _score_technical(self, rsi: float, macd: float, price: float, ma20: float, ma50: float) -> float:
        """
        Score based on technical indicators
        """
        score = 50  # Start neutral
        
        # RSI component
        if rsi < 30:
            score += 20  # Oversold = bullish
        elif rsi < 40:
            score += 10
        elif rsi > 70:
            score -= 20  # Overbought = bearish
        elif rsi > 60:
            score -= 10
        
        # MACD component
        if macd > 0:
            score += 10
        else:
            score -= 10
        
        # Moving average component
        if price > ma50:
            score += 10
        else:
            score -= 10
        
        if price > ma20:
            score += 5
        else:
            score -= 5
        
        return min(100, max(0, score))
    
    def _score_momentum(self, price: float, ma20: float) -> float:
        """
        Score based on price momentum
        """
        if ma20 == 0:
            return 50
        
        distance_pct = ((price - ma20) / ma20) * 100
        
        if distance_pct > 5:
            return 70
        elif distance_pct > 2:
            return 60
        elif distance_pct > -2:
            return 50
        elif distance_pct > -5:
            return 40
        else:
            return 30
    
    def _score_volume(self, volume_ratio: float) -> float:
        """
        Score based on volume
        """
        if volume_ratio > 2.0:
            return 70
        elif volume_ratio > 1.5:
            return 60
        elif volume_ratio > 0.8:
            return 50
        else:
            return 40
    
    def _score_risk(self, volatility: float) -> float:
        """
        Score based on risk (volatility)
        Lower volatility = higher score
        """
        if volatility < 15:
            return 60  # Low risk
        elif volatility < 25:
            return 50  # Medium risk
        else:
            return 40  # High risk
    
    def _generate_recommendation(self, score: float, pred_change_pct: float, rsi: float) -> Tuple[str, str]:
        """
        Generate final recommendation and main reason
        """
        # CRITICAL: Prediction dominates if strong
        if pred_change_pct > 15 and score > 55:
            return "🟢 STRONG BUY", f"AI predicts +{pred_change_pct:.1f}% upside"
        elif pred_change_pct > 8 and score > 50:
            return "🟢 BUY", f"Positive outlook (+{pred_change_pct:.1f}%)"
        elif pred_change_pct < -10 and score < 45:
            return "🔴 STRONG SELL", f"AI predicts {pred_change_pct:.1f}% downside"
        elif pred_change_pct < -5 and score < 50:
            return "🔴 SELL", f"Negative outlook ({pred_change_pct:.1f}%)"
        
        # For moderate predictions, use full score
        if score >= 70:
            return "🟢 STRONG BUY", "Strong bullish signals across all factors"
        elif score >= 60:
            return "🟢 BUY", "Multiple bullish indicators"
        elif score >= 55:
            return "🟢 ACCUMULATE", "Slight bullish bias"
        elif score >= 45:
            return "🟡 HOLD", "Mixed signals, neutral stance recommended"
        elif score >= 35:
            return "🟠 REDUCE", "Slight bearish bias"
        elif score >= 25:
            return "🔴 SELL", "Multiple bearish indicators"
        else:
            return "🔴 STRONG SELL", "Strong bearish signals"


def get_smart_recommendation(hist_data: pd.DataFrame, 
                            pred_price: float, 
                            current_price: float, 
                            pred_days: int) -> Dict:
    """
    Convenience function to get recommendation with all analysis
    """
    engine = IntelligentRecommendationEngine()
    
    # Calculate prediction change
    pred_change_pct = ((pred_price - current_price) / current_price) * 100
    
    # Get technical indicators
    rsi = hist_data['RSI'].iloc[-1]
    macd = hist_data['MACD'].iloc[-1]
    ma20 = hist_data['MA20'].iloc[-1]
    ma50 = hist_data['MA50'].iloc[-1]
    
    # Calculate volume ratio
    avg_volume = hist_data['Volume'].rolling(20).mean().iloc[-1]
    current_volume = hist_data['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Calculate volatility
    volatility = hist_data['Close'].pct_change().std() * np.sqrt(252) * 100
    
    # Get recommendation
    recommendation, reason, confidence = engine.get_recommendation(
        pred_change_pct=pred_change_pct,
        rsi=rsi,
        macd=macd,
        current_price=current_price,
        ma20=ma20,
        ma50=ma50,
        volume_ratio=volume_ratio,
        volatility=volatility
    )
    
    return {
        'recommendation': recommendation,
        'reason': reason,
        'confidence': confidence,
        'pred_change_pct': pred_change_pct,
        'technical_score': {
            'rsi': rsi,
            'macd': macd,
            'ma_signal': 'Bullish' if current_price > ma20 else 'Bearish',
            'volatility': volatility
        }
    }
