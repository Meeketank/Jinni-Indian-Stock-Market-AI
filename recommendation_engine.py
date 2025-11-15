# recommendation_engine.py
"""
Enhanced Recommendation Engine with Explainability
Always provides recommendations with confidence levels and reasons
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import math

class RecommendationEngine:
    def __init__(self, min_confidence_threshold: float = 25.0):
        self.min_confidence_threshold = min_confidence_threshold
        self.explainability_enabled = True
    
    def score_recommendation(self, 
                           predicted_return_pct: float,
                           confidence: float,
                           volatility: float,
                           technical_signals: Dict[str, Any],
                           fundamental_data: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
        """
        Multi-factor scoring with explainability
        Returns: (score, reasons_list)
        """
        score = 0.0
        reasons = []
        
        # Factor 1: Magnitude of predicted return (40% weight)
        return_score = abs(predicted_return_pct) / 100.0  # Normalize
        score += return_score * 0.4
        if abs(predicted_return_pct) >= 10:
            reasons.append(f"Strong predicted move: {predicted_return_pct:+.2f}%")
        elif abs(predicted_return_pct) >= 5:
            reasons.append(f"Moderate predicted move: {predicted_return_pct:+.2f}%")
        else:
            reasons.append(f"Small predicted move: {predicted_return_pct:+.2f}%")
        
        # Factor 2: Model confidence (30% weight)
        conf_score = confidence / 100.0
        score += conf_score * 0.3
        if confidence >= 70:
            reasons.append(f"High model confidence: {confidence:.1f}%")
        elif confidence >= 40:
            reasons.append(f"Moderate confidence: {confidence:.1f}%")
        else:
            reasons.append(f"Low confidence: {confidence:.1f}% - treat with caution")
        
        # Factor 3: Risk-adjusted return (15% weight)
        if volatility > 0:
            sharpe_like = abs(predicted_return_pct) / (volatility * 100)
            risk_adj_score = min(1.0, sharpe_like / 2.0)  # Cap at 1.0
            score += risk_adj_score * 0.15
            if sharpe_like >= 1.5:
                reasons.append(f"Excellent risk-return profile (vol: {volatility*100:.1f}%)")
            elif sharpe_like >= 0.8:
                reasons.append(f"Good risk-return profile (vol: {volatility*100:.1f}%)")
            else:
                reasons.append(f"High volatility risk (vol: {volatility*100:.1f}%)")
        
        # Factor 4: Technical indicators (10% weight)
        tech_score = 0.0
        tech_reasons = []
        
        if technical_signals:
            rsi = technical_signals.get('RSI')
            macd = technical_signals.get('MACD')
            macd_signal = technical_signals.get('MACD_Signal')
            ma20 = technical_signals.get('MA20')
            ma50 = technical_signals.get('MA50')
            current_price = technical_signals.get('current_price')
            
            # RSI signals
            if rsi:
                if rsi > 70:
                    tech_reasons.append("Overbought (RSI > 70)")
                    tech_score += 0.3 if predicted_return_pct < 0 else -0.3
                elif rsi < 30:
                    tech_reasons.append("Oversold (RSI < 30)")
                    tech_score += 0.3 if predicted_return_pct > 0 else -0.3
                else:
                    tech_reasons.append(f"RSI neutral ({rsi:.1f})")
            
            # MACD signals
            if macd and macd_signal:
                if macd > macd_signal:
                    tech_reasons.append("MACD bullish crossover")
                    tech_score += 0.3 if predicted_return_pct > 0 else 0
                else:
                    tech_reasons.append("MACD bearish")
                    tech_score += 0.3 if predicted_return_pct < 0 else 0
            
            # MA trend
            if ma20 and ma50 and current_price:
                if ma20 > ma50 and current_price > ma20:
                    tech_reasons.append("Strong uptrend (price > MA20 > MA50)")
                    tech_score += 0.4
                elif ma20 < ma50 and current_price < ma20:
                    tech_reasons.append("Strong downtrend (price < MA20 < MA50)")
                    tech_score += 0.4 if predicted_return_pct < 0 else 0
        
        tech_score = max(0, min(1.0, tech_score))  # Clamp to [0,1]
        score += tech_score * 0.1
        reasons.extend(tech_reasons)
        
        # Factor 5: Fundamentals (5% weight)
        if fundamental_data:
            fund_score = 0.0
            fund_reasons = []
            
            pe = fundamental_data.get('trailingPE') or fundamental_data.get('forwardPE')
            pb = fundamental_data.get('priceToBook')
            
            if pe:
                if pe < 15:
                    fund_reasons.append(f"Undervalued P/E: {pe:.1f}")
                    fund_score += 0.5
                elif pe > 40:
                    fund_reasons.append(f"Overvalued P/E: {pe:.1f}")
                    fund_score -= 0.3
                else:
                    fund_reasons.append(f"P/E: {pe:.1f}")
            
            if pb:
                if pb < 2:
                    fund_reasons.append(f"Low P/B: {pb:.2f}")
                    fund_score += 0.3
            
            fund_score = max(0, min(1.0, fund_score))
            score += fund_score * 0.05
            reasons.extend(fund_reasons)
        
        # Normalize final score to 0-100 range
        final_score = score * 100
        
        return final_score, reasons
    
    def generate_recommendations(self,
                                scan_results: List[Dict],
                                min_expected_pct: float,
                                always_show_top_n: int = 10) -> Dict[str, Any]:
        """
        Generate recommendations with explainability.
        ALWAYS returns some results (best available) even if below threshold.
        """
        if not scan_results:
            return {
                'status': 'no_data',
                'message': 'No stocks were scanned. Please run a scan first.',
                'recommendations': [],
                'near_misses': []
            }
        
        # Score all results
        scored_results = []
        for result in scan_results:
            est_pct = result.get('est_pct', 0)
            confidence = result.get('confidence', 0)
            volatility = result.get('volatility', 0.2)
            
            technical_signals = {
                'RSI': result.get('RSI'),
                'MACD': result.get('MACD'),
                'MACD_Signal': result.get('MACD_Signal'),
                'MA20': result.get('MA20'),
                'MA50': result.get('MA50'),
                'current_price': result.get('last_close')
            }
            
            fundamental_data = result.get('fundamentals')
            
            score, reasons = self.score_recommendation(
                est_pct, confidence, volatility, 
                technical_signals, fundamental_data
            )
            
            result_copy = result.copy()
            result_copy['overall_score'] = score
            result_copy['reasons'] = reasons
            result_copy['meets_threshold'] = abs(est_pct) >= min_expected_pct
            
            scored_results.append(result_copy)
        
        # Sort by overall score
        scored_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Separate into categories
        high_confidence = [r for r in scored_results if r['meets_threshold'] and r['confidence'] >= 50]
        above_threshold = [r for r in scored_results if r['meets_threshold']]
        near_misses = [r for r in scored_results if not r['meets_threshold']][:always_show_top_n]
        
        # Response structure
        response = {
            'status': 'success',
            'total_scanned': len(scan_results),
            'min_threshold': min_expected_pct,
            'high_confidence_recommendations': high_confidence[:always_show_top_n],
            'all_above_threshold': above_threshold[:50],  # Cap at 50
            'near_misses': near_misses,
            'best_overall': scored_results[:always_show_top_n]
        }
        
        # Generate summary message
        if high_confidence:
            response['message'] = f"Found {len(high_confidence)} high-confidence opportunities above {min_expected_pct}% threshold."
        elif above_threshold:
            response['message'] = f"Found {len(above_threshold)} opportunities above {min_expected_pct}% threshold (mixed confidence)."
        elif near_misses:
            max_move = max([abs(r['est_pct']) for r in near_misses])
            response['message'] = f"No opportunities above {min_expected_pct}% threshold. Showing best {len(near_misses)} candidates (max predicted: {max_move:.2f}%)."
            response['suggestion'] = f"Consider lowering threshold to {max_move*0.8:.1f}% or improving model accuracy."
        else:
            response['message'] = "No viable opportunities found in current scan."
        
        return response
    
    def explain_recommendation(self, recommendation: Dict) -> str:
        """
        Generate human-readable explanation for a recommendation
        """
        symbol = recommendation.get('symbol', 'Unknown')
        est_pct = recommendation.get('est_pct', 0)
        confidence = recommendation.get('confidence', 0)
        overall_score = recommendation.get('overall_score', 0)
        reasons = recommendation.get('reasons', [])
        
        direction = "upward" if est_pct > 0 else "downward"
        strength = "strong" if abs(est_pct) >= 10 else ("moderate" if abs(est_pct) >= 5 else "weak")
        
        explanation = f"{symbol}: {strength.capitalize()} {direction} movement expected ({est_pct:+.2f}%).\n"
        explanation += f"Overall Score: {overall_score:.1f}/100 | Model Confidence: {confidence:.1f}%\n"
        explanation += "\nKey Factors:\n"
        
        for i, reason in enumerate(reasons, 1):
            explanation += f"  {i}. {reason}\n"
        
        return explanation
    
    def generate_comparison_table(self, recommendations: List[Dict]) -> pd.DataFrame:
        """
        Create a comparison DataFrame for multiple recommendations
        """
        if not recommendations:
            return pd.DataFrame()
        
        data = []
        for rec in recommendations:
            data.append({
                'Symbol': rec.get('symbol'),
                'Predicted Move %': f"{rec.get('est_pct', 0):+.2f}",
                'Confidence %': f"{rec.get('confidence', 0):.1f}",
                'Overall Score': f"{rec.get('overall_score', 0):.1f}",
                'Last Price': f"₹{rec.get('last_close', 0):.2f}",
                'Top Reason': rec.get('reasons', ['N/A'])[0] if rec.get('reasons') else 'N/A'
            })
        
        return pd.DataFrame(data)
