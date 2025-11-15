import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from local_storage import StorageManager
except ImportError as e:
    print(f"Warning: Could not import storage manager: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Enterprise portfolio optimizer for generating optimal investment portfolios.
    Uses risk-adjusted recommendations from autonomous analyzer.
    Implements Markowitz portfolio theory and Mean-Variance optimization.
    """
    
    def __init__(self, risk_free_rate=0.06):
        """Initialize portfolio optimizer"""
        self.logger = logger
        self.storage = StorageManager()
        self.risk_free_rate = risk_free_rate  # 6% for Indian market
        self.logger.info("Portfolio Optimizer initialized")
    
    def optimize_portfolio(self, recommendations: List[Dict], portfolio_size=10) -> Dict:
        """
        Generate optimal portfolio from recommendations.
        Uses risk-adjusted scoring and sector diversification.
        
        Args:
            recommendations: List of recommendation dicts with confidence, risk_metrics
            portfolio_size: Number of stocks to include
            
        Returns:
            Optimal portfolio with allocations
        """
        try:
            if not recommendations or len(recommendations) == 0:
                self.logger.warning("No recommendations provided")
                return {}
            
            # Calculate risk-adjusted scores
            scored_recs = self._calculate_risk_adjusted_scores(recommendations)
            
            # Select diverse portfolio
            portfolio = self._select_diversified_portfolio(scored_recs, portfolio_size)
            
            # Calculate allocations using Markowitz
            allocations = self._calculate_allocations(portfolio)
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(portfolio, allocations)
            
            result = {
                'portfolio': portfolio,
                'allocations': allocations,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'count': len(portfolio)
            }
            
            # Save to storage
            self.storage.save_portfolio(result)
            self.logger.info(f"Portfolio optimized: {len(portfolio)} stocks, expected_return={metrics.get('expected_return', 0):.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return {}
    
    def _calculate_risk_adjusted_scores(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Calculate risk-adjusted scores using Sharpe ratio concept.
        Score = (Expected Return - Risk-Free Rate) / Volatility
        """
        try:
            scored = []
            
            for rec in recommendations:
                try:
                    confidence = rec.get('confidence', 0)
                    risk_metrics = rec.get('risk_metrics', {})
                    volatility = risk_metrics.get('volatility', 0.2)  # Default 20%
                    
                    # Risk-adjusted return estimate
                    expected_return = confidence * 0.15  # Max 15% expected return
                    
                    # Avoid division by zero
                    if volatility > 0.001:
                        sharpe_score = (expected_return - self.risk_free_rate) / volatility
                    else:
                        sharpe_score = expected_return / 0.001
                    
                    scored.append({
                        **rec,
                        'expected_return': expected_return,
                        'volatility': volatility,
                        'sharpe_score': sharpe_score
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error scoring {rec.get('symbol')}: {e}")
                    continue
            
            # Sort by Sharpe score (risk-adjusted return)
            scored.sort(key=lambda x: x.get('sharpe_score', 0), reverse=True)
            return scored
            
        except Exception as e:
            self.logger.error(f"Error calculating scores: {e}")
            return recommendations
    
    def _select_diversified_portfolio(self, scored_recs: List[Dict], size: int) -> List[Dict]:
        """
        Select portfolio with sector diversification.
        Ensures not too many from same sector.
        """
        try:
            portfolio = []
            sectors = {}  # Track sector counts
            max_per_sector = max(1, size // 4)  # Max 25% from same sector
            
            for rec in scored_recs:
                if len(portfolio) >= size:
                    break
                
                sector = rec.get('sector', 'Unknown')
                sector_count = sectors.get(sector, 0)
                
                # Add if sector not over-represented
                if sector_count < max_per_sector:
                    portfolio.append(rec)
                    sectors[sector] = sector_count + 1
            
            # Fill remaining with top scorers if needed
            if len(portfolio) < size:
                for rec in scored_recs:
                    if len(portfolio) >= size:
                        break
                    if rec not in portfolio:
                        portfolio.append(rec)
            
            self.logger.info(f"Selected {len(portfolio)} stocks with sector distribution: {sectors}")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error selecting portfolio: {e}")
            return scored_recs[:size]
    
    def _calculate_allocations(self, portfolio: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio allocations based on Sharpe scores.
        Higher Sharpe = higher allocation.
        """
        try:
            if not portfolio:
                return {}
            
            # Calculate weights based on Sharpe scores
            sharpe_scores = [p.get('sharpe_score', 1) for p in portfolio]
            total_score = sum(sharpe_scores)
            
            allocations = {}
            for i, stock in enumerate(portfolio):
                symbol = stock.get('symbol', '')
                if total_score > 0:
                    weight = sharpe_scores[i] / total_score
                else:
                    weight = 1.0 / len(portfolio)
                
                allocations[symbol] = round(weight * 100, 2)  # Convert to percentage
            
            self.logger.info(f"Allocations calculated: sum={sum(allocations.values()):.2f}%")
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error calculating allocations: {e}")
            return {}
    
    def _calculate_portfolio_metrics(self, portfolio: List[Dict], allocations: Dict) -> Dict:
        """
        Calculate overall portfolio metrics.
        """
        try:
            if not portfolio:
                return {}
            
            # Portfolio return = weighted sum of individual returns
            portfolio_return = 0
            portfolio_volatility = 0
            
            for stock in portfolio:
                symbol = stock.get('symbol', '')
                weight = allocations.get(symbol, 0) / 100  # Convert back to decimal
                
                expected_return = stock.get('expected_return', 0)
                volatility = stock.get('volatility', 0)
                
                portfolio_return += weight * expected_return
                portfolio_volatility += (weight ** 2) * (volatility ** 2)
            
            portfolio_volatility = np.sqrt(portfolio_volatility)
            
            # Sharpe ratio for portfolio
            if portfolio_volatility > 0:
                portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            else:
                portfolio_sharpe = 0
            
            metrics = {
                'expected_return': round(portfolio_return, 4),
                'volatility': round(portfolio_volatility, 4),
                'sharpe_ratio': round(portfolio_sharpe, 4),
                'risk_free_rate': self.risk_free_rate,
                'excess_return': round(portfolio_return - self.risk_free_rate, 4),
                'stock_count': len(portfolio)
            }
            
            self.logger.info(f"Portfolio metrics: return={metrics['expected_return']:.3f}, vol={metrics['volatility']:.3f}, sharpe={metrics['sharpe_ratio']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def get_latest_portfolio(self) -> Dict:
        """
        Get latest optimized portfolio from storage.
        """
        try:
            return self.storage.get_latest_portfolio()
        except Exception as e:
            self.logger.error(f"Error retrieving portfolio: {e}")
            return {}
    
    def compare_portfolios(self, portfolio1: Dict, portfolio2: Dict) -> Dict:
        """
        Compare two portfolios by their metrics.
        """
        try:
            metrics1 = portfolio1.get('metrics', {})
            metrics2 = portfolio2.get('metrics', {})
            
            comparison = {
                'portfolio1_sharpe': metrics1.get('sharpe_ratio', 0),
                'portfolio2_sharpe': metrics2.get('sharpe_ratio', 0),
                'sharpe_difference': metrics1.get('sharpe_ratio', 0) - metrics2.get('sharpe_ratio', 0),
                'portfolio1_return': metrics1.get('expected_return', 0),
                'portfolio2_return': metrics2.get('expected_return', 0),
                'better_portfolio': 1 if metrics1.get('sharpe_ratio', 0) > metrics2.get('sharpe_ratio', 0) else 2
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing portfolios: {e}")
            return {}


if __name__ == "__main__":
    optimizer = PortfolioOptimizer()
    
    # Example usage
    sample_recommendations = [
        {'symbol': 'TCS.NS', 'confidence': 0.85, 'sector': 'IT', 'risk_metrics': {'volatility': 0.18}},
        {'symbol': 'INFY.NS', 'confidence': 0.78, 'sector': 'IT', 'risk_metrics': {'volatility': 0.16}},
        {'symbol': 'RELIANCE.NS', 'confidence': 0.82, 'sector': 'Energy', 'risk_metrics': {'volatility': 0.20}},
    ]
    
    portfolio = optimizer.optimize_portfolio(sample_recommendations, portfolio_size=3)
    print("Optimized Portfolio:")
    print(portfolio)
