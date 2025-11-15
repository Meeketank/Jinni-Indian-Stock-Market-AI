"""
Risk Analytics & Portfolio Optimization - BlackRock Aladdin-Competitive System

This module implements institutional-grade risk management with:
- Modern Portfolio Theory (MPT) optimization
- Value at Risk (VaR) calculations
- Sharpe ratio and risk-adjusted returns
- Correlation matrix analysis
- Efficient frontier visualization
- Monte Carlo simulations
- Sector exposure analysis
- Drawdown analysis

Author: JINNI Development Team
Designed to compete with BlackRock's Aladdin risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Portfolio optimization
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
from scipy.optimize import minimize
import cvxpy as cp

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskAnalytics:
    """
    Comprehensive risk management and portfolio optimization.
    
    Features:
    - Portfolio optimization using Modern Portfolio Theory
    - Value at Risk (VaR) analysis
    - Correlation and covariance matrix
    - Efficient frontier calculation
    - Monte Carlo risk simulation
    - Sharpe ratio optimization
    """
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize risk analytics with price data.
        
        Args:
            price_data: DataFrame with stock prices (columns=tickers, index=dates)
        """
        self.price_data = price_data
        self.returns = price_data.pct_change().dropna()
        self.tickers = price_data.columns.tolist()
        self.n_assets = len(self.tickers)
        
        # Calculate key statistics
        self.mean_returns = expected_returns.mean_historical_return(price_data)
        self.cov_matrix = risk_models.sample_cov(price_data)
        
        logger.info(f"Initialized RiskAnalytics for {self.n_assets} assets")
    
    def calculate_var(self, portfolio_value: float, confidence_level: float = 0.95, 
                     time_horizon: int = 1) -> Dict:
        """
        Calculate Value at Risk (VaR) for portfolio.
        
        Args:
            portfolio_value: Total portfolio value
            confidence_level: Confidence level (default 95%)
            time_horizon: Time horizon in days
        
        Returns:
            Dictionary with VaR metrics
        """
        # Historical VaR
        portfolio_returns = self.returns.mean(axis=1)  # Equal-weighted for now
        var_percentile = (1 - confidence_level) * 100
        historical_var = np.percentile(portfolio_returns, var_percentile)
        var_amount = portfolio_value * abs(historical_var) * np.sqrt(time_horizon)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        z_score = abs(np.percentile(np.random.normal(0, 1, 10000), var_percentile))
        parametric_var = portfolio_value * (mean_return - z_score * std_return) * np.sqrt(time_horizon)
        
        # Conditional VaR (CVaR or Expected Shortfall)
        cvar = portfolio_returns[portfolio_returns <= historical_var].mean()
        cvar_amount = portfolio_value * abs(cvar) * np.sqrt(time_horizon)
        
        logger.info(f"VaR calculated: Historical={var_amount:.2f}, Parametric={abs(parametric_var):.2f}")
        
        return {
            'historical_var': var_amount,
            'parametric_var': abs(parametric_var),
            'conditional_var': cvar_amount,
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon
        }
    
    def optimize_portfolio(self, target_return: Optional[float] = None, 
                          risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize portfolio using Modern Portfolio Theory.
        
        Args:
            target_return: Target return (if None, optimizes for max Sharpe ratio)
            risk_free_rate: Risk-free rate for Sharpe calculation
        
        Returns:
            Dictionary with optimized weights and metrics
        """
        logger.info("Optimizing portfolio using MPT...")
        
        # Create efficient frontier
        ef = EfficientFrontier(self.mean_returns, self.cov_matrix)
        
        if target_return:
            # Optimize for target return
            weights = ef.efficient_return(target_return)
        else:
            # Optimize for maximum Sharpe ratio
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        
        cleaned_weights = ef.clean_weights()
        
        # Calculate portfolio performance
        expected_return, volatility, sharpe = ef.portfolio_performance(
            risk_free_rate=risk_free_rate, verbose=False
        )
        
        # Filter out zero weights
        nonzero_weights = {k: v for k, v in cleaned_weights.items() if v > 0.001}
        
        logger.info(f"Optimized portfolio: Return={expected_return:.2%}, "
                   f"Volatility={volatility:.2%}, Sharpe={sharpe:.2f}")
        
        return {
            'weights': nonzero_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'optimization_method': 'max_sharpe' if not target_return else 'efficient_return'
        }
    
    def calculate_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Calculate efficient frontier for visualization.
        
        Args:
            n_points: Number of points on the frontier
        
        Returns:
            DataFrame with returns and volatilities
        """
        logger.info("Calculating efficient frontier...")
        
        returns_range = np.linspace(
            self.mean_returns.min(), 
            self.mean_returns.max(), 
            n_points
        )
        
        frontier_volatility = []
        frontier_returns = []
        frontier_sharpe = []
        
        for target_return in returns_range:
            try:
                ef = EfficientFrontier(self.mean_returns, self.cov_matrix)
                ef.efficient_return(target_return)
                ret, vol, sharpe = ef.portfolio_performance(verbose=False)
                
                frontier_returns.append(ret)
                frontier_volatility.append(vol)
                frontier_sharpe.append(sharpe)
            except:
                continue
        
        frontier_df = pd.DataFrame({
            'Return': frontier_returns,
            'Volatility': frontier_volatility,
            'Sharpe': frontier_sharpe
        })
        
        logger.info(f"Efficient frontier calculated with {len(frontier_df)} points")
        return frontier_df
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns.
        
        Returns:
            Correlation matrix DataFrame
        """
        correlation = self.returns.corr()
        logger.info("Correlation matrix calculated")
        return correlation
    
    def identify_diversification_opportunities(self, threshold: float = 0.3) -> List[Tuple]:
        """
        Identify pairs of assets with low correlation for diversification.
        
        Args:
            threshold: Correlation threshold (lower = more diverse)
        
        Returns:
            List of (ticker1, ticker2, correlation) tuples
        """
        correlation = self.calculate_correlation_matrix()
        
        opportunities = []
        for i in range(len(self.tickers)):
            for j in range(i+1, len(self.tickers)):
                corr = correlation.iloc[i, j]
                if abs(corr) < threshold:
                    opportunities.append((
                        self.tickers[i], 
                        self.tickers[j], 
                        corr
                    ))
        
        opportunities.sort(key=lambda x: abs(x[2]))
        logger.info(f"Found {len(opportunities)} diversification opportunities")
        return opportunities[:10]  # Return top 10
    
    def calculate_sharpe_ratio(self, weights: Dict[str, float], 
                              risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio for given portfolio weights.
        
        Args:
            weights: Dictionary of ticker -> weight
            risk_free_rate: Risk-free rate
        
        Returns:
            Sharpe ratio
        """
        # Convert weights dict to array
        weight_array = np.array([weights.get(ticker, 0) for ticker in self.tickers])
        
        # Portfolio return
        portfolio_return = np.sum(self.mean_returns * weight_array)
        
        # Portfolio volatility
        portfolio_std = np.sqrt(
            np.dot(weight_array.T, np.dot(self.cov_matrix, weight_array))
        )
        
        # Sharpe ratio
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        
        return sharpe
    
    def monte_carlo_simulation(self, n_simulations: int = 10000, 
                              time_horizon: int = 252) -> Dict:
        """
        Run Monte Carlo simulation for portfolio risk assessment.
        
        Args:
            n_simulations: Number of simulation runs
            time_horizon: Time horizon in days (default 252 = 1 year)
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} scenarios...")
        
        # Get mean returns and covariance
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        # Run simulations
        portfolio_returns = []
        
        for _ in range(n_simulations):
            # Generate random weights
            weights = np.random.random(self.n_assets)
            weights /= weights.sum()
            
            # Calculate portfolio return
            returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
            portfolio_return = np.sum(returns @ weights, axis=0)
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        results = {
            'mean_return': portfolio_returns.mean(),
            'std_return': portfolio_returns.std(),
            'var_95': np.percentile(portfolio_returns, 5),
            'var_99': np.percentile(portfolio_returns, 1),
            'best_case': np.percentile(portfolio_returns, 95),
            'worst_case': np.percentile(portfolio_returns, 5),
            'all_returns': portfolio_returns
        }
        
        logger.info(f"Monte Carlo complete: Mean={results['mean_return']:.2%}, "
                   f"Std={results['std_return']:.2%}")
        return results
    
    def calculate_max_drawdown(self) -> Dict:
        """
        Calculate maximum drawdown for the portfolio.
        
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate cumulative returns
        cum_returns = (1 + self.returns.mean(axis=1)).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_idx = running_max[:max_dd_idx].idxmax()
        
        # Recovery
        recovery_idx = None
        if max_dd_idx in cum_returns.index:
            recovery_data = cum_returns[max_dd_idx:]
            recovery_threshold = running_max[max_dd_idx]
            recovered = recovery_data[recovery_data >= recovery_threshold]
            if len(recovered) > 0:
                recovery_idx = recovered.index[0]
        
        results = {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_idx,
            'peak_date': peak_idx,
            'recovery_date': recovery_idx,
            'drawdown_duration': (max_dd_idx - peak_idx).days if hasattr(max_dd_idx - peak_idx, 'days') else None
        }
        
        logger.info(f"Max drawdown: {max_dd:.2%}")
        return results
    
    def get_risk_summary(self, portfolio_value: float = 100000) -> pd.DataFrame:
        """
        Generate comprehensive risk summary.
        
        Args:
            portfolio_value: Total portfolio value
        
        Returns:
            DataFrame with risk metrics
        """
        logger.info("Generating comprehensive risk summary...")
        
        # VaR metrics
        var_metrics = self.calculate_var(portfolio_value)
        
        # Drawdown
        drawdown = self.calculate_max_drawdown()
        
        # Optimization
        optimal = self.optimize_portfolio()
        
        summary = {
            'Metric': [
                'Portfolio Value',
                'Expected Annual Return',
                'Annual Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'VaR (95%)',
                'CVaR (95%)',
                'Number of Assets',
                'Avg Correlation'
            ],
            'Value': [
                f"₹{portfolio_value:,.0f}",
                f"{optimal['expected_return']:.2%}",
                f"{optimal['volatility']:.2%}",
                f"{optimal['sharpe_ratio']:.2f}",
                f"{drawdown['max_drawdown']:.2%}",
                f"₹{var_metrics['historical_var']:,.0f}",
                f"₹{var_metrics['conditional_var']:,.0f}",
                f"{self.n_assets}",
                f"{self.calculate_correlation_matrix().values[np.triu_indices_from(self.calculate_correlation_matrix().values, k=1)].mean():.2f}"
            ]
        }
        
        return pd.DataFrame(summary)


# Example usage
if __name__ == "__main__":
    print("Risk Analytics & Portfolio Optimization - BlackRock Aladdin Competitive")
    print("="*75)
    
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    n_stocks = 10
    tickers = [f'STOCK{i}' for i in range(1, n_stocks+1)]
    
    # Create price data (random walk with drift)
    price_data = pd.DataFrame(
        np.cumsum(np.random.randn(len(dates), n_stocks) * 0.02 + 0.0005, axis=0) + 100,
        columns=tickers,
        index=dates
    )
    
    # Initialize risk analytics
    risk = RiskAnalytics(price_data)
    
    # Calculate VaR
    print("\n1. Value at Risk Analysis:")
    var = risk.calculate_var(portfolio_value=1000000)
    print(f"   Historical VaR (95%): ₹{var['historical_var']:,.0f}")
    print(f"   Parametric VaR (95%): ₹{var['parametric_var']:,.0f}")
    print(f"   Conditional VaR (95%): ₹{var['conditional_var']:,.0f}")
    
    # Optimize portfolio
    print("\n2. Portfolio Optimization:")
    optimal = risk.optimize_portfolio()
    print(f"   Expected Return: {optimal['expected_return']:.2%}")
    print(f"   Volatility: {optimal['volatility']:.2%}")
    print(f"   Sharpe Ratio: {optimal['sharpe_ratio']:.2f}")
    print(f"   Top Weights:")
    for ticker, weight in sorted(optimal['weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {ticker}: {weight:.1%}")
    
    # Diversification opportunities
    print("\n3. Diversification Opportunities:")
    opportunities = risk.identify_diversification_opportunities()
    for i, (t1, t2, corr) in enumerate(opportunities[:5], 1):
        print(f"   {i}. {t1} + {t2}: Correlation = {corr:.2f}")
    
    # Monte Carlo simulation
    print("\n4. Monte Carlo Simulation (10,000 scenarios):")
    mc = risk.monte_carlo_simulation(n_simulations=10000)
    print(f"   Expected Return: {mc['mean_return']:.2%}")
    print(f"   Risk (Std Dev): {mc['std_return']:.2%}")
    print(f"   Best Case (95th percentile): {mc['best_case']:.2%}")
    print(f"   Worst Case (5th percentile): {mc['worst_case']:.2%}")
    
    # Risk summary
    print("\n5. Comprehensive Risk Summary:")
    summary = risk.get_risk_summary(portfolio_value=1000000)
    print(summary.to_string(index=False))
    
    print("\n" + "="*75)
    print("Risk analytics module ready for BlackRock Aladdin-competitive deployment!")
