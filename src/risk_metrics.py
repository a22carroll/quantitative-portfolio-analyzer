"""
Risk Metrics Module

Comprehensive portfolio risk assessment functions including VaR calculations,
performance metrics, and correlation analysis. Implements industry-standard
risk measurement techniques used in quantitative finance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]
DEFAULT_RISK_FREE_RATE = 0.02

#All returns are assumed to be cleaned and over have more than 30 observations
class RiskMetrics:
    """
    Portfolio risk assessment and calculation engine.
    
    Provides comprehensive risk metrics including VaR calculations,
    performance ratios, and correlation analysis for portfolio management.
    """
    
    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE):
        """
        Initialize RiskMetrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate as decimal (default: 0.02)
        """
        pass
    
    def calculate_historical_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical simulation method.
        
        Args:
            returns: Series of asset returns
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            VaR as positive number (loss amount)
        """
        alpha = 1 - confidence_level
        var_percentile = np.percentile(returns, alpha * 100)
        historical_var = -var_percentile if var_percentile < 0 else 0.0
        
        return historical_var
    
    def calculate_parametric_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            returns: Series of asset returns
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            VaR as positive number (loss amount)
        """
        # Calculate mean and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Get z-score for the confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate parametric VaR
        parametric_var = -(mean_return + z_score * std_return)
        
        # Ensure VaR is non-negative
        parametric_var = max(parametric_var, 0.0)
        
        return parametric_var

    
    def calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95, num_simulations: int = 10000) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            returns: Series of asset returns
            confidence_level: Confidence level (0.95 = 95%)
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR as positive number (loss amount)
        """
        
        # Calculate mean and standard deviation from historical data
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns using normal distribution
        np.random.seed(42)  # For reproducible results
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calculate the percentile corresponding to the confidence level
        alpha = 1 - confidence_level
        var_percentile = np.percentile(simulated_returns, alpha * 100)
        
        # Return VaR as positive number (loss amount)
        monte_carlo_var = -var_percentile if var_percentile < 0 else 0.0
        
        return monte_carlo_var
    
    def calculate_conditional_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Series of asset returns
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            CVaR as positive number (expected loss beyond VaR)
        """
        historical_var = self.calculate_historical_var(returns, confidence_level)
        
        if historical_var == 0.0:
            return 0.0
        
        # Find all returns that are worse than the VaR threshold
        # VaR is positive (loss), so we look for returns <= -VaR (actual losses)
        var_threshold = -historical_var
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            # No losses beyond VaR, return the VaR itself
            return historical_var
        
        # Calculate the average of all losses beyond VaR
        expected_shortfall = -tail_losses.mean()  # Make positive (loss amount)
        
        return expected_shortfall
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Args:
            returns: Series of asset returns
            risk_free_rate: Risk-free rate (uses instance default if None)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / returns.std()
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Args:
            returns: Series of asset returns
            risk_free_rate: Risk-free rate (uses instance default if None)
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std()

        return sortino_ratio
    
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk vs market).
        
        Args:
            asset_returns: Series of asset returns
            market_returns: Series of market benchmark returns
            
        Returns:
            Beta coefficient
        """
        beta = asset_returns.cov(market_returns) / market_returns.var()
        return beta
    
    def calculate_portfolio_volatility(self, returns_df: pd.DataFrame, weights: List[float]) -> float:
        """
        Calculate portfolio volatility accounting for correlations.
        
        Args:
            returns_df: DataFrame with returns for each asset
            weights: List of portfolio weights (should sum to 1.0)
            
        Returns:
            Portfolio volatility (annualized)
        """
        weights = np.array(weights)
        cov_matrix = returns_df.cov() * 252  # Annualized covariance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        return np.sqrt(portfolio_variance)
    
    def calculate_portfolio_var(self, individual_vars: List[float], correlation_matrix: pd.DataFrame, weights: List[float]) -> float:
        """
        Calculate portfolio VaR from individual asset VaRs.
        
        Args:
            individual_vars: List of individual asset VaRs
            correlation_matrix: Asset correlation matrix
            weights: Portfolio weights
            
        Returns:
            Portfolio VaR
        """
        if len(individual_vars) != len(weights) or len(weights) != len(correlation_matrix):
            logger.error("Dimension mismatch in portfolio VaR calculation")
            return 0.0
        
        # Convert to numpy arrays
        vars_array = np.array(individual_vars)
        weights_array = np.array(weights)
        
        # Calculate portfolio VaR using matrix multiplication
        # Portfolio VaR = sqrt(w^T * (VaR * VaR^T * Corr) * w)
        var_matrix = np.outer(vars_array, vars_array) * correlation_matrix.values
        portfolio_var_squared = np.dot(weights_array.T, np.dot(var_matrix, weights_array))
        
        return np.sqrt(portfolio_var_squared)
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio assets.
        
        Args:
            returns_df: DataFrame with returns for each asset
            
        Returns:
            Correlation matrix
        """
        if returns_df.empty:
            logger.warning("Empty returns DataFrame provided")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def annualize_volatility(self, daily_vol: float) -> float:
        """
        Convert daily volatility to annualized volatility.
        
        Args:
            daily_vol: Daily volatility
            
        Returns:
            Annualized volatility
        """
        return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    def portfolio_risk_summary(self, portfolio_data: Dict, weights: List[float]) -> Dict:
        """
        Generate comprehensive risk assessment for portfolio.
        
        Args:
            portfolio_data: Dictionary with asset data from DataFetcher
            weights: Portfolio weights
            
        Returns:
            Dictionary with comprehensive risk metrics
        """
        try:
            # Extract tickers and validate weights
            tickers = list(portfolio_data.keys())
            if len(weights) != len(tickers):
                logger.error("Number of weights doesn't match number of assets")
                return {}
            
            # Normalize weights to ensure they sum to 1.0
            weights_array = np.array(weights)
            weights_normalized = weights_array / weights_array.sum()
            
            # Build returns DataFrame
            returns_data = {}
            for ticker in tickers:
                if (portfolio_data[ticker]['price_history'] is not None and 
                    not portfolio_data[ticker]['price_history'].empty):
                    returns_data[ticker] = portfolio_data[ticker]['price_history']['Daily_Return']
            
            if not returns_data:
                logger.error("No valid return data found")
                return {}
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            # Calculate individual asset metrics
            individual_metrics = {}
            individual_vars_95 = []
            individual_vars_99 = []
            
            for i, ticker in enumerate(tickers):
                if ticker in returns_df.columns:
                    returns = returns_df[ticker]
                    
                    # Calculate various metrics for this asset
                    var_95 = self.calculate_historical_var(returns, 0.95)
                    var_99 = self.calculate_historical_var(returns, 0.99)
                    sharpe = self.calculate_sharpe_ratio(returns)
                    vol = self.annualize_volatility(returns.std())
                    
                    individual_metrics[ticker] = {
                        'weight': weights_normalized[i],
                        'volatility_annualized': vol,
                        'var_95': var_95,
                        'var_99': var_99,
                        'sharpe_ratio': sharpe
                    }
                    
                    individual_vars_95.append(var_95)
                    individual_vars_99.append(var_99)
            
            # Calculate portfolio-level metrics
            portfolio_volatility = self.calculate_portfolio_volatility(returns_df, weights_normalized.tolist())
            correlation_matrix = self.calculate_correlation_matrix(returns_df)
            
            # Portfolio VaR calculations
            portfolio_var_95 = self.calculate_portfolio_var(individual_vars_95, correlation_matrix, weights_normalized.tolist())
            portfolio_var_99 = self.calculate_portfolio_var(individual_vars_99, correlation_matrix, weights_normalized.tolist())
            
            # Calculate portfolio returns for additional metrics
            portfolio_returns = (returns_df * weights_normalized).sum(axis=1)
            portfolio_sharpe = self.calculate_sharpe_ratio(portfolio_returns)
            portfolio_sortino = self.calculate_sortino_ratio(portfolio_returns)
            
            # Compile comprehensive summary
            risk_summary = {
                'portfolio_metrics': {
                    'total_volatility_annualized': portfolio_volatility,
                    'var_95_percent': portfolio_var_95,
                    'var_99_percent': portfolio_var_99,
                    'sharpe_ratio': portfolio_sharpe,
                    'sortino_ratio': portfolio_sortino,
                    'number_of_assets': len(tickers),
                    'effective_diversification': portfolio_volatility < np.average([m['volatility_annualized'] for m in individual_metrics.values()], weights=weights_normalized)
                },
                'individual_assets': individual_metrics,
                'correlation_matrix': correlation_matrix.to_dict(),
                'risk_analysis': {
                    'highest_risk_asset': max(individual_metrics.items(), key=lambda x: x[1]['volatility_annualized'])[0],
                    'lowest_risk_asset': min(individual_metrics.items(), key=lambda x: x[1]['volatility_annualized'])[0],
                    'diversification_benefit': {
                        'weighted_avg_volatility': np.average([m['volatility_annualized'] for m in individual_metrics.values()], weights=weights_normalized),
                        'portfolio_volatility': portfolio_volatility,
                        'risk_reduction': np.average([m['volatility_annualized'] for m in individual_metrics.values()], weights=weights_normalized) - portfolio_volatility
                    }
                },
                'summary_date': datetime.now().isoformat()
            }
            
            return risk_summary
            
        except Exception as e:
            logger.error(f"Error in portfolio risk summary: {e}")
            return {}