"""
Valuation Metrics Module

Portfolio valuation analysis including fundamental ratios,
dividend metrics, and fair value assessments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValuationMetrics:
    """
    Portfolio valuation assessment and calculation engine.
    
    Provides fundamental analysis metrics including P/E ratios,
    dividend yields, and portfolio-weighted valuations.
    """
    
    def __init__(self):
        """Initialize ValuationMetrics calculator."""
        pass
    
    def calculate_weighted_pe(self, portfolio_data: Dict, weights: List[float]) -> float:
        """
        Calculate portfolio-weighted P/E ratio.
        
        Args:
            portfolio_data: Dictionary with asset data from DataFetcher
            weights: Portfolio weights (as percentages)
            
        Returns:
            Weighted P/E ratio
        """
        if not portfolio_data or not weights:
            logger.warning("Empty portfolio data or weights provided")
            return 0.0
        
        tickers = list(portfolio_data.keys())
        if len(weights) != len(tickers):
            logger.error("Number of weights doesn't match number of assets")
            return 0.0
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_normalized = weights_array / weights_array.sum()
        
        weighted_pe = 0.0
        total_weight = 0.0
        
        for i, ticker in enumerate(tickers):
            fundamentals = portfolio_data[ticker].get('fundamentals')
            if fundamentals and fundamentals.get('trailing_pe'):
                pe_ratio = fundamentals['trailing_pe']
                if pe_ratio > 0:  # Only include positive P/E ratios
                    weighted_pe += pe_ratio * weights_normalized[i]
                    total_weight += weights_normalized[i]
        
        return weighted_pe / total_weight if total_weight > 0 else 0.0
    
    def calculate_weighted_pb(self, portfolio_data: Dict, weights: List[float]) -> float:
        """
        Calculate portfolio-weighted P/B ratio.
        
        Args:
            portfolio_data: Dictionary with asset data from DataFetcher
            weights: Portfolio weights (as percentages)
            
        Returns:
            Weighted P/B ratio
        """
        if not portfolio_data or not weights:
            logger.warning("Empty portfolio data or weights provided")
            return 0.0
        
        tickers = list(portfolio_data.keys())
        if len(weights) != len(tickers):
            logger.error("Number of weights doesn't match number of assets")
            return 0.0
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_normalized = weights_array / weights_array.sum()
        
        weighted_pb = 0.0
        total_weight = 0.0
        
        for i, ticker in enumerate(tickers):
            fundamentals = portfolio_data[ticker].get('fundamentals')
            if fundamentals and fundamentals.get('price_to_book'):
                pb_ratio = fundamentals['price_to_book']
                if pb_ratio > 0:  # Only include positive P/B ratios
                    weighted_pb += pb_ratio * weights_normalized[i]
                    total_weight += weights_normalized[i]
        
        return weighted_pb / total_weight if total_weight > 0 else 0.0
    
    def calculate_weighted_ps(self, portfolio_data: Dict, weights: List[float]) -> float:
        """
        Calculate portfolio-weighted P/S ratio.
        
        Args:
            portfolio_data: Dictionary with asset data from DataFetcher
            weights: Portfolio weights (as percentages)
            
        Returns:
            Weighted P/S ratio
        """
        if not portfolio_data or not weights:
            logger.warning("Empty portfolio data or weights provided")
            return 0.0
        
        tickers = list(portfolio_data.keys())
        if len(weights) != len(tickers):
            logger.error("Number of weights doesn't match number of assets")
            return 0.0
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_normalized = weights_array / weights_array.sum()
        
        weighted_ps = 0.0
        total_weight = 0.0
        
        for i, ticker in enumerate(tickers):
            fundamentals = portfolio_data[ticker].get('fundamentals')
            if fundamentals and fundamentals.get('price_to_sales'):
                ps_ratio = fundamentals['price_to_sales']
                if ps_ratio > 0:  # Only include positive P/S ratios
                    weighted_ps += ps_ratio * weights_normalized[i]
                    total_weight += weights_normalized[i]
        
        return weighted_ps / total_weight if total_weight > 0 else 0.0
    
    def calculate_dividend_yield(self, portfolio_data: Dict, weights: List[float]) -> float:
        """
        Calculate portfolio-weighted dividend yield.
        
        Args:
            portfolio_data: Dictionary with asset data from DataFetcher
            weights: Portfolio weights (as percentages)
            
        Returns:
            Weighted dividend yield as percentage
        """
        if not portfolio_data or not weights:
            logger.warning("Empty portfolio data or weights provided")
            return 0.0
        
        tickers = list(portfolio_data.keys())
        if len(weights) != len(tickers):
            logger.error("Number of weights doesn't match number of assets")
            return 0.0
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_normalized = weights_array / weights_array.sum()
        
        weighted_yield = 0.0
        
        for i, ticker in enumerate(tickers):
            fundamentals = portfolio_data[ticker].get('fundamentals')
            if fundamentals and fundamentals.get('dividend_yield'):
                div_yield = fundamentals['dividend_yield'] * 100  # Convert to percentage
                weighted_yield += div_yield * weights_normalized[i]
        
        return weighted_yield
    
    def calculate_momentum_score(self, price_history: pd.DataFrame) -> float:
        """
        Calculate simple momentum score based on price performance.
        
        Args:
            price_history: DataFrame with price history
            
        Returns:
            Momentum score (3-month vs 1-month return ratio)
        """
        if price_history is None or price_history.empty:
            return 0.0
        
        prices = price_history['Close']
        
        if len(prices) < 90:  # Need at least 3 months of data
            return 0.0
        
        # Calculate returns over different periods
        current_price = prices.iloc[-1]
        price_1m = prices.iloc[-22] if len(prices) >= 22 else prices.iloc[0]
        price_3m = prices.iloc[-66] if len(prices) >= 66 else prices.iloc[0]
        
        # Calculate returns
        return_1m = (current_price - price_1m) / price_1m
        return_3m = (current_price - price_3m) / price_3m
        
        # Momentum score: favor recent acceleration
        if return_1m == 0:
            return 0.0
        
        momentum_score = return_3m / return_1m if return_1m != 0 else 0.0
        return momentum_score
    
    def portfolio_valuation_summary(self, portfolio_data: Dict, weights: List[float]) -> Dict:
        """
        Generate comprehensive valuation assessment for portfolio.
        
        Args:
            portfolio_data: Dictionary with asset data from DataFetcher
            weights: Portfolio weights
            
        Returns:
            Dictionary with comprehensive valuation metrics
        """
        try:
            if not portfolio_data or not weights:
                logger.error("Empty portfolio data or weights provided")
                return {}
            
            tickers = list(portfolio_data.keys())
            if len(weights) != len(tickers):
                logger.error("Number of weights doesn't match number of assets")
                return {}
            
            # Normalize weights
            weights_array = np.array(weights)
            weights_normalized = weights_array / weights_array.sum()
            
            # Calculate portfolio-level metrics
            weighted_pe = self.calculate_weighted_pe(portfolio_data, weights)
            weighted_pb = self.calculate_weighted_pb(portfolio_data, weights)
            weighted_ps = self.calculate_weighted_ps(portfolio_data, weights)
            dividend_yield = self.calculate_dividend_yield(portfolio_data, weights)
            
            # Calculate individual asset metrics
            individual_valuations = {}
            sector_breakdown = {}
            
            for i, ticker in enumerate(tickers):
                fundamentals = portfolio_data[ticker].get('fundamentals')
                price_history = portfolio_data[ticker].get('price_history')
                
                if fundamentals:
                    # Individual metrics
                    individual_valuations[ticker] = {
                        'weight': weights_normalized[i],
                        'pe_ratio': fundamentals.get('trailing_pe'),
                        'pb_ratio': fundamentals.get('price_to_book'),
                        'ps_ratio': fundamentals.get('price_to_sales'),
                        'dividend_yield': fundamentals.get('dividend_yield', 0) * 100 if fundamentals.get('dividend_yield') else 0,
                        'momentum_score': self.calculate_momentum_score(price_history),
                        'sector': fundamentals.get('sector', 'Unknown')
                    }
                    
                    # Sector breakdown
                    sector = fundamentals.get('sector', 'Unknown')
                    if sector not in sector_breakdown:
                        sector_breakdown[sector] = 0.0
                    sector_breakdown[sector] += weights_normalized[i]
            
            # Market comparison benchmarks (approximate market averages)
            market_benchmarks = {
                'market_pe': 20.0,
                'market_pb': 3.0,
                'market_ps': 2.5,
                'market_dividend_yield': 2.0
            }
            
            # Valuation assessment
            valuation_signals = {
                'pe_vs_market': 'Undervalued' if weighted_pe < market_benchmarks['market_pe'] else 'Overvalued' if weighted_pe > 0 else 'No Data',
                'pb_vs_market': 'Undervalued' if weighted_pb < market_benchmarks['market_pb'] else 'Overvalued' if weighted_pb > 0 else 'No Data',
                'dividend_attractiveness': 'Above Average' if dividend_yield > market_benchmarks['market_dividend_yield'] else 'Below Average'
            }
            
            # Compile comprehensive summary
            valuation_summary = {
                'portfolio_metrics': {
                    'weighted_pe_ratio': weighted_pe,
                    'weighted_pb_ratio': weighted_pb,
                    'weighted_ps_ratio': weighted_ps,
                    'portfolio_dividend_yield': dividend_yield,
                    'number_of_assets': len(tickers)
                },
                'market_comparison': {
                    'pe_vs_market': valuation_signals['pe_vs_market'],
                    'pb_vs_market': valuation_signals['pb_vs_market'],
                    'dividend_attractiveness': valuation_signals['dividend_attractiveness']
                },
                'individual_assets': individual_valuations,
                'sector_breakdown': sector_breakdown,
                'valuation_insights': {
                    'most_expensive_pe': max([v for v in individual_valuations.values() if v.get('pe_ratio') and v['pe_ratio'] > 0], 
                                           key=lambda x: x['pe_ratio'], default={'pe_ratio': 0})['pe_ratio'] if individual_valuations else 0,
                    'cheapest_pe': min([v for v in individual_valuations.values() if v.get('pe_ratio') and v['pe_ratio'] > 0], 
                                     key=lambda x: x['pe_ratio'], default={'pe_ratio': 0})['pe_ratio'] if individual_valuations else 0,
                    'highest_dividend_yield': max([v['dividend_yield'] for v in individual_valuations.values()], default=0)
                },
                'summary_date': datetime.now().isoformat()
            }
            
            return valuation_summary
            
        except Exception as e:
            logger.error(f"Error in portfolio valuation summary: {e}")
            return {}


# Simple test function
def test_valuation_metrics():
    """Test basic functionality of ValuationMetrics"""
    valuator = ValuationMetrics()
    
    # Mock portfolio data for testing
    mock_data = {
        'AAPL': {
            'fundamentals': {
                'trailing_pe': 25.0,
                'price_to_book': 5.0,
                'price_to_sales': 6.0,
                'dividend_yield': 0.005,
                'sector': 'Technology'
            },
            'price_history': pd.DataFrame({
                'Close': [150, 155, 160, 158, 162, 165]
            })
        }
    }
    
    weights = [100]  # 100% weight
    
    pe_ratio = valuator.calculate_weighted_pe(mock_data, weights)
    print(f"Weighted P/E: {pe_ratio}")
    
    summary = valuator.portfolio_valuation_summary(mock_data, weights)
    print(f"Valuation summary keys: {list(summary.keys())}")


if __name__ == "__main__":
    test_valuation_metrics()