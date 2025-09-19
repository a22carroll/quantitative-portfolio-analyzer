"""
Quantitative Portfolio Analyzer - Core Package

A comprehensive toolkit for portfolio risk assessment and valuation analysis.
Implements industry-standard metrics including VaR calculations, factor analysis,
and multi-metric valuation models.

Author: Aidan
Version: 1.0.0
"""

from .portfolio_analyzer import PortfolioAnalyzer
from .risk_metrics import RiskMetrics
from .valuation_metrics import ValuationMetrics
from .data_fetcher import DataFetcher
# Add other imports as needed

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    'PortfolioAnalyzer',
    'RiskMetrics', 
    'ValuationMetrics',
    'DataFetcher'
]