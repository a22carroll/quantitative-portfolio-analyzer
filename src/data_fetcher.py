"""
This module provides functions to fetch and process data from various sources.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, delay_seconds: float = 0.1):
        """
        Initialize DataFetcher with rate limiting.
        
        Args:
            delay_seconds: Delay between API calls to avoid rate limits
        """
        self.delay_seconds = delay_seconds
        self._cache = {}  # Simple in-memory cache
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current stock price for a single ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Current price or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            
            if hist.empty:
                logger.warning(f"No price data found for {ticker}")
                return None
                
            return float(hist['Close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return None
    
    def get_price_history(self, ticker: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Get historical price data for a single ticker.
        
        Args:
            ticker: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data found for {ticker}")
                return None
            
            # Add daily returns column
            hist['Daily_Return'] = hist['Close'].pct_change()
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return None
    
    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """
        Get fundamental data for a single ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with fundamental metrics or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                logger.warning(f"No fundamental data found for {ticker}")
                return None
            
            # Extract key metrics with safe fallbacks
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'book_value': info.get('bookValue'),
                'earnings_growth': info.get('earningsGrowth'),
                'revenue_growth': info.get('revenueGrowth'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return None
    
    def get_portfolio_data(self, tickers: List[str], period: str = '1y') -> Dict[str, Dict]:
        """
        Get comprehensive data for multiple tickers (portfolio).
        
        Args:
            tickers: List of stock symbols
            period: Time period for historical data
            
        Returns:
            Dictionary with ticker as key and data dict as value
        """
        portfolio_data = {}
        total_tickers = len(tickers)
        
        logger.info(f"Fetching data for {total_tickers} tickers...")
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{total_tickers})")
            
            # Get all data for this ticker
            current_price = self.get_current_price(ticker)
            price_history = self.get_price_history(ticker, period)
            fundamentals = self.get_fundamentals(ticker)
            
            portfolio_data[ticker] = {
                'current_price': current_price,
                'price_history': price_history,
                'fundamentals': fundamentals,
                'fetch_time': datetime.now(),
                'valid': all([current_price is not None, 
                            price_history is not None, 
                            fundamentals is not None])
            }
            
            # Rate limiting
            if i < total_tickers:  # Don't delay after last ticker
                time.sleep(self.delay_seconds)
        
        valid_count = sum(1 for data in portfolio_data.values() if data['valid'])
        logger.info(f"Successfully fetched data for {valid_count}/{total_tickers} tickers")
        
        return portfolio_data
    
    def get_market_benchmark(self, benchmark: str = '^GSPC', period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Get benchmark index data (default: S&P 500).
        
        Args:
            benchmark: Benchmark symbol ('^GSPC' for S&P 500, '^DJI' for Dow, etc.)
            period: Time period
            
        Returns:
            DataFrame with benchmark price history
        """
        logger.info(f"Fetching benchmark data for {benchmark}")
        return self.get_price_history(benchmark, period)
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Quick validation to check if ticker exists.
        
        Args:
            ticker: Stock symbol to validate
            
        Returns:
            True if ticker is valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return bool(info and 'symbol' in info)
        except:
            return False
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (10-year Treasury).
        
        Returns:
            Risk-free rate as decimal (e.g., 0.045 for 4.5%)
        """
        try:
            treasury = yf.Ticker('^TNX')
            hist = treasury.history(period='5d')
            
            if not hist.empty:
                # ^TNX gives percentage, convert to decimal
                return hist['Close'].iloc[-1] / 100
            else:
                logger.warning("Could not fetch risk-free rate, using default 2%")
                return 0.02  # Default 2%
                
        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {e}")
            return 0.02  # Default fallback

