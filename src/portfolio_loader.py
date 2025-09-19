"""
This module loads a csv portfolio file and cleans it.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
import time
from .data_fetcher import DataFetcher


class PortfolioLoader:
    def __init__(self, filepath: str, date_format: str = "%Y-%m-%d", fetcher: Optional[DataFetcher] = None):
        """
        Initialize PortfolioLoader with file path and date format.
        
        Args:
            filepath: Path to the CSV file containing portfolio data
            date_format: Date format in the CSV file
            fetcher: Optional DataFetcher instance for fetching prices
        """
        self.filepath = filepath
        self.date_format = date_format
        self.df = pd.DataFrame()
        self.fetcher = fetcher or DataFetcher()
    
    def load_portfolio(self) -> pd.DataFrame:
        """
        Load and clean the portfolio data from the CSV file.
        
        Returns:
            Cleaned DataFrame with portfolio data
        """
        try:
            self.df = pd.read_csv(self.filepath)
            self._clean_data()
            self._add_current_prices()
            return self.df
        except Exception as e:
            logging.error(f"Error loading portfolio from {self.filepath}: {e}")
            return pd.DataFrame()
    
    def _clean_data(self):
        """
        Clean and process the loaded DataFrame.
        """
        # Ensure required columns are present
        required_columns = {'Symbol', 'Quantity','Percent Of Account', 'Average Cost Basis'}
        if not required_columns.issubset(self.df.columns):
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        self.df.rename(columns={'Symbol': 'Ticker', 'Quantity': 'Shares','Percent Of Account': 'Weight', 'Average Cost Basis': 'Avg Cost'}, inplace=True)

        #Drop all columns except Ticker, Shares, Avg Cost
        self.df = self.df[['Ticker', 'Shares','Weight', 'Avg Cost']]
        
        # Drop rows with invalid Ticker
        self.df = self.df[self.df['Ticker'].apply(lambda x: self.fetcher.validate_ticker(x))]
            
        
        # Ensure 'Shares' is numeric and positive
        self.df['Shares'] = pd.to_numeric(self.df['Shares'], errors='coerce')
        self.df = self.df[self.df['Shares'] > 0]
        
    
    def _add_current_prices(self):
        """
        Add current prices to the DataFrame.
        """
        self.df['Current Price'] = self.df['Ticker'].apply(lambda ticker: self.fetcher.get_current_price(ticker))
        self.df['Current Value'] = self.df['Shares'] * self.df['Current Price']