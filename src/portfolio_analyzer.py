"""
Portfolio Analyzer - Main Analysis Engine

Orchestrates all portfolio analysis components including data loading,
risk assessment, and valuation analysis into comprehensive reports.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

from .portfolio_loader import PortfolioLoader
from .data_fetcher import DataFetcher
from .risk_metrics import RiskMetrics
from .valuation_metrics import ValuationMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """
    Main portfolio analysis engine that coordinates all analysis components.
    
    Provides complete portfolio assessment including risk metrics,
    valuation analysis, and comprehensive reporting.
    """
    
    def __init__(self, delay_seconds: float = 0.1):
        """
        Initialize PortfolioAnalyzer with all component modules.
        
        Args:
            delay_seconds: Delay between API calls for rate limiting
        """
        self.portfolio_loader = None
        self.data_fetcher = DataFetcher(delay_seconds=delay_seconds)
        self.risk_metrics = RiskMetrics()
        self.valuation_metrics = ValuationMetrics()
        self.portfolio_data = None
        self.portfolio_df = None
    
    def load_portfolio_from_csv(self, csv_path: str) -> bool:
        """
        Load portfolio from Fidelity CSV file.
        
        Args:
            csv_path: Path to the Fidelity CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading portfolio from {csv_path}")
            
            # Create PortfolioLoader with the filepath
            self.portfolio_loader = PortfolioLoader(csv_path)
            
            # Load the portfolio
            self.portfolio_df = self.portfolio_loader.load_portfolio()

            # Add this debugging:
            print(f"Portfolio columns: {self.portfolio_df.columns.tolist()}")
            print(f"First few rows:\n{self.portfolio_df.head()}")
            
            if self.portfolio_df is not None and not self.portfolio_df.empty:
                logger.info(f"Successfully loaded {len(self.portfolio_df)} positions")
                return True
            else:
                logger.error("Failed to load portfolio from CSV")
                return False
                
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return False
    
    def load_portfolio_from_dataframe(self, portfolio_df: pd.DataFrame) -> bool:
        """
        Load portfolio from DataFrame with Ticker and Weight columns.
        
        Args:
            portfolio_df: DataFrame with 'Ticker' and 'Weight' columns
            
        Returns:
            True if successful, False otherwise
        """
        try:
            required_columns = ['Ticker', 'Weight']
            if not all(col in portfolio_df.columns for col in required_columns):
                logger.error(f"DataFrame must contain columns: {required_columns}")
                return False
            
            self.portfolio_df = portfolio_df.copy()
            logger.info(f"Successfully loaded {len(self.portfolio_df)} positions from DataFrame")
            return True
            
        except Exception as e:
            logger.error(f"Error loading portfolio from DataFrame: {e}")
            return False
    
    def fetch_market_data(self, period: str = '1y') -> bool:
        """
        Fetch market data for all portfolio positions.
        
        Args:
            period: Time period for historical data
            
        Returns:
            True if successful, False otherwise
        """
        if self.portfolio_df is None:
            logger.error("No portfolio loaded. Call load_portfolio_from_csv() first.")
            return False
        
        try:
            tickers = self.portfolio_df['Ticker'].tolist()
            logger.info(f"Fetching market data for {len(tickers)} tickers")
            
            self.portfolio_data = self.data_fetcher.get_portfolio_data(tickers, period)
            
            valid_count = sum(1 for data in self.portfolio_data.values() if data['valid'])
            logger.info(f"Successfully fetched data for {valid_count}/{len(tickers)} tickers")
            
            return valid_count > 0
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return False
    
    def calculate_risk_metrics(self) -> Optional[Dict]:
        """
        Calculate comprehensive risk metrics for the portfolio.
        
        Returns:
            Dictionary with risk analysis or None if error
        """
        if self.portfolio_data is None:
            logger.error("No market data available. Call fetch_market_data() first.")
            return None
        
        try:
            weights = self.portfolio_df['Weight'].tolist()
            risk_summary = self.risk_metrics.portfolio_risk_summary(self.portfolio_data, weights)
            
            if risk_summary:
                logger.info("Risk metrics calculated successfully")
                return risk_summary
            else:
                logger.error("Failed to calculate risk metrics")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None
    
    def calculate_valuation_metrics(self) -> Optional[Dict]:
        """
        Calculate comprehensive valuation metrics for the portfolio.
        
        Returns:
            Dictionary with valuation analysis or None if error
        """
        if self.portfolio_data is None:
            logger.error("No market data available. Call fetch_market_data() first.")
            return None
        
        try:
            weights = self.portfolio_df['Weight'].tolist()
            valuation_summary = self.valuation_metrics.portfolio_valuation_summary(self.portfolio_data, weights)
            
            if valuation_summary:
                logger.info("Valuation metrics calculated successfully")
                return valuation_summary
            else:
                logger.error("Failed to calculate valuation metrics")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating valuation metrics: {e}")
            return None
    
    def analyze_portfolio(self, csv_path: Optional[str] = None, 
                         portfolio_df: Optional[pd.DataFrame] = None,
                         period: str = '1y') -> Dict:
        """
        Complete portfolio analysis workflow.
        
        Args:
            csv_path: Path to Fidelity CSV (if loading from file)
            portfolio_df: Portfolio DataFrame (if loading from memory)
            period: Time period for historical data
            
        Returns:
            Complete analysis results dictionary
        """
        try:
            # Load portfolio
            if csv_path:
                if not self.load_portfolio_from_csv(csv_path):
                    return {'error': 'Failed to load portfolio from CSV'}
            elif portfolio_df is not None:
                if not self.load_portfolio_from_dataframe(portfolio_df):
                    return {'error': 'Failed to load portfolio from DataFrame'}
            else:
                return {'error': 'Must provide either csv_path or portfolio_df'}
            
            # Fetch market data
            if not self.fetch_market_data(period):
                return {'error': 'Failed to fetch market data'}
            
            # Calculate metrics
            risk_analysis = self.calculate_risk_metrics()
            valuation_analysis = self.calculate_valuation_metrics()
            
            # Compile results
            analysis_results = {
                'analysis_metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'period_analyzed': period,
                    'total_positions': len(self.portfolio_df),
                    'valid_data_positions': sum(1 for data in self.portfolio_data.values() if data['valid'])
                },
                'portfolio_composition': self.portfolio_df.to_dict('records'),
                'risk_analysis': risk_analysis if risk_analysis else {},
                'valuation_analysis': valuation_analysis if valuation_analysis else {},
                'summary': self._generate_executive_summary(risk_analysis, valuation_analysis)
            }
            
            logger.info("Portfolio analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _generate_executive_summary(self, risk_analysis: Optional[Dict], 
                                   valuation_analysis: Optional[Dict]) -> Dict:
        """
        Generate high-level executive summary of portfolio analysis.
        
        Args:
            risk_analysis: Risk metrics dictionary
            valuation_analysis: Valuation metrics dictionary
            
        Returns:
            Executive summary dictionary
        """
        summary = {
            'overall_assessment': 'Analysis completed',
            'key_insights': []
        }
        
        try:
            # Risk insights
            if risk_analysis and 'portfolio_metrics' in risk_analysis:
                risk_metrics = risk_analysis['portfolio_metrics']
                
                if 'sharpe_ratio' in risk_metrics:
                    sharpe = risk_metrics['sharpe_ratio']
                    if sharpe > 1.0:
                        summary['key_insights'].append(f"Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
                    elif sharpe < 0:
                        summary['key_insights'].append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
                
                if 'effective_diversification' in risk_metrics:
                    if risk_metrics['effective_diversification']:
                        summary['key_insights'].append("Portfolio shows effective diversification benefits")
                    else:
                        summary['key_insights'].append("Limited diversification benefits detected")
            
            # Valuation insights  
            if valuation_analysis and 'market_comparison' in valuation_analysis:
                market_comp = valuation_analysis['market_comparison']
                
                if market_comp.get('pe_vs_market') == 'Undervalued':
                    summary['key_insights'].append("Portfolio appears undervalued vs market (P/E basis)")
                elif market_comp.get('pe_vs_market') == 'Overvalued':
                    summary['key_insights'].append("Portfolio appears overvalued vs market (P/E basis)")
                
                if market_comp.get('dividend_attractiveness') == 'Above Average':
                    summary['key_insights'].append("Above-average dividend yield portfolio")
            
            # Default insight if none found
            if not summary['key_insights']:
                summary['key_insights'].append("Portfolio analysis completed - review detailed metrics")
                
        except Exception as e:
            logger.warning(f"Error generating executive summary: {e}")
            summary['key_insights'].append("Unable to generate insights - check detailed analysis")
        
        return summary
    
    def save_analysis_to_json(self, analysis_results: Dict, output_path: str) -> bool:
        """
        Save analysis results to JSON file.
        
        Args:
            analysis_results: Analysis results dictionary
            output_path: Path for output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis to JSON: {e}")
            return False
    
    def get_portfolio_summary(self) -> Optional[Dict]:
        """
        Get basic portfolio composition summary.
        
        Returns:
            Portfolio summary dictionary or None if no portfolio loaded
        """
        if self.portfolio_df is None:
            return None
        
        summary = {
            'total_positions': len(self.portfolio_df),
            'tickers': self.portfolio_df['Ticker'].tolist(),
            'weights': self.portfolio_df['Weight'].tolist(),
            'largest_position': {
                'ticker': self.portfolio_df.loc[self.portfolio_df['Weight'].idxmax(), 'Ticker'],
                'weight': self.portfolio_df['Weight'].max()
            },
            'weight_distribution': {
                'mean': self.portfolio_df['Weight'].mean(),
                'std': self.portfolio_df['Weight'].std(),
                'sum': self.portfolio_df['Weight'].sum()
            }
        }
        
        return summary


# Simple test function
def test_portfolio_analyzer():
    """Test basic functionality of PortfolioAnalyzer"""
    analyzer = PortfolioAnalyzer()
    
    # Test with mock DataFrame
    test_portfolio = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'Weight': [40, 35, 25]
    })
    
    print("Testing portfolio analyzer...")
    success = analyzer.load_portfolio_from_dataframe(test_portfolio)
    print(f"Portfolio loading: {'Success' if success else 'Failed'}")
    
    summary = analyzer.get_portfolio_summary()
    if summary:
        print(f"Portfolio summary: {summary['total_positions']} positions")
        print(f"Largest position: {summary['largest_position']['ticker']} ({summary['largest_position']['weight']}%)")


if __name__ == "__main__":
    test_portfolio_analyzer()