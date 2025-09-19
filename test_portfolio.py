"""
Test script for the complete portfolio analysis pipeline.
Run this to test your Fidelity CSV analysis end-to-end.
"""

from src.portfolio_analyzer import PortfolioAnalyzer

def test_my_portfolio():
    """Test with your actual Fidelity CSV file"""
    analyzer = PortfolioAnalyzer()
    
    # Update this path to your CSV file location
    csv_path = r"C:\Users\a22ca\OneDrive\Desktop\Portfolio_Positions_Sep-2025.csv"
    
    print("🚀 Starting portfolio analysis...")
    print(f"Loading portfolio from: {csv_path}")
    
    results = analyzer.analyze_portfolio(csv_path)
    
    # Check for errors
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Print results
    print("✅ Analysis completed successfully!")
    print("\n📊 Portfolio Overview:")
    print(f"- Total positions: {results['analysis_metadata']['total_positions']}")
    print(f"- Positions with valid data: {results['analysis_metadata']['valid_data_positions']}")
    
    # Risk metrics
    if 'risk_analysis' in results and results['risk_analysis']:
        risk = results['risk_analysis']['portfolio_metrics']
        print(f"\n⚠️ Risk Metrics:")
        print(f"- Portfolio volatility: {risk.get('total_volatility_annualized', 'N/A'):.2%}")
        print(f"- VaR (95%): {risk.get('var_95_percent', 'N/A'):.2%}")
        print(f"- Sharpe ratio: {risk.get('sharpe_ratio', 'N/A'):.2f}")
    
    # Valuation metrics
    if 'valuation_analysis' in results and results['valuation_analysis']:
        val = results['valuation_analysis']['portfolio_metrics']
        print(f"\n💰 Valuation Metrics:")
        print(f"- Weighted P/E ratio: {val.get('weighted_pe_ratio', 'N/A'):.1f}")
        print(f"- Dividend yield: {val.get('portfolio_dividend_yield', 'N/A'):.2f}%")
    
    # Key insights
    if 'summary' in results and 'key_insights' in results['summary']:
        print(f"\n🔍 Key Insights:")
        for insight in results['summary']['key_insights']:
            print(f"- {insight}")
    
    # Save detailed results
    output_file = "my_portfolio_analysis.json"
    analyzer.save_analysis_to_json(results, output_file)
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    print("Portfolio Analysis Test")
    print("=" * 40)
    
    # Run the analysis
    results = test_my_portfolio()
    
    if results and 'error' not in results:
        print("\n🎉 Test completed successfully!")
        print("Check the JSON file for complete analysis details.")
    else:
        print("\n❌ Test failed. Check the error messages above.")