"""
NHL xG Analysis Runner
Simple script to run different types of analysis using consolidated modules
"""

import sys
import argparse
from nhl_xg_core import NHLxGAnalyzer
from nhl_business_analysis import NHLBusinessAnalyzer

def run_basic_analysis():
    """Run basic xG model analysis."""
    print("🏒 RUNNING BASIC NHL xG ANALYSIS")
    print("="*60)
    
    analyzer = NHLxGAnalyzer()
    analyzer.load_shot_data()
    analyzer.engineer_features()
    analyzer.train_models()
    analyzer.analyze_business_constraints()
    analyzer.create_comprehensive_visualization('basic_analysis.png')
    analyzer.generate_summary_report()

def run_business_analysis():
    """Run comprehensive business analysis."""
    print("🏒 RUNNING BUSINESS CONSTRAINT ANALYSIS")
    print("="*60)
    
    analyzer = NHLBusinessAnalyzer()
    analyzer.load_shot_data()
    analyzer.engineer_features()
    analyzer.train_models()
    analyzer.analyze_dual_constraints()
    analyzer.analyze_constraint_sensitivity()
    analyzer.analyze_prefiltering_strategies()
    analyzer.combination_results = analyzer.test_prefilter_model_combinations()
    analyzer.create_business_visualization('business_analysis.png')
    analyzer.generate_business_recommendations()

def run_custom_analysis(alpha_max=0.25, beta_max=0.40):
    """Run analysis with custom constraints."""
    print(f"🏒 RUNNING CUSTOM ANALYSIS (α ≤ {alpha_max:.1%}, β ≤ {beta_max:.1%})")
    print("="*60)
    
    analyzer = NHLBusinessAnalyzer()
    analyzer.load_shot_data()
    analyzer.engineer_features()
    analyzer.train_models()
    analyzer.analyze_dual_constraints(alpha_max, beta_max)
    analyzer.analyze_prefiltering_strategies()
    analyzer.combination_results = analyzer.test_prefilter_model_combinations()
    analyzer.create_business_visualization(f'custom_analysis_a{int(alpha_max*100)}_b{int(beta_max*100)}.png')
    analyzer.generate_business_recommendations()

def main():
    """Main runner with command line options."""
    parser = argparse.ArgumentParser(description='NHL xG Analysis Runner')
    parser.add_argument('--analysis', choices=['basic', 'business', 'custom'], 
                       default='basic', help='Type of analysis to run')
    parser.add_argument('--alpha', type=float, default=0.25, 
                       help='Maximum miss rate (alpha) for custom analysis')
    parser.add_argument('--beta', type=float, default=0.40, 
                       help='Maximum review rate (beta) for custom analysis')
    
    args = parser.parse_args()
    
    if args.analysis == 'basic':
        run_basic_analysis()
    elif args.analysis == 'business':
        run_business_analysis()
    elif args.analysis == 'custom':
        run_custom_analysis(args.alpha, args.beta)
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 