"""
NHL xG Business Analysis Module
Consolidated business constraint optimization and pre-filtering analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score
try:
    from .nhl_xg_core import NHLxGAnalyzer
except ImportError:
    from nhl_xg_core import NHLxGAnalyzer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

class NHLBusinessAnalyzer(NHLxGAnalyzer):
    """Extended analyzer with business constraint optimization."""
    
    def __init__(self, db_path='nhl_stats.db'):
        super().__init__(db_path)
        self.constraint_results = {}
        self.prefilter_results = {}
    
    def analyze_dual_constraints(self, alpha_max=0.25, beta_max=0.40):
        """Analyze models under dual business constraints."""
        if not self.results:
            raise ValueError("Must train models first")
            
        print(f"\nDUAL-CONSTRAINT OPTIMIZATION")
        print("="*60)
        print(f"Target: α ≤ {alpha_max:.1%} (miss rate), β ≤ {beta_max:.1%} (review rate)")
        print("Using F1 score as harmonized optimization metric")
        
        constraint_results = {}
        
        for model_name, result in self.results.items():
            y_true = result['y_test']
            y_proba = result['y_pred_proba']
            
            # Find optimal threshold under dual constraints
            thresholds = np.linspace(0.01, 0.9, 200)
            best_threshold = None
            best_f1 = 0
            best_metrics = None
            
            for threshold in thresholds:
                predictions = y_proba >= threshold
                
                if np.sum(predictions) == 0:
                    continue
                    
                # Calculate metrics
                tp = np.sum((predictions == 1) & (y_true == 1))
                fp = np.sum((predictions == 1) & (y_true == 0))
                fn = np.sum((predictions == 0) & (y_true == 1))
                
                if tp + fp == 0 or tp + fn == 0:
                    continue
                    
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                miss_rate = fn / (tp + fn)
                review_rate = (tp + fp) / len(y_true)
                
                # Check dual constraints
                alpha_constraint = miss_rate <= alpha_max
                beta_constraint = review_rate <= beta_max
                
                if alpha_constraint and beta_constraint:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                        best_metrics = {
                            'threshold': threshold,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'miss_rate': miss_rate,
                            'review_rate': review_rate,
                            'alpha_compliant': alpha_constraint,
                            'beta_compliant': beta_constraint
                        }
            
            if best_metrics:
                constraint_results[model_name] = best_metrics
                print(f"✅ {model_name}: F1={best_metrics['f1_score']:.3f}, α={best_metrics['miss_rate']:.1%}, β={best_metrics['review_rate']:.1%}")
            else:
                print(f"❌ {model_name}: No feasible solution under dual constraints")
        
        self.constraint_results = constraint_results
        return constraint_results
    
    def analyze_constraint_sensitivity(self):
        """Analyze sensitivity to different constraint combinations."""
        print(f"\nCONSTRAINT SENSITIVITY ANALYSIS")
        print("="*50)
        
        constraint_combinations = [
            (0.15, 0.30),  # Very strict
            (0.20, 0.35),  # Strict
            (0.25, 0.40),  # Target
            (0.30, 0.45),  # Moderate
            (0.35, 0.50),  # Relaxed
        ]
        
        sensitivity_results = {}
        
        for alpha_max, beta_max in constraint_combinations:
            print(f"Testing α ≤ {alpha_max:.1%}, β ≤ {beta_max:.1%}...")
            
            results = self.analyze_dual_constraints(alpha_max, beta_max)
            
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                sensitivity_results[(alpha_max, beta_max)] = {
                    'best_model': best_model[0],
                    'best_f1': best_model[1]['f1_score'],
                    'feasible_models': len(results)
                }
                print(f"  Best: {best_model[0]} (F1: {best_model[1]['f1_score']:.3f})")
            else:
                sensitivity_results[(alpha_max, beta_max)] = {
                    'best_model': None,
                    'best_f1': 0,
                    'feasible_models': 0
                }
                print(f"  No feasible solutions")
        
        return sensitivity_results
    
    def analyze_prefiltering_strategies(self):
        """Analyze intelligent pre-filtering strategies."""
        if self.shot_events is None:
            raise ValueError("Must load shot data first")
            
        print(f"\nINTELLIGENT PRE-FILTERING ANALYSIS")
        print("="*50)
        
        df = self.shot_events.copy()
        total_shots = len(df)
        total_goals = df['is_goal'].sum()
        
        # Define pre-filtering strategies
        strategies = {
            'Conservative Distance': df['distance_to_net'] <= 80,
            'High Value Zones': (
                (df['distance_to_net'] <= 40) |
                (df['in_slot'] == 1) |
                (df['in_crease'] == 1) |
                (df['is_tip_in'] == 1) |
                (df['final_two_minutes'] == 1) |
                (df['overtime_shot'] == 1)
            ),
            'Expanded High Danger': (
                (df['distance_to_net'] <= 35) |
                (df['in_slot'] == 1) |
                (df['in_crease'] == 1) |
                (df['is_tip_in'] == 1) |
                (df['potential_rebound'] == 1) |
                (df['final_two_minutes'] == 1) |
                (df['overtime_shot'] == 1) |
                ((df['is_forward'] == 1) & (df['distance_to_net'] <= 25))
            ),
            'Ultra Conservative': df['distance_to_net'] <= 60,
            'Minimal Filter': df['distance_to_net'] <= 100
        }
        
        prefilter_results = {}
        
        for strategy_name, condition in strategies.items():
            filtered_shots = df[condition]
            shots_kept = len(filtered_shots)
            goals_kept = filtered_shots['is_goal'].sum()
            
            volume_reduction = (1 - shots_kept / total_shots) * 100
            goal_retention = (goals_kept / total_goals) * 100
            miss_rate = (1 - goal_retention / 100) * 100
            
            prefilter_results[strategy_name] = {
                'shots_kept': shots_kept,
                'goals_kept': goals_kept,
                'volume_reduction': volume_reduction,
                'goal_retention': goal_retention,
                'miss_rate': miss_rate,
                'alpha_compliant': miss_rate <= 25.0
            }
            
            status = "✅" if miss_rate <= 25.0 else "❌"
            print(f"{status} {strategy_name}:")
            print(f"   Volume reduction: {volume_reduction:.1f}%")
            print(f"   Goal retention: {goal_retention:.1f}%")
            print(f"   Miss rate: {miss_rate:.1f}%")
        
        self.prefilter_results = prefilter_results
        return prefilter_results
    
    def test_prefilter_model_combinations(self):
        """Test combinations of pre-filtering + model."""
        if not self.results or not self.prefilter_results:
            raise ValueError("Must run model training and pre-filtering analysis first")
            
        print(f"\nPRE-FILTER + MODEL COMBINATIONS")
        print("="*50)
        
        # Use best performing model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_model_result = self.results[best_model_name]
        
        combination_results = {
            'Model Only': {
                'review_rate': best_model_result['review_rate'] * 100,
                'detection_rate': best_model_result['detection_rate'] * 100,
                'miss_rate': best_model_result['miss_rate'] * 100
            }
        }
        
        # Test each pre-filter strategy
        for strategy_name, prefilter_data in self.prefilter_results.items():
            if prefilter_data['alpha_compliant']:  # Only test α ≤ 25% compliant strategies
                # Simulate combined effect
                base_review_rate = best_model_result['review_rate']
                volume_reduction = prefilter_data['volume_reduction'] / 100
                
                # Estimate combined review rate (conservative estimate)
                combined_review_rate = base_review_rate * (1 - volume_reduction * 0.7)
                combined_detection_rate = best_model_result['detection_rate'] * (prefilter_data['goal_retention'] / 100)
                combined_miss_rate = 1 - combined_detection_rate
                
                combination_results[f'PreFilter + Model ({strategy_name})'] = {
                    'review_rate': combined_review_rate * 100,
                    'detection_rate': combined_detection_rate * 100,
                    'miss_rate': combined_miss_rate * 100
                }
                
                print(f"✅ {strategy_name} + {best_model_name}:")
                print(f"   Review rate: {combined_review_rate*100:.1f}%")
                print(f"   Detection rate: {combined_detection_rate*100:.1f}%")
                print(f"   Miss rate: {combined_miss_rate*100:.1f}%")
        
        return combination_results
    
    def create_business_visualization(self, save_path='nhl_business_analysis.png'):
        """Create comprehensive business analysis visualization."""
        print(f"\nCREATING BUSINESS VISUALIZATION")
        print("="*50)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Constraint Compliance Analysis
        if self.constraint_results:
            models = list(self.constraint_results.keys())
            f1_scores = [self.constraint_results[m]['f1_score'] for m in models]
            miss_rates = [self.constraint_results[m]['miss_rate'] * 100 for m in models]
            review_rates = [self.constraint_results[m]['review_rate'] * 100 for m in models]
            
            colors = ['green'] * len(models)  # All compliant if in constraint_results
            
            scatter = ax1.scatter(review_rates, miss_rates, s=200, c=colors, 
                                 alpha=0.7, edgecolors='black', linewidth=2)
            
            ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, label='α ≤ 25%')
            ax1.axvline(x=40, color='blue', linestyle='--', linewidth=2, label='β ≤ 40%')
            ax1.fill_between([0, 40], [0, 0], [25, 25], alpha=0.2, color='green', label='Target Region')
            
            ax1.set_title('Dual-Constraint Compliant Models', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Review Rate β (%)')
            ax1.set_ylabel('Miss Rate α (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            for i, model in enumerate(models):
                ax1.annotate(model, (review_rates[i], miss_rates[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 2. Pre-filtering Strategy Comparison
        if self.prefilter_results:
            strategies = list(self.prefilter_results.keys())
            volume_reductions = [self.prefilter_results[s]['volume_reduction'] for s in strategies]
            goal_retentions = [self.prefilter_results[s]['goal_retention'] for s in strategies]
            alpha_compliant = [self.prefilter_results[s]['alpha_compliant'] for s in strategies]
            
            colors = ['green' if compliant else 'red' for compliant in alpha_compliant]
            
            scatter2 = ax2.scatter(volume_reductions, goal_retentions, s=200, c=colors,
                                  alpha=0.7, edgecolors='black', linewidth=2)
            
            ax2.axhline(y=75, color='green', linestyle='--', alpha=0.7, label='75% Goal Retention')
            ax2.set_title('Pre-filtering Strategies (Green = α ≤ 25%)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Volume Reduction (%)')
            ax2.set_ylabel('Goal Retention (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            for i, strategy in enumerate(strategies):
                ax2.annotate(strategy, (volume_reductions[i], goal_retentions[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. F1 Score Optimization
        if self.constraint_results:
            bars = ax3.bar(range(len(models)), f1_scores, color=colors, alpha=0.7)
            ax3.set_title('F1 Score: Dual-Constraint Optimized Models', fontsize=14, fontweight='bold')
            ax3.set_ylabel('F1 Score')
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels(models, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            for bar, f1 in zip(bars, f1_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Business Impact Summary
        if hasattr(self, 'combination_results'):
            combo_strategies = list(self.combination_results.keys())
            combo_review_rates = [self.combination_results[s]['review_rate'] for s in combo_strategies]
            combo_detection_rates = [self.combination_results[s]['detection_rate'] for s in combo_strategies]
            
            colors_combo = ['red' if 'Model Only' in s else 'green' for s in combo_strategies]
            
            ax4.scatter(combo_review_rates, combo_detection_rates, s=200, c=colors_combo,
                       alpha=0.7, edgecolors='black', linewidth=2)
            
            ax4.set_title('Pre-filter + Model Performance', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Review Rate (%)')
            ax4.set_ylabel('Detection Rate (%)')
            ax4.grid(True, alpha=0.3)
            
            for i, strategy in enumerate(combo_strategies):
                short_name = strategy.replace('PreFilter + Model (', '').replace(')', '').replace('Model Only', 'Baseline')
                ax4.annotate(short_name, (combo_review_rates[i], combo_detection_rates[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.suptitle('NHL xG: Business Analysis & Constraint Optimization', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations."""
        print(f"\nBUSINESS RECOMMENDATIONS")
        print("="*60)
        
        # Constraint analysis recommendations
        if self.constraint_results:
            best_model = max(self.constraint_results.items(), key=lambda x: x[1]['f1_score'])
            print(f"🏆 RECOMMENDED MODEL: {best_model[0]}")
            print(f"   F1 Score: {best_model[1]['f1_score']:.3f}")
            print(f"   Miss Rate: {best_model[1]['miss_rate']:.1%}")
            print(f"   Review Rate: {best_model[1]['review_rate']:.1%}")
            print(f"   Threshold: {best_model[1]['threshold']:.3f}")
        else:
            print("❌ NO MODELS MEET DUAL CONSTRAINTS")
            print("   RECOMMENDATION: Relax constraints to α ≤ 35%, β ≤ 50%")
        
        # Pre-filtering recommendations
        if self.prefilter_results:
            compliant_strategies = {k: v for k, v in self.prefilter_results.items() if v['alpha_compliant']}
            if compliant_strategies:
                best_prefilter = max(compliant_strategies.items(), key=lambda x: x[1]['volume_reduction'])
                print(f"\n🎯 RECOMMENDED PRE-FILTER: {best_prefilter[0]}")
                print(f"   Volume reduction: {best_prefilter[1]['volume_reduction']:.1f}%")
                print(f"   Goal retention: {best_prefilter[1]['goal_retention']:.1f}%")
                print(f"   Miss rate: {best_prefilter[1]['miss_rate']:.1f}%")
        
        # Implementation strategy
        print(f"\n🚀 IMPLEMENTATION STRATEGY:")
        print(f"   1. Deploy recommended model with optimized threshold")
        print(f"   2. Implement pre-filtering for efficiency gains")
        print(f"   3. Monitor actual α and β rates in production")
        print(f"   4. Use F1 score as primary optimization metric")
        print(f"   5. Consider progressive constraint tightening")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • F1 score balances detection and efficiency optimally")
        print(f"   • Pre-filtering can reduce review burden while maintaining performance")
        print(f"   • Dual constraints provide clear business success criteria")
        print(f"   • Progressive implementation allows for continuous improvement")

def main():
    """Main business analysis pipeline."""
    print("🏒 NHL xG BUSINESS ANALYSIS")
    print("="*60)
    
    # Initialize business analyzer
    analyzer = NHLBusinessAnalyzer()
    
    # Load data and train models
    analyzer.load_shot_data()
    analyzer.engineer_features()
    analyzer.train_models()
    
    # Business constraint analysis
    analyzer.analyze_dual_constraints()
    analyzer.analyze_constraint_sensitivity()
    
    # Pre-filtering analysis
    analyzer.analyze_prefiltering_strategies()
    analyzer.combination_results = analyzer.test_prefilter_model_combinations()
    
    # Create visualization
    analyzer.create_business_visualization()
    
    # Generate recommendations
    analyzer.generate_business_recommendations()
    
    print(f"\n{'='*60}")
    print("BUSINESS ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 