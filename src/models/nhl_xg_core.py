"""
NHL xG Core Module
Consolidated data loading, feature engineering, and modeling functionality
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

class NHLxGAnalyzer:
    """Core NHL xG analysis class with all essential functionality."""
    
    def __init__(self, db_path='nhl_stats.db'):
        self.db_path = db_path
        self.shot_events = None
        self.models = {}
        self.results = {}
        
    def load_shot_data(self):
        """Load and prepare shot event data."""
        print("LOADING NHL SHOT DATA")
        print("="*50)
        
        conn = sqlite3.connect(self.db_path)
        
        # Load shot events
        query = """
        SELECT 
            e.gamePk,
            e.eventType,
            e.period,
            e.periodTime,
            e.teamId,
            e.x,
            e.y,
            e.details,
            g.gameDate
        FROM events e
        JOIN games g ON e.gamePk = g.gamePk
        WHERE e.eventType IN ('goal', 'shot-on-goal')
        AND e.x IS NOT NULL 
        AND e.y IS NOT NULL
        AND e.details IS NOT NULL
        ORDER BY g.gameDate, e.gamePk, e.eventIdx
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Load player positions
        players_query = """
        SELECT playerId, position, shootsCatches
        FROM players
        WHERE position IS NOT NULL
        """
        players_df = pd.read_sql_query(players_query, conn)
        conn.close()
        
        # Process shot data
        shot_data = []
        for _, row in df.iterrows():
            try:
                details = json.loads(row['details'])
                shot_info = {
                    'gamePk': row['gamePk'],
                    'eventType': row['eventType'],
                    'period': row['period'],
                    'periodTime': row['periodTime'],
                    'teamId': row['teamId'],
                    'x': row['x'],
                    'y': row['y'],
                    'gameDate': row['gameDate']
                }
                
                # Extract shooter info
                if 'details' in details:
                    inner_details = details['details']
                    if row['eventType'] == 'goal':
                        shot_info['shooterId'] = inner_details.get('scoringPlayerId')
                        shot_info['shotType'] = inner_details.get('shotType', 'Unknown')
                    elif row['eventType'] == 'shot-on-goal':
                        shot_info['shooterId'] = inner_details.get('shootingPlayerId')
                        shot_info['shotType'] = inner_details.get('shotType', 'Unknown')
                
                shot_data.append(shot_info)
            except:
                continue
        
        shot_events = pd.DataFrame(shot_data)
        shot_events = shot_events.dropna(subset=['x', 'y'])
        
        # Merge with player positions
        shot_events = shot_events.merge(
            players_df.rename(columns={'playerId': 'shooterId'}),
            on='shooterId',
            how='left'
        )
        
        print(f"Loaded {len(shot_events):,} shot events")
        print(f"Goals: {(shot_events['eventType'] == 'goal').sum():,}")
        print(f"Shots on goal: {(shot_events['eventType'] == 'shot-on-goal').sum():,}")
        
        self.shot_events = shot_events
        return shot_events
    
    def engineer_features(self):
        """Engineer all features for modeling."""
        if self.shot_events is None:
            raise ValueError("Must load shot data first")
            
        print("\nENGINEERING FEATURES")
        print("="*50)
        
        df = self.shot_events.copy()
        
        # Target variable
        df['is_goal'] = (df['eventType'] == 'goal').astype(int)
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        
        # Basic geometric features
        df['distance_to_net'] = np.minimum(
            np.sqrt((df['x'] - 89)**2 + df['y']**2),
            np.sqrt((df['x'] + 89)**2 + df['y']**2)
        )
        df['angle_to_net'] = np.abs(np.arctan2(np.abs(df['y']), 
                                               np.abs(np.abs(df['x']) - 89)) * 180 / np.pi)
        
        # Time features
        df['period_minutes'] = df['periodTime'].str.split(':').str[0].astype(float)
        df['period_seconds'] = df['periodTime'].str.split(':').str[1].astype(float)
        df['total_seconds'] = (df['period'] - 1) * 1200 + df['period_minutes'] * 60 + df['period_seconds']
        
        # Zone features
        df['in_crease'] = (df['distance_to_net'] <= 6).astype(int)
        df['in_slot'] = ((df['distance_to_net'] <= 20) & (df['angle_to_net'] <= 45)).astype(int)
        df['from_point'] = (df['distance_to_net'] >= 50).astype(int)
        
        # Shot type features
        df['is_wrist_shot'] = (df['shotType'] == 'Wrist').astype(int)
        df['is_slap_shot'] = (df['shotType'] == 'Slap').astype(int)
        df['is_snap_shot'] = (df['shotType'] == 'Snap').astype(int)
        df['is_backhand'] = (df['shotType'] == 'Backhand').astype(int)
        df['is_tip_in'] = (df['shotType'] == 'Tip-In').astype(int)
        
        # Position features
        df['is_forward'] = df['position'].isin(['C', 'LW', 'RW']).astype(int)
        df['is_defenseman'] = (df['position'] == 'D').astype(int)
        
        # Time-based features (streaming-safe)
        df = df.sort_values(['gamePk', 'total_seconds'])
        df['time_since_last_shot_same_team'] = df.groupby(['gamePk', 'teamId'])['total_seconds'].diff()
        df['potential_rebound'] = (
            (df['time_since_last_shot_same_team'] <= 5) & 
            (df['time_since_last_shot_same_team'] > 0)
        ).astype(int)
        
        # Pressure situations
        period_length = 1200
        df['time_remaining_period'] = period_length - (df['period_minutes'] * 60 + df['period_seconds'])
        df['final_two_minutes'] = (
            (df['period'] == 3) & 
            (df['time_remaining_period'] <= 120)
        ).astype(int)
        df['overtime_shot'] = (df['period'] > 3).astype(int)
        
        df = df.fillna(0)
        
        print(f"Engineered {len([c for c in df.columns if c not in ['gamePk', 'eventType', 'teamId', 'x', 'y', 'gameDate', 'shooterId', 'shotType', 'position', 'shootsCatches', 'periodTime']])} features")
        
        self.shot_events = df
        return df
    
    def get_feature_sets(self):
        """Define different feature sets for model comparison."""
        return {
            'Basic': ['distance_to_net', 'angle_to_net', 'period', 'total_seconds'],
            'Zone Enhanced': ['distance_to_net', 'angle_to_net', 'period', 'total_seconds',
                             'in_crease', 'in_slot', 'from_point'],
            'Shot Type Enhanced': ['distance_to_net', 'angle_to_net', 'period', 'total_seconds',
                                  'in_crease', 'in_slot', 'from_point',
                                  'is_wrist_shot', 'is_slap_shot', 'is_snap_shot', 'is_backhand', 'is_tip_in'],
            'Position Enhanced': ['distance_to_net', 'angle_to_net', 'period', 'total_seconds',
                                 'in_crease', 'in_slot', 'from_point',
                                 'is_wrist_shot', 'is_slap_shot', 'is_snap_shot', 'is_backhand', 'is_tip_in',
                                 'is_forward', 'is_defenseman'],
            'Time Enhanced': ['distance_to_net', 'angle_to_net', 'period', 'total_seconds',
                             'in_crease', 'in_slot', 'from_point',
                             'is_wrist_shot', 'is_slap_shot', 'is_snap_shot', 'is_backhand', 'is_tip_in',
                             'is_forward', 'is_defenseman',
                             'potential_rebound', 'final_two_minutes', 'overtime_shot', 'time_remaining_period']
        }
    
    def train_models(self, feature_sets=None):
        """Train models with different feature sets."""
        if self.shot_events is None:
            raise ValueError("Must load and engineer features first")
            
        print("\nTRAINING MODELS")
        print("="*50)
        
        if feature_sets is None:
            feature_sets = self.get_feature_sets()
        
        # Prepare data with temporal split
        df = self.shot_events.copy()
        dates = df['gameDate']
        date_order = dates.argsort()
        df_sorted = df.iloc[date_order]
        
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        print(f"Training set: {len(train_df):,} shots, {train_df['is_goal'].sum():,} goals")
        print(f"Test set: {len(test_df):,} shots, {test_df['is_goal'].sum():,} goals")
        
        results = {}
        
        for model_name, features in feature_sets.items():
            print(f"\nTraining {model_name} ({len(features)} features)...")
            
            # Prepare features
            X_train = train_df[features].fillna(0).values
            X_test = test_df[features].fillna(0).values
            y_train = train_df['is_goal'].values
            y_test = test_df['is_goal'].values
            
            # Train Random Forest with class balancing
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight={0: 1, 1: 8},  # Balance for 10% goal rate
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Find optimal threshold for F1 score
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Business metrics at optimal threshold
            y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
            
            true_goals = np.sum(y_test)
            detected_goals = np.sum(y_test[y_pred_binary == 1])
            total_flagged = np.sum(y_pred_binary)
            
            detection_rate = detected_goals / true_goals if true_goals > 0 else 0
            precision_rate = detected_goals / total_flagged if total_flagged > 0 else 0
            review_rate = total_flagged / len(y_test)
            miss_rate = 1 - detection_rate
            f1_score_val = f1_scores[optimal_idx]
            
            results[model_name] = {
                'model': model,
                'features': features,
                'auc': auc,
                'avg_precision': avg_precision,
                'optimal_threshold': optimal_threshold,
                'detection_rate': detection_rate,
                'precision': precision_rate,
                'review_rate': review_rate,
                'miss_rate': miss_rate,
                'f1_score': f1_score_val,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  AUC: {auc:.3f}")
            print(f"  Detection Rate: {detection_rate:.1%}")
            print(f"  Miss Rate: {miss_rate:.1%}")
            print(f"  Review Rate: {review_rate:.1%}")
            print(f"  F1 Score: {f1_score_val:.3f}")
        
        self.results = results
        return results
    
    def analyze_business_constraints(self, alpha_max=0.25, beta_max=0.40):
        """Analyze models against business constraints."""
        if not self.results:
            raise ValueError("Must train models first")
            
        print(f"\nBUSINESS CONSTRAINT ANALYSIS")
        print("="*50)
        print(f"Constraints: α ≤ {alpha_max:.1%} (miss rate), β ≤ {beta_max:.1%} (review rate)")
        
        constraint_results = {}
        
        for model_name, result in self.results.items():
            alpha_constraint = result['miss_rate'] <= alpha_max
            beta_constraint = result['review_rate'] <= beta_max
            
            constraint_results[model_name] = {
                'alpha_compliant': alpha_constraint,
                'beta_compliant': beta_constraint,
                'dual_compliant': alpha_constraint and beta_constraint,
                'miss_rate': result['miss_rate'],
                'review_rate': result['review_rate'],
                'f1_score': result['f1_score'],
                'detection_rate': result['detection_rate']
            }
            
            status = "✅" if alpha_constraint and beta_constraint else "❌"
            print(f"{status} {model_name}:")
            print(f"   α = {result['miss_rate']:.1%} ({'✅' if alpha_constraint else '❌'})")
            print(f"   β = {result['review_rate']:.1%} ({'✅' if beta_constraint else '❌'})")
            print(f"   F1 = {result['f1_score']:.3f}")
        
        # Find best compliant model
        compliant_models = {k: v for k, v in constraint_results.items() if v['dual_compliant']}
        
        if compliant_models:
            best_model = max(compliant_models.items(), key=lambda x: x[1]['f1_score'])
            print(f"\n🏆 BEST COMPLIANT MODEL: {best_model[0]}")
            print(f"   F1 Score: {best_model[1]['f1_score']:.3f}")
        else:
            print(f"\n❌ NO MODELS MEET DUAL CONSTRAINTS")
            print(f"   Consider relaxing constraints or improving models")
        
        return constraint_results
    
    def create_comprehensive_visualization(self, save_path='nhl_xg_analysis.png'):
        """Create comprehensive visualization of results."""
        if not self.results:
            raise ValueError("Must train models first")
            
        print(f"\nCREATING VISUALIZATION")
        print("="*50)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model Performance Comparison
        models = list(self.results.keys())
        aucs = [self.results[m]['auc'] for m in models]
        f1s = [self.results[m]['f1_score'] for m in models]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, aucs, width, label='AUC', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, f1s, width, label='F1 Score', alpha=0.7)
        
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Business Constraints Analysis
        miss_rates = [self.results[m]['miss_rate'] * 100 for m in models]
        review_rates = [self.results[m]['review_rate'] * 100 for m in models]
        
        colors = ['green' if mr <= 25 and rr <= 40 else 'red' 
                 for mr, rr in zip(miss_rates, review_rates)]
        
        scatter = ax2.scatter(review_rates, miss_rates, s=200, c=colors, 
                             alpha=0.7, edgecolors='black', linewidth=2)
        
        ax2.axhline(y=25, color='red', linestyle='--', linewidth=2, label='α ≤ 25%')
        ax2.axvline(x=40, color='blue', linestyle='--', linewidth=2, label='β ≤ 40%')
        ax2.fill_between([0, 40], [0, 0], [25, 25], alpha=0.2, color='green', label='Target Region')
        
        ax2.set_title('Business Constraints Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Review Rate β (%)')
        ax2.set_ylabel('Miss Rate α (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for i, model in enumerate(models):
            ax2.annotate(model, (review_rates[i], miss_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 3. Feature Count vs Performance
        feature_counts = [len(self.results[m]['features']) for m in models]
        
        ax3.scatter(feature_counts, aucs, s=150, alpha=0.7, color='blue', label='AUC')
        ax3.scatter(feature_counts, f1s, s=150, alpha=0.7, color='red', label='F1 Score')
        
        ax3.set_title('Feature Count vs Performance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Performance Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Detection vs Review Trade-off
        detection_rates = [self.results[m]['detection_rate'] * 100 for m in models]
        
        ax4.scatter(review_rates, detection_rates, s=200, c=f1s, cmap='viridis',
                   alpha=0.7, edgecolors='black', linewidth=2)
        
        ax4.set_title('Detection vs Review Trade-off', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Review Rate (%)')
        ax4.set_ylabel('Detection Rate (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.colorbar(ax4.collections[0], ax=ax4, label='F1 Score')
        
        for i, model in enumerate(models):
            ax4.annotate(model, (review_rates[i], detection_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.suptitle('NHL xG Model Analysis: Comprehensive Results', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if not self.results:
            raise ValueError("Must train models first")
            
        print(f"\nNHL xG MODEL SUMMARY REPORT")
        print("="*60)
        
        # Dataset summary
        total_shots = len(self.shot_events)
        total_goals = self.shot_events['is_goal'].sum()
        goal_rate = total_goals / total_shots
        
        print(f"DATASET SUMMARY:")
        print(f"  Total shots: {total_shots:,}")
        print(f"  Total goals: {total_goals:,}")
        print(f"  Goal rate: {goal_rate:.1%}")
        
        # Model performance summary
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Features: {len(result['features'])}")
            print(f"  AUC: {result['auc']:.3f}")
            print(f"  F1 Score: {result['f1_score']:.3f}")
            print(f"  Detection Rate: {result['detection_rate']:.1%}")
            print(f"  Miss Rate: {result['miss_rate']:.1%}")
            print(f"  Review Rate: {result['review_rate']:.1%}")
        
        # Best model recommendation
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['auc'])
        best_f1_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\nRECOMMENDATIONS:")
        print(f"  Best AUC: {best_auc_model[0]} ({best_auc_model[1]['auc']:.3f})")
        print(f"  Best F1: {best_f1_model[0]} ({best_f1_model[1]['f1_score']:.3f})")
        
        # Business deployment recommendation
        alpha_max, beta_max = 0.25, 0.40
        compliant_models = {k: v for k, v in self.results.items() 
                           if v['miss_rate'] <= alpha_max and v['review_rate'] <= beta_max}
        
        if compliant_models:
            best_compliant = max(compliant_models.items(), key=lambda x: x[1]['f1_score'])
            print(f"  Business Deployment: {best_compliant[0]} (meets α ≤ 25%, β ≤ 40%)")
        else:
            print(f"  Business Deployment: None meet dual constraints - consider relaxing")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")

def main():
    """Main analysis pipeline."""
    print("🏒 NHL xG COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = NHLxGAnalyzer()
    
    # Load and prepare data
    analyzer.load_shot_data()
    analyzer.engineer_features()
    
    # Train models
    analyzer.train_models()
    
    # Analyze business constraints
    analyzer.analyze_business_constraints()
    
    # Create visualization
    analyzer.create_comprehensive_visualization()
    
    # Generate summary report
    analyzer.generate_summary_report()

if __name__ == "__main__":
    main() 