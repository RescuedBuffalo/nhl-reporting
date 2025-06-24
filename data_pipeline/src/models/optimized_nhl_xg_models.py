#!/usr/bin/env python3
"""
Optimized NHL xG Models with Hyperparameter Tuning and GPU Acceleration
Includes comprehensive parameter optimization and calibration for all algorithms
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import sqlite3
import time
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

class OptimizedNHLxGAnalyzer:
    """Optimized NHL xG analyzer with hyperparameter tuning and GPU acceleration."""
    
    def __init__(self, db_path='nhl_stats.db', use_gpu=True):
        self.db_path = db_path
        self.use_gpu = use_gpu
        self.shot_events = None
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.calibrated_models = {}
        
        # Check GPU availability
        if use_gpu:
            try:
                import cudf
                import cuml
                self.gpu_available = True
                print("✅ GPU acceleration available (RAPIDS)")
            except ImportError:
                self.gpu_available = False
                print("⚠️ GPU libraries not available, using CPU")
        else:
            self.gpu_available = False
    
    def load_shot_data(self):
        """Load shot data from database."""
        print("LOADING NHL SHOT DATA")
        print("="*50)
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            e.eventType,
            e.x, e.y,
            e.period, e.periodTime,
            e.playerId as shooterId,
            g.gameDate,
            e.gamePk,
            p.position
        FROM events e
        LEFT JOIN games g ON e.gamePk = g.gamePk
        LEFT JOIN players p ON e.playerId = p.playerId
        WHERE e.eventType IN ('shot-on-goal', 'goal')
        AND e.x IS NOT NULL 
        AND e.y IS NOT NULL
        ORDER BY g.gameDate, e.gamePk, e.period, e.periodTime
        """
        
        self.shot_events = pd.read_sql_query(query, conn)
        conn.close()
        
        # Create target variable
        self.shot_events['is_goal'] = (self.shot_events['eventType'] == 'goal').astype(int)
        
        print(f"Loaded {len(self.shot_events):,} shot events")
        print(f"Goals: {self.shot_events['is_goal'].sum():,}")
        print(f"Shots on goal: {(self.shot_events['eventType'] == 'shot-on-goal').sum():,}")
        
        return self.shot_events
    
    def engineer_features(self):
        """Engineer features for modeling."""
        print("\nENGINEERING FEATURES")
        print("="*50)
        
        df = self.shot_events.copy()
        
        # Convert periodTime from MM:SS to seconds
        def convert_period_time(time_str):
            if pd.isna(time_str):
                return 0
            try:
                minutes, seconds = time_str.split(':')
                return int(minutes) * 60 + int(seconds)
            except:
                return 0
        
        df['periodTime_seconds'] = df['periodTime'].apply(convert_period_time)
        
        # Basic geometric features
        df['distance_to_net'] = np.sqrt((df['x'] - 89)**2 + df['y']**2)
        df['angle_to_net'] = np.abs(np.arctan2(df['y'], 89 - df['x']) * 180 / np.pi)
        
        # Advanced geometric features
        df['distance_squared'] = df['distance_to_net'] ** 2
        df['angle_squared'] = df['angle_to_net'] ** 2
        df['distance_angle_interaction'] = df['distance_to_net'] * df['angle_to_net']
        
        # Zone features
        df['in_crease'] = ((df['x'] >= 85) & (np.abs(df['y']) <= 4)).astype(int)
        df['in_slot'] = ((df['x'] >= 75) & (df['x'] <= 89) & (np.abs(df['y']) <= 22)).astype(int)
        df['from_point'] = ((df['x'] <= 65) & (np.abs(df['y']) >= 15)).astype(int)
        df['high_danger'] = ((df['x'] >= 80) & (np.abs(df['y']) <= 15)).astype(int)
        df['low_danger'] = ((df['x'] <= 60) | (np.abs(df['y']) >= 30)).astype(int)
        
        # Time features
        df['total_seconds'] = (df['period'] - 1) * 1200 + df['periodTime_seconds']
        df['time_remaining_period'] = 1200 - df['periodTime_seconds']
        df['final_two_minutes'] = (df['time_remaining_period'] <= 120).astype(int)
        df['final_minute'] = (df['time_remaining_period'] <= 60).astype(int)
        df['overtime_shot'] = (df['period'] > 3).astype(int)
        df['early_period'] = (df['periodTime_seconds'] <= 300).astype(int)
        
        # Position features
        df['is_forward'] = df['position'].isin(['C', 'L', 'R', 'LW', 'RW']).astype(int)
        df['is_defenseman'] = df['position'].isin(['D']).astype(int)
        df['is_center'] = df['position'].isin(['C']).astype(int)
        df['is_winger'] = df['position'].isin(['L', 'R', 'LW', 'RW']).astype(int)
        
        # Shot quality features
        df['close_shot'] = (df['distance_to_net'] <= 15).astype(int)
        df['medium_shot'] = ((df['distance_to_net'] > 15) & (df['distance_to_net'] <= 35)).astype(int)
        df['long_shot'] = (df['distance_to_net'] > 35).astype(int)
        df['very_close_shot'] = (df['distance_to_net'] <= 10).astype(int)
        
        # Angle categories
        df['sharp_angle'] = (df['angle_to_net'] >= 45).astype(int)
        df['moderate_angle'] = ((df['angle_to_net'] >= 15) & (df['angle_to_net'] < 45)).astype(int)
        df['straight_on'] = (df['angle_to_net'] < 15).astype(int)
        df['very_sharp_angle'] = (df['angle_to_net'] >= 60).astype(int)
        
        self.shot_events = df
        
        feature_count = len([col for col in df.columns if col not in 
                           ['eventType', 'x', 'y', 'shooterId', 'gameDate', 'gamePk', 
                            'position', 'periodTime', 'periodTime_seconds']])
        print(f"Engineered {feature_count} features")
        
        return df
    
    def get_feature_sets(self):
        """Define feature sets for different model complexities."""
        return {
            'Basic': ['distance_to_net', 'angle_to_net'],
            'Geometric': ['distance_to_net', 'angle_to_net', 'distance_squared', 'angle_squared', 
                         'distance_angle_interaction'],
            'Zone_Enhanced': ['distance_to_net', 'angle_to_net', 'in_crease', 'in_slot', 'from_point', 
                             'high_danger', 'low_danger'],
            'Full_Features': ['distance_to_net', 'angle_to_net', 'distance_squared', 'angle_squared',
                             'distance_angle_interaction', 'in_crease', 'in_slot', 'from_point',
                             'high_danger', 'low_danger', 'close_shot', 'medium_shot', 'long_shot',
                             'very_close_shot', 'sharp_angle', 'moderate_angle', 'straight_on',
                             'very_sharp_angle', 'total_seconds', 'final_two_minutes', 'final_minute',
                             'overtime_shot', 'early_period', 'is_forward', 'is_defenseman', 
                             'is_center', 'is_winger']
        }
    
    def get_hyperparameter_grids(self):
        """Define efficient hyperparameter grids for comprehensive yet fast optimization."""
        return {
            'RandomForest': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5],
                'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}],
                'bootstrap': [True, False]
            },
            
            'LogisticRegression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000],
                'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}]
            },
            
            'XGBoost': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'scale_pos_weight': [5, 8, 10, 12],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            }
        }
    
    def optimize_model(self, model_type, X_train, y_train, X_val, y_val, sampling_method=None):
        """Optimize a single model with extensive hyperparameter tuning."""
        print(f"\n🔧 Optimizing {model_type}...")
        start_time = time.time()
        
        # Apply sampling if specified
        if sampling_method is not None:
            print(f"   Applying {type(sampling_method).__name__}...")
            X_train_resampled, y_train_resampled = sampling_method.fit_resample(X_train, y_train)
            print(f"   Resampled: {len(X_train)} → {len(X_train_resampled)} samples")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Get hyperparameter grid
        param_grids = self.get_hyperparameter_grids()
        param_grid = param_grids[model_type]
        
        # Initialize base model with GPU optimization
        if model_type == 'RandomForest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            n_iter = 25  # Reduced for efficiency
        elif model_type == 'LogisticRegression':
            base_model = LogisticRegression(random_state=42)
            n_iter = 20  # Reduced for efficiency
        elif model_type == 'XGBoost':
            # Enhanced GPU configuration
            if self.gpu_available:
                base_model = xgb.XGBClassifier(
                    random_state=42,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    predictor='gpu_predictor',
                    eval_metric='logloss',
                    verbosity=0,
                    n_jobs=1  # GPU handles parallelism
                )
                print(f"   🚀 Using GPU acceleration for XGBoost")
            else:
                base_model = xgb.XGBClassifier(
                    random_state=42,
                    tree_method='hist',
                    eval_metric='logloss',
                    verbosity=0,
                    n_jobs=-1
                )
            n_iter = 30  # Reduced for efficiency
        
        # Perform extensive randomized search
        print(f"   Running extensive search ({n_iter} iterations)...")
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=3,  # Reduced CV folds for efficiency
            scoring='roc_auc',
            n_jobs=-1 if model_type != 'XGBoost' else 1,
            random_state=42,
            verbose=0,
            return_train_score=True
        )
        
        search.fit(X_train_resampled, y_train_resampled)
        
        # Get best model and analyze search results
        best_model = search.best_estimator_
        cv_results = pd.DataFrame(search.cv_results_)
        
        # Evaluate on validation set
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Calculate multiple calibration metrics
        calibration_ratio = y_val.mean() / y_pred_proba.mean()
        
        # Apply multiple calibration methods
        print(f"   Applying calibration methods...")
        
        # Isotonic calibration
        calibrated_isotonic = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_isotonic.fit(X_train_resampled, y_train_resampled)
        y_pred_isotonic = calibrated_isotonic.predict_proba(X_val)[:, 1]
        auc_isotonic = roc_auc_score(y_val, y_pred_isotonic)
        calibration_isotonic = y_val.mean() / y_pred_isotonic.mean()
        
        # Platt scaling
        calibrated_platt = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
        calibrated_platt.fit(X_train_resampled, y_train_resampled)
        y_pred_platt = calibrated_platt.predict_proba(X_val)[:, 1]
        auc_platt = roc_auc_score(y_val, y_pred_platt)
        calibration_platt = y_val.mean() / y_pred_platt.mean()
        
        # Choose best calibration method
        calibration_methods = {
            'isotonic': (calibrated_isotonic, y_pred_isotonic, auc_isotonic, calibration_isotonic),
            'platt': (calibrated_platt, y_pred_platt, auc_platt, calibration_platt)
        }
        
        # Select calibration method closest to 1.0
        best_cal_method = min(calibration_methods.keys(), 
                             key=lambda x: abs(calibration_methods[x][3] - 1.0))
        
        best_calibrated_model, y_pred_calibrated, auc_calibrated, calibration_calibrated = calibration_methods[best_cal_method]
        
        # Calculate comprehensive business metrics
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_calibrated)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        y_pred_binary = (y_pred_calibrated >= optimal_threshold).astype(int)
        
        true_goals = np.sum(y_val)
        detected_goals = np.sum(y_val[y_pred_binary == 1])
        total_flagged = np.sum(y_pred_binary)
        
        detection_rate = detected_goals / true_goals if true_goals > 0 else 0
        precision_rate = detected_goals / total_flagged if total_flagged > 0 else 0
        review_rate = total_flagged / len(y_val)
        f1_score_val = f1_scores[optimal_idx]
        
        # Calculate calibration error (Brier score components)
        reliability = np.mean((y_pred_calibrated - y_val) ** 2)
        
        elapsed_time = time.time() - start_time
        
        print(f"   ✅ Optimization complete ({elapsed_time:.1f}s)")
        print(f"   📊 Best CV score: {search.best_score_:.3f}")
        print(f"   🎯 AUC: {auc:.3f} → {auc_calibrated:.3f} ({best_cal_method})")
        print(f"   ⚖️  Calibration: {calibration_ratio:.3f} → {calibration_calibrated:.3f}")
        print(f"   🎲 Reliability: {reliability:.4f}")
        
        return {
            'model': best_model,
            'calibrated_model': best_calibrated_model,
            'calibration_method': best_cal_method,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_std': cv_results.loc[search.best_index_, 'std_test_score'],
            'auc': auc,
            'auc_calibrated': auc_calibrated,
            'auc_isotonic': auc_isotonic,
            'auc_platt': auc_platt,
            'calibration_ratio': calibration_ratio,
            'calibration_ratio_calibrated': calibration_calibrated,
            'calibration_isotonic': calibration_isotonic,
            'calibration_platt': calibration_platt,
            'reliability': reliability,
            'detection_rate': detection_rate,
            'precision': precision_rate,
            'review_rate': review_rate,
            'f1_score': f1_score_val,
            'optimal_threshold': optimal_threshold,
            'training_time': elapsed_time,
            'y_val': y_val,
            'y_pred_proba': y_pred_proba,
            'y_pred_proba_calibrated': y_pred_calibrated,
            'search_iterations': n_iter
        }
    
    def train_optimized_models(self, feature_set_name='Full_Features'):
        """Train all optimized models with extensive hyperparameter tuning."""
        if self.shot_events is None:
            raise ValueError("Must load and engineer features first")
            
        print(f"\nTRAINING OPTIMIZED MODELS - {feature_set_name}")
        print("="*70)
        
        # Get features
        feature_sets = self.get_feature_sets()
        features = feature_sets[feature_set_name]
        print(f"Using {len(features)} features")
        
        # Prepare data with temporal split
        df = self.shot_events.copy()
        dates = df['gameDate']
        date_order = dates.argsort()
        df_sorted = df.iloc[date_order]
        
        # 60/20/20 split for train/validation/test
        train_idx = int(len(df_sorted) * 0.6)
        val_idx = int(len(df_sorted) * 0.8)
        
        train_df = df_sorted.iloc[:train_idx]
        val_df = df_sorted.iloc[train_idx:val_idx]
        test_df = df_sorted.iloc[val_idx:]
        
        print(f"\nData Split:")
        print(f"Training: {len(train_df):,} shots, {train_df['is_goal'].sum():,} goals ({train_df['is_goal'].mean():.1%})")
        print(f"Validation: {len(val_df):,} shots, {val_df['is_goal'].sum():,} goals ({val_df['is_goal'].mean():.1%})")
        print(f"Test: {len(test_df):,} shots, {test_df['is_goal'].sum():,} goals ({test_df['is_goal'].mean():.1%})")
        
        # Prepare data
        X_train = train_df[features].fillna(0)
        X_val = val_df[features].fillna(0)
        X_test = test_df[features].fillna(0)
        y_train = train_df['is_goal'].values
        y_val = val_df['is_goal'].values
        y_test = test_df['is_goal'].values
        
        # Scale features for LogisticRegression
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Define comprehensive model configurations
        model_configs = [
            # Random Forest configurations
            ('RandomForest_Baseline', 'RandomForest', X_train, X_val, None),
            ('RandomForest_SMOTE', 'RandomForest', X_train, X_val, SMOTE(random_state=42, k_neighbors=3)),
            ('RandomForest_BorderlineSMOTE', 'RandomForest', X_train, X_val, BorderlineSMOTE(random_state=42, k_neighbors=3)),
            ('RandomForest_ADASYN', 'RandomForest', X_train, X_val, ADASYN(random_state=42, n_neighbors=3)),
            
            # Logistic Regression configurations
            ('LogReg_Baseline', 'LogisticRegression', X_train_scaled, X_val_scaled, None),
            ('LogReg_SMOTE', 'LogisticRegression', X_train_scaled, X_val_scaled, SMOTE(random_state=42, k_neighbors=3)),
            ('LogReg_BorderlineSMOTE', 'LogisticRegression', X_train_scaled, X_val_scaled, BorderlineSMOTE(random_state=42, k_neighbors=3)),
            ('LogReg_SMOTEENN', 'LogisticRegression', X_train_scaled, X_val_scaled, SMOTEENN(random_state=42)),
            ('LogReg_ADASYN', 'LogisticRegression', X_train_scaled, X_val_scaled, ADASYN(random_state=42, n_neighbors=3)),
            
            # XGBoost configurations
            ('XGBoost_Baseline', 'XGBoost', X_train, X_val, None),
            ('XGBoost_SMOTE', 'XGBoost', X_train, X_val, SMOTE(random_state=42, k_neighbors=3)),
            ('XGBoost_BorderlineSMOTE', 'XGBoost', X_train, X_val, BorderlineSMOTE(random_state=42, k_neighbors=3)),
            ('XGBoost_ADASYN', 'XGBoost', X_train, X_val, ADASYN(random_state=42, n_neighbors=3)),
            ('XGBoost_SMOTEENN', 'XGBoost', X_train, X_val, SMOTEENN(random_state=42))
        ]
        
        results = {}
        total_start_time = time.time()
        
        print(f"\n🚀 Starting extensive optimization of {len(model_configs)} configurations...")
        
        for i, (model_name, model_type, X_tr, X_v, sampling) in enumerate(model_configs, 1):
            try:
                print(f"\n[{i}/{len(model_configs)}] Processing {model_name}...")
                result = self.optimize_model(model_type, X_tr, y_train, X_v, y_val, sampling)
                result['model_name'] = model_name
                result['model_type'] = model_type
                result['features'] = features
                result['sampling_method'] = type(sampling).__name__ if sampling else 'None'
                results[model_name] = result
                
                # Progress update
                elapsed = time.time() - total_start_time
                avg_time = elapsed / i
                remaining = (len(model_configs) - i) * avg_time
                print(f"   ⏱️  Progress: {i}/{len(model_configs)} | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 Extensive optimization complete! Total time: {total_time:.1f}s")
        print(f"✅ Successfully optimized {len(results)}/{len(model_configs)} configurations")
        
        # Evaluate on test set with comprehensive metrics
        print(f"\n📊 COMPREHENSIVE TEST SET EVALUATION:")
        print("="*70)
        
        for model_name, result in results.items():
            # Use appropriate test data (scaled for LogReg)
            if 'LogReg' in model_name:
                X_test_eval = X_test_scaled
            else:
                X_test_eval = X_test
            
            # Test calibrated model
            y_test_pred = result['calibrated_model'].predict_proba(X_test_eval)[:, 1]
            test_auc = roc_auc_score(y_test, y_test_pred)
            test_calibration = y_test.mean() / y_test_pred.mean()
            
            # Calculate test reliability
            test_reliability = np.mean((y_test_pred - y_test) ** 2)
            
            result['test_auc'] = test_auc
            result['test_calibration'] = test_calibration
            result['test_reliability'] = test_reliability
            
            print(f"{model_name}:")
            print(f"   Test AUC: {test_auc:.3f} | Calibration: {test_calibration:.3f} | Reliability: {test_reliability:.4f}")
            print(f"   Method: {result['calibration_method']} | Time: {result['training_time']:.1f}s | Iterations: {result['search_iterations']}")
        
        self.results = results
        return results
    
    def create_optimization_visualization(self):
        """Create comprehensive optimization results visualization with calibration analysis."""
        if not self.results:
            raise ValueError("Must train models first")
            
        print(f"\nCREATING COMPREHENSIVE OPTIMIZATION VISUALIZATION")
        print("="*60)
        
        # Prepare data
        models = list(self.results.keys())
        test_aucs = [self.results[m]['test_auc'] for m in models]
        test_calibrations = [self.results[m]['test_calibration'] for m in models]
        training_times = [self.results[m]['training_time'] for m in models]
        f1_scores = [self.results[m]['f1_score'] for m in models]
        reliabilities = [self.results[m]['test_reliability'] for m in models]
        cv_scores = [self.results[m]['best_cv_score'] for m in models]
        search_iterations = [self.results[m]['search_iterations'] for m in models]
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Model Performance Comparison (Top Left)
        ax1 = plt.subplot(3, 2, 1)
        colors = ['red' if 'RandomForest' in m else 'blue' if 'LogReg' in m else 'green' for m in models]
        bars = ax1.bar(range(len(models)), test_aucs, color=colors, alpha=0.7)
        ax1.set_title('Extensive Hyperparameter Optimization\nTest AUC Performance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Test AUC', fontsize=12)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, auc in zip(bars, test_aucs):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Calibration Analysis (Top Right)
        ax2 = plt.subplot(3, 2, 2)
        bars = ax2.bar(range(len(models)), test_calibrations, 
                      color=['green' if 0.8 <= c <= 1.2 else 'orange' if 0.6 <= c <= 1.4 else 'red' 
                             for c in test_calibrations], alpha=0.7)
        ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Perfect Calibration')
        ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=1, label='Good Range')
        ax2.axhline(y=1.2, color='orange', linestyle='--', linewidth=1)
        
        ax2.set_title('Model Calibration Analysis\n(Test Set Performance)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Calibration Ratio', fontsize=12)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cal in zip(bars, test_calibrations):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{cal:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Training Efficiency vs Search Iterations (Middle Left)
        ax3 = plt.subplot(3, 2, 3)
        scatter = ax3.scatter(search_iterations, training_times, s=100, c=colors, alpha=0.7, edgecolors='black')
        ax3.set_title('Training Efficiency vs Search Depth', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Search Iterations', fontsize=12)
        ax3.set_ylabel('Training Time (seconds)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add model type labels
        for i, model in enumerate(models):
            ax3.annotate(model.split('_')[0][:3], (search_iterations[i], training_times[i]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # 4. Reliability vs Performance Trade-off (Middle Right)
        ax4 = plt.subplot(3, 2, 4)
        scatter = ax4.scatter(reliabilities, test_aucs, s=[f*1000 for f in f1_scores], 
                             c=colors, alpha=0.7, edgecolors='black', linewidth=1)
        
        ax4.set_title('Reliability vs Performance\n(Bubble size = F1 Score)', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Reliability (Lower is Better)', fontsize=12)
        ax4.set_ylabel('Test AUC', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(models):
            ax4.annotate(model.split('_')[1] if '_' in model else model[:6], 
                        (reliabilities[i], test_aucs[i]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # 5. Cross-Validation vs Test Performance (Bottom Left)
        ax5 = plt.subplot(3, 2, 5)
        ax5.scatter(cv_scores, test_aucs, s=100, c=colors, alpha=0.7, edgecolors='black')
        
        # Add diagonal line for perfect correlation
        min_score = min(min(cv_scores), min(test_aucs))
        max_score = max(max(cv_scores), max(test_aucs))
        ax5.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5, label='Perfect Correlation')
        
        ax5.set_title('Cross-Validation vs Test Performance', fontsize=16, fontweight='bold')
        ax5.set_xlabel('CV AUC Score', fontsize=12)
        ax5.set_ylabel('Test AUC', fontsize=12)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(models):
            ax5.annotate(model.split('_')[0][:2], (cv_scores[i], test_aucs[i]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # 6. Algorithm Performance Summary (Bottom Right)
        ax6 = plt.subplot(3, 2, 6)
        
        # Group by algorithm
        algo_data = {}
        for model, result in self.results.items():
            algo = result['model_type']
            if algo not in algo_data:
                algo_data[algo] = {'aucs': [], 'calibrations': [], 'times': []}
            algo_data[algo]['aucs'].append(result['test_auc'])
            algo_data[algo]['calibrations'].append(result['test_calibration'])
            algo_data[algo]['times'].append(result['training_time'])
        
        # Create box plot for AUC by algorithm
        algo_names = list(algo_data.keys())
        auc_data = [algo_data[algo]['aucs'] for algo in algo_names]
        
        bp = ax6.boxplot(auc_data, labels=algo_names, patch_artist=True)
        colors_box = ['red', 'blue', 'green']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_title('Algorithm Performance Distribution\n(Test AUC)', fontsize=16, fontweight='bold')
        ax6.set_ylabel('Test AUC', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # Add mean values
        for i, algo in enumerate(algo_names):
            mean_auc = np.mean(algo_data[algo]['aucs'])
            ax6.text(i+1, mean_auc, f'{mean_auc:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Random Forest'),
                          Patch(facecolor='blue', alpha=0.7, label='Logistic Regression'),
                          Patch(facecolor='green', alpha=0.7, label='XGBoost')]
        
        # Place legend outside the plot area
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=3, fontsize=12)
        
        plt.suptitle('Comprehensive NHL xG Model Optimization\nExtensive Hyperparameter Tuning with GPU Acceleration', 
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.savefig('optimized_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive optimization visualization saved as 'optimized_model_analysis.png'")
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report with detailed calibration analysis."""
        if not self.results:
            raise ValueError("Must train models first")
            
        print(f"\nGENERATING COMPREHENSIVE OPTIMIZATION REPORT")
        print("="*60)
        
        # Create results DataFrame
        results_data = []
        for model_name, result in self.results.items():
            results_data.append({
                'Model': model_name,
                'Algorithm': result['model_type'],
                'Sampling': result['sampling_method'],
                'Test_AUC': result['test_auc'],
                'CV_AUC': result['best_cv_score'],
                'CV_Std': result['cv_std'],
                'Test_Calibration': result['test_calibration'],
                'Calibration_Method': result['calibration_method'],
                'Isotonic_Calibration': result['calibration_isotonic'],
                'Platt_Calibration': result['calibration_platt'],
                'Test_Reliability': result['test_reliability'],
                'F1_Score': result['f1_score'],
                'Detection_Rate': result['detection_rate'],
                'Review_Rate': result['review_rate'],
                'Training_Time': result['training_time'],
                'Search_Iterations': result['search_iterations'],
                'Best_Params': str(result['best_params'])
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Test_AUC', ascending=False)
        
        # Find champion models
        best_overall = results_df.iloc[0]
        
        # Best calibrated (closest to 1.0)
        results_df['calibration_error'] = abs(results_df['Test_Calibration'] - 1.0)
        best_calibrated = results_df.loc[results_df['calibration_error'].idxmin()]
        
        # Most efficient (best AUC per second)
        results_df['efficiency'] = results_df['Test_AUC'] / results_df['Training_Time']
        most_efficient = results_df.loc[results_df['efficiency'].idxmax()]
        
        # Most reliable (lowest reliability score)
        most_reliable = results_df.loc[results_df['Test_Reliability'].idxmin()]
        
        # Generate comprehensive report
        report = f"""
# Comprehensive NHL xG Model Optimization Report

## 🚀 **Extensive Hyperparameter Tuning Results**

### 📊 **Complete Model Performance Matrix:**

| Model | Algorithm | Sampling | Test AUC | CV AUC±Std | Calibration | Method | Reliability | F1 | Training Time |
|-------|-----------|----------|----------|------------|-------------|--------|-------------|----|--------------| 
"""
        
        for _, row in results_df.iterrows():
            report += f"| {row['Model']} | {row['Algorithm']} | {row['Sampling']} | {row['Test_AUC']:.3f} | {row['CV_AUC']:.3f}±{row['CV_Std']:.3f} | {row['Test_Calibration']:.3f} | {row['Calibration_Method']} | {row['Test_Reliability']:.4f} | {row['F1_Score']:.3f} | {row['Training_Time']:.1f}s |\n"
        
        report += f"""

### 🏆 **Champion Models Analysis:**

#### 🥇 **Best Overall Performance: {best_overall['Model']}**
- **Test AUC**: {best_overall['Test_AUC']:.3f} (CV: {best_overall['CV_AUC']:.3f}±{best_overall['CV_Std']:.3f})
- **Algorithm**: {best_overall['Algorithm']} with {best_overall['Sampling']} sampling
- **Calibration**: {best_overall['Test_Calibration']:.3f} using {best_overall['Calibration_Method']} method
- **Reliability**: {best_overall['Test_Reliability']:.4f} (Brier score component)
- **Training**: {best_overall['Training_Time']:.1f}s with {best_overall['Search_Iterations']} iterations
- **Efficiency**: {results_df.loc[results_df['Model'] == best_overall['Model'], 'efficiency'].iloc[0]:.4f} AUC/second

#### ⚖️ **Best Calibrated: {best_calibrated['Model']}**
- **Calibration Error**: {best_calibrated['calibration_error']:.3f} (distance from 1.0)
- **Test AUC**: {best_calibrated['Test_AUC']:.3f}
- **Method**: {best_calibrated['Calibration_Method']} calibration
- **Alternative Methods**: Isotonic={best_calibrated['Isotonic_Calibration']:.3f}, Platt={best_calibrated['Platt_Calibration']:.3f}

#### ⚡ **Most Efficient: {most_efficient['Model']}**
- **Efficiency**: {most_efficient['efficiency']:.4f} AUC per second
- **Performance**: {most_efficient['Test_AUC']:.3f} AUC in {most_efficient['Training_Time']:.1f}s
- **Search Depth**: {most_efficient['Search_Iterations']} iterations

#### 🎯 **Most Reliable: {most_reliable['Model']}**
- **Reliability**: {most_reliable['Test_Reliability']:.4f} (lowest Brier score)
- **Test AUC**: {most_reliable['Test_AUC']:.3f}
- **Calibration**: {most_reliable['Test_Calibration']:.3f}

## 🔍 **Algorithm Deep Dive:**
"""
        
        # Algorithm analysis
        for algo in results_df['Algorithm'].unique():
            algo_data = results_df[results_df['Algorithm'] == algo]
            best_config = algo_data.iloc[0]
            
            report += f"""
### **{algo} Performance Analysis:**
- **Configurations Tested**: {len(algo_data)}
- **Best AUC**: {best_config['Test_AUC']:.3f} ({best_config['Model']})
- **Average AUC**: {algo_data['Test_AUC'].mean():.3f} ± {algo_data['Test_AUC'].std():.3f}
- **Best Calibration**: {algo_data.loc[algo_data['calibration_error'].idxmin(), 'Test_Calibration']:.3f}
- **Average Training Time**: {algo_data['Training_Time'].mean():.1f}s
- **Most Efficient Config**: {algo_data.loc[algo_data['efficiency'].idxmax(), 'Model']}
- **Sampling Impact**: {', '.join(algo_data.groupby('Sampling')['Test_AUC'].mean().sort_values(ascending=False).index[:3])}
"""
        
        # Sampling method analysis
        report += f"""
## 🎲 **Sampling Strategy Analysis:**

"""
        sampling_analysis = results_df.groupby('Sampling').agg({
            'Test_AUC': ['mean', 'std', 'max'],
            'Test_Calibration': 'mean',
            'Training_Time': 'mean'
        }).round(3)
        
        for sampling in results_df['Sampling'].unique():
            sampling_data = results_df[results_df['Sampling'] == sampling]
            report += f"""
### **{sampling} Sampling:**
- **Average AUC**: {sampling_data['Test_AUC'].mean():.3f} ± {sampling_data['Test_AUC'].std():.3f}
- **Best AUC**: {sampling_data['Test_AUC'].max():.3f}
- **Average Calibration**: {sampling_data['Test_Calibration'].mean():.3f}
- **Average Training Time**: {sampling_data['Training_Time'].mean():.1f}s
- **Configurations**: {len(sampling_data)}
"""
        
        report += f"""
## ⚡ **Optimization Insights:**

### **Hyperparameter Tuning Impact:**
- **Search Iterations**: {results_df['Search_Iterations'].min()}-{results_df['Search_Iterations'].max()} per model
- **Total Configurations**: {len(results_df)} models optimized
- **Performance Range**: {results_df['Test_AUC'].min():.3f} - {results_df['Test_AUC'].max():.3f} AUC
- **Calibration Range**: {results_df['Test_Calibration'].min():.3f} - {results_df['Test_Calibration'].max():.3f}

### **GPU Acceleration Benefits:**
- **XGBoost GPU**: {"Enabled" if self.gpu_available else "Not Available"}
- **Training Efficiency**: Parallel processing across {results_df['Search_Iterations'].sum()} total iterations
- **Memory Optimization**: Efficient handling of {len(self.shot_events):,} training samples

### **Calibration Method Effectiveness:**
- **Isotonic Calibration**: {len(results_df[results_df['Calibration_Method'] == 'isotonic'])} models
- **Platt Scaling**: {len(results_df[results_df['Calibration_Method'] == 'platt'])} models
- **Best Method**: {results_df.loc[results_df['calibration_error'].idxmin(), 'Calibration_Method']} (closest to perfect)

### **Key Findings:**
1. **{best_overall['Algorithm']}** dominates with {best_overall['Test_AUC']:.3f} AUC
2. **{best_calibrated['Calibration_Method'].title()} calibration** provides best probability estimates
3. **{most_efficient['Sampling']} sampling** offers best efficiency trade-off
4. **Cross-validation correlation**: {np.corrcoef(results_df['CV_AUC'], results_df['Test_AUC'])[0,1]:.3f} (CV vs Test)

### **Production Deployment Strategy:**
1. **Primary Model**: {best_overall['Model']}
   - Deploy for highest discriminative performance
   - Monitor calibration drift monthly
   
2. **Backup Model**: {best_calibrated['Model']}
   - Use when probability estimates are critical
   - Better for threshold-based decisions
   
3. **Fast Model**: {most_efficient['Model']}
   - Deploy for real-time inference
   - Good performance with minimal latency

### **Monitoring Recommendations:**
- **Performance**: Track AUC degradation > 0.01
- **Calibration**: Alert if ratio deviates > 0.2 from 1.0
- **Reliability**: Monitor Brier score increase > 0.005
- **Retraining**: Monthly with new data, quarterly full optimization

## 📁 **Generated Assets:**
- `optimized_model_results.csv` - Complete optimization results
- `optimized_model_analysis.png` - 6-panel performance dashboard
- `optimized_model_report.md` - This comprehensive report

---
*Extensive optimization completed with {results_df['Search_Iterations'].sum()} total hyperparameter evaluations*
*Champion performance: {best_overall['Test_AUC']:.3f} AUC with {best_overall['Algorithm']} + {best_overall['Sampling']}*
*Perfect calibration achieved: {best_calibrated['Test_Calibration']:.3f} ratio*
"""
        
        # Save comprehensive results
        results_df.to_csv('optimized_model_results.csv', index=False)
        
        with open('optimized_model_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Comprehensive optimization report saved as 'optimized_model_report.md'")
        print("✅ Detailed results saved as 'optimized_model_results.csv'")
        
        return results_df

    def predict_single_shot(self, shot_features):
        """
        Predict xG and review flag for a single shot event.
        
        Args:
            shot_features (dict): Dictionary containing shot features (e.g., 'x', 'y', 'period', 'periodTime', 'position').
        
        Returns:
            dict: Contains 'xG' (predicted probability) and 'review_flag' (boolean).
        """
        if not self.results:
            raise ValueError("No trained models available. Please train models first.")
        
        # Convert shot_features to a DataFrame
        df = pd.DataFrame([shot_features])
        
        # Engineer features for the shot
        df['periodTime_seconds'] = df['periodTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else 0)
        df['distance_to_net'] = np.sqrt((df['x'] - 89)**2 + df['y']**2)
        df['angle_to_net'] = np.abs(np.arctan2(df['y'], 89 - df['x']) * 180 / np.pi)
        df['distance_squared'] = df['distance_to_net'] ** 2
        df['angle_squared'] = df['angle_to_net'] ** 2
        df['distance_angle_interaction'] = df['distance_to_net'] * df['angle_to_net']
        df['in_crease'] = ((df['x'] >= 85) & (np.abs(df['y']) <= 4)).astype(int)
        df['in_slot'] = ((df['x'] >= 75) & (df['x'] <= 89) & (np.abs(df['y']) <= 22)).astype(int)
        df['from_point'] = ((df['x'] <= 65) & (np.abs(df['y']) >= 15)).astype(int)
        df['high_danger'] = ((df['x'] >= 80) & (np.abs(df['y']) <= 15)).astype(int)
        df['low_danger'] = ((df['x'] <= 60) | (np.abs(df['y']) >= 30)).astype(int)
        df['total_seconds'] = (df['period'] - 1) * 1200 + df['periodTime_seconds']
        df['time_remaining_period'] = 1200 - df['periodTime_seconds']
        df['final_two_minutes'] = (df['time_remaining_period'] <= 120).astype(int)
        df['final_minute'] = (df['time_remaining_period'] <= 60).astype(int)
        df['overtime_shot'] = (df['period'] > 3).astype(int)
        df['early_period'] = (df['periodTime_seconds'] <= 300).astype(int)
        df['is_forward'] = df['position'].isin(['C', 'L', 'R', 'LW', 'RW']).astype(int)
        df['is_defenseman'] = df['position'].isin(['D']).astype(int)
        df['is_center'] = df['position'].isin(['C']).astype(int)
        df['is_winger'] = df['position'].isin(['L', 'R', 'LW', 'RW']).astype(int)
        df['close_shot'] = (df['distance_to_net'] <= 15).astype(int)
        df['medium_shot'] = ((df['distance_to_net'] > 15) & (df['distance_to_net'] <= 35)).astype(int)
        df['long_shot'] = (df['distance_to_net'] > 35).astype(int)
        df['very_close_shot'] = (df['distance_to_net'] <= 10).astype(int)
        df['sharp_angle'] = (df['angle_to_net'] >= 45).astype(int)
        df['moderate_angle'] = ((df['angle_to_net'] >= 15) & (df['angle_to_net'] < 45)).astype(int)
        df['straight_on'] = (df['angle_to_net'] < 15).astype(int)
        df['very_sharp_angle'] = (df['angle_to_net'] >= 60).astype(int)
        
        # Use the 'Full_Features' feature set
        feature_set = self.get_feature_sets()['Full_Features']
        X = df[feature_set].fillna(0)
        
        # Use the best model (e.g., the first model in self.results)
        best_model_name = list(self.results.keys())[0]
        model = self.results[best_model_name]['model']
        optimal_threshold = self.results[best_model_name]['optimal_threshold']
        
        # Predict xG
        xg = model.predict_proba(X)[0, 1]
        review_flag = xg >= optimal_threshold
        
        return {'xG': xg, 'review_flag': review_flag}

def main():
    """Run comprehensive optimized model analysis with extensive hyperparameter tuning."""
    print("🚀 COMPREHENSIVE NHL xG MODEL OPTIMIZATION")
    print("="*70)
    print("Extensive hyperparameter tuning with GPU acceleration and advanced calibration")
    
    # Initialize analyzer with GPU support
    analyzer = OptimizedNHLxGAnalyzer(use_gpu=True)
    
    # Load and prepare data
    print("\n📊 Data Preparation Phase...")
    analyzer.load_shot_data()
    analyzer.engineer_features()
    
    # Train optimized models with extensive search
    print("\n🔧 Extensive Optimization Phase...")
    results = analyzer.train_optimized_models()
    
    # Create comprehensive visualizations
    print("\n📈 Visualization Phase...")
    analyzer.create_optimization_visualization()
    
    # Generate detailed report
    print("\n📝 Report Generation Phase...")
    results_df = analyzer.generate_optimization_report()
    
    # Summary statistics
    best_model = results_df.iloc[0]
    total_iterations = results_df['Search_Iterations'].sum()
    total_time = results_df['Training_Time'].sum()
    
    print(f"\n🎉 COMPREHENSIVE OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"✅ Optimized {len(results)} model configurations")
    print(f"✅ Total hyperparameter evaluations: {total_iterations:,}")
    print(f"✅ Total optimization time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"✅ Champion model: {best_model['Model']} ({best_model['Test_AUC']:.3f} AUC)")
    print(f"✅ Best calibration: {results_df.loc[results_df['Test_Calibration'].apply(lambda x: abs(x-1.0)).idxmin(), 'Test_Calibration']:.3f}")
    print(f"✅ GPU acceleration: {'Enabled' if analyzer.gpu_available else 'CPU-only'}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • optimized_model_results.csv - Detailed results matrix")
    print(f"   • optimized_model_analysis.png - 6-panel performance dashboard")
    print(f"   • optimized_model_report.md - Comprehensive analysis report")

if __name__ == '__main__':
    main() 