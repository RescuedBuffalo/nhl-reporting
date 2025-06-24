#!/usr/bin/env python3
"""
Enhanced Context-Aware NHL Shot Clustering Analysis
====================================================

This script performs advanced unsupervised learning on real NHL shot data with enhanced contextual features:
- Special teams situations (power play, penalty kill, even strength)
- Fatigue patterns (time remaining in period and game)
- Player scoring history (previous season goals/shooting %)

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedContextualNHLClustering:
    def __init__(self, db_path='../data_pipeline/nhl_stats.db', sample_ratio=1.0):
        self.db_path = db_path
        self.sample_ratio = sample_ratio
        self.shot_data = None
        self.clustering_results = {}
        
        print("🏒 ENHANCED CONTEXT-AWARE NHL SHOT CLUSTERING ANALYSIS")
        print("="*70)
        print(f"Using real NHL data with advanced contextual features (most recent season, {sample_ratio*100:.0f}% data)")
        print("⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
        print("🔧 Enhanced features: Special teams, fatigue patterns, player scoring history")
        print("📊 Key Innovation: Corrected player scoring tiers using proper goal/shot player ID mapping")
    
    def load_enhanced_contextual_data(self):
        """Load real shot data with enhanced contextual information."""
        print("\n🏒 LOADING ENHANCED NHL CONTEXTUAL SHOT DATA")
        print("="*60)
        print("⏳ Connecting to database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load shot events with enhanced context from most recent season only, excluding empty net goals
        print("⏳ Executing enhanced database query...")
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
            g.gameDate,
            g.season
        FROM events e
        JOIN games g ON e.gamePk = g.gamePk
        WHERE e.eventType IN ('goal', 'shot-on-goal')
        AND e.x IS NOT NULL 
        AND e.y IS NOT NULL
        AND g.gameDate >= '2024-01-01'  -- Most recent season only
        AND (e.details IS NULL OR e.details NOT LIKE '%Empty Net%')  -- Exclude empty net goals
        ORDER BY g.gameDate, e.gamePk, e.period, e.periodTime
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"📊 Loaded {len(df):,} total shot events from most recent season (excluding empty net)")
        print(f"🎯 Goals: {(df['eventType'] == 'goal').sum():,}")
        print(f"🏒 Shots on goal: {(df['eventType'] == 'shot-on-goal').sum():,}")
        print(f"📅 Games: {df['gamePk'].nunique():,}")
        print(f"📈 Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
        
        # Create target variable
        df['is_goal'] = (df['eventType'] == 'goal').astype(int)
        
        # Apply sampling if less than 100%
        if self.sample_ratio < 1.0:
            print(f"\n📊 Applying stratified sampling ({self.sample_ratio*100:.0f}% of data)...")
            print("⏳ Splitting data by goal status...")
            
            # Split by goal status to maintain ratio
            goals = df[df['is_goal'] == 1]
            non_goals = df[df['is_goal'] == 0]
            
            # Sample each group
            n_goals_sample = int(len(goals) * self.sample_ratio)
            n_non_goals_sample = int(len(non_goals) * self.sample_ratio)
            
            print(f"⏳ Sampling {n_goals_sample:,} goals and {n_non_goals_sample:,} non-goals...")
            goals_sampled = goals.sample(n=n_goals_sample, random_state=42)
            non_goals_sampled = non_goals.sample(n=n_non_goals_sample, random_state=42)
            
            # Combine sampled data
            df_sampled = pd.concat([goals_sampled, non_goals_sampled]).reset_index(drop=True)
            
            print(f"✅ Sampled {len(df_sampled):,} shots ({len(df_sampled)/len(df)*100:.1f}% of total)")
            print(f"🎯 Sampled goals: {df_sampled['is_goal'].sum():,} ({df_sampled['is_goal'].mean():.1%})")
            print(f"📊 Original goal rate: {df['is_goal'].mean():.1%}")
            print(f"📊 Sampled goal rate: {df_sampled['is_goal'].mean():.1%}")
            
            return df_sampled
        else:
            print(f"\n📊 Using complete dataset (100% of most recent season data)")
            print(f"✅ Complete dataset: {len(df):,} shots")
            print(f"🎯 Total goals: {df['is_goal'].sum():,} ({df['is_goal'].mean():.1%})")
            print(f"📊 Goal rate: {df['is_goal'].mean():.1%}")
            
            return df
    
    def engineer_enhanced_contextual_features(self, df):
        """Engineer enhanced contextual features including special teams, fatigue, and player history."""
        print("\n🔧 ENGINEERING ENHANCED CONTEXTUAL FEATURES")
        print("="*60)
        print("⏳ Processing spatial coordinates...")
        
        # Use absolute x-axis for ice symmetry (both sides of ice are equivalent)
        df['x_abs'] = np.abs(df['x'])
        
        print("⏳ Calculating geometric features...")
        # Basic geometric features using absolute x (single net at x=89)
        df['distance_to_net'] = np.sqrt((df['x_abs'] - 89)**2 + df['y']**2)
        df['angle_to_net'] = np.abs(np.arctan2(df['y'], 89 - df['x_abs']) * 180 / np.pi)
        
        print("⏳ Creating zone features...")
        # Zone features using absolute x (single net)
        df['in_crease'] = ((df['x_abs'] >= 85) & (np.abs(df['y']) <= 4)).astype(int)
        df['in_slot'] = ((df['x_abs'] >= 75) & (df['x_abs'] <= 89) & (np.abs(df['y']) <= 22)).astype(int)
        df['from_point'] = ((df['x_abs'] <= 65) & (np.abs(df['y']) >= 15)).astype(int)
        df['high_danger'] = ((df['x_abs'] >= 80) & (np.abs(df['y']) <= 15)).astype(int)
        
        print("⏳ Creating distance and angle categories...")
        # Distance categories
        df['close_shot'] = (df['distance_to_net'] <= 15).astype(int)
        df['medium_shot'] = ((df['distance_to_net'] > 15) & (df['distance_to_net'] <= 35)).astype(int)
        df['long_shot'] = (df['distance_to_net'] > 35).astype(int)
        
        # Angle categories
        df['sharp_angle'] = (df['angle_to_net'] >= 45).astype(int)
        df['moderate_angle'] = ((df['angle_to_net'] >= 15) & (df['angle_to_net'] < 45)).astype(int)
        df['straight_on'] = (df['angle_to_net'] < 15).astype(int)
        
        print("⏳ Processing enhanced time and fatigue features...")
        # Time features from real data
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df['period_minutes'] = df['periodTime'].str.split(':').str[0].astype(float)
        df['period_seconds'] = df['periodTime'].str.split(':').str[1].astype(float)
        df['total_seconds'] = (df['period'] - 1) * 1200 + df['period_minutes'] * 60 + df['period_seconds']
        df['time_remaining_period'] = 1200 - (df['period_minutes'] * 60 + df['period_seconds'])
        
        # Enhanced fatigue patterns
        print("⏳ Creating fatigue pattern features...")
        # Time remaining in game (regulation = 3600 seconds)
        df['time_remaining_game'] = np.where(
            df['period'] <= 3,
            3600 - df['total_seconds'],
            0  # Overtime has no "time remaining"
        )
        
        # Fatigue categories based on time remaining in period
        df['period_fatigue'] = pd.cut(
            df['time_remaining_period'], 
            bins=[0, 300, 600, 900, 1200], 
            labels=['very_tired', 'tired', 'moderate', 'fresh'],
            include_lowest=True
        ).astype(str)
        
        # Game fatigue categories
        df['game_fatigue'] = pd.cut(
            df['time_remaining_game'], 
            bins=[0, 900, 1800, 2700, 3600], 
            labels=['very_tired', 'tired', 'moderate', 'fresh'],
            include_lowest=True
        ).astype(str)
        
        print("⏳ Extracting special teams situations...")
        # Parse JSON details for enhanced context
        df['shooting_player_id'] = None
        df['situation_code'] = None
        df['shot_type'] = None
        
        for idx, row in df.iterrows():
            if pd.notna(row['details']):
                try:
                    details = json.loads(row['details'])
                    # Extract shooting player ID
                    if 'details' in details and 'shootingPlayerId' in details['details']:
                        df.at[idx, 'shooting_player_id'] = details['details']['shootingPlayerId']
                    
                    # Extract situation code for special teams
                    if 'situationCode' in details:
                        df.at[idx, 'situation_code'] = details['situationCode']
                    
                    # Extract shot type
                    if 'details' in details and 'shotType' in details['details']:
                        df.at[idx, 'shot_type'] = details['details']['shotType']
                        
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Special teams classification based on situation codes
        print("⏳ Classifying special teams situations...")
        df['special_teams'] = 'unknown'
        df['power_play'] = 0
        df['penalty_kill'] = 0
        df['even_strength'] = 0
        
        # NHL situation codes: AABB where AA = away skaters, BB = home skaters
        # 1551 = 5v5 even strength, 1541 = 5v4 power play, 1451 = 4v5 penalty kill, etc.
        for idx, row in df.iterrows():
            if pd.notna(row['situation_code']):
                code = str(row['situation_code'])
                if len(code) == 4:
                    away_skaters = int(code[:2]) if code[:2].isdigit() else 0
                    home_skaters = int(code[2:]) if code[2:].isdigit() else 0
                    
                    # Determine special teams situation
                    if away_skaters == home_skaters:
                        df.at[idx, 'special_teams'] = 'even_strength'
                        df.at[idx, 'even_strength'] = 1
                    elif away_skaters > home_skaters:
                        df.at[idx, 'special_teams'] = 'away_power_play'
                        df.at[idx, 'power_play'] = 1 if row['teamId'] != row['teamId'] else 0  # Need to determine team
                    elif home_skaters > away_skaters:
                        df.at[idx, 'special_teams'] = 'home_power_play'
                        df.at[idx, 'power_play'] = 1 if row['teamId'] == row['teamId'] else 0  # Need to determine team
        
        # Simplified special teams (we'll classify based on situation patterns)
        df['on_power_play'] = ((df['situation_code'].str.contains('1541|1531|1521', na=False)) | 
                              (df['situation_code'].str.contains('0651|0641|0631', na=False))).astype(int)
        df['on_penalty_kill'] = ((df['situation_code'].str.contains('1451|1351|1251', na=False)) | 
                                (df['situation_code'].str.contains('0561|0461|0361', na=False))).astype(int)
        df['even_strength'] = ((df['situation_code'].str.contains('1551|1010', na=False)) | 
                              (df['on_power_play'] == 0) & (df['on_penalty_kill'] == 0)).astype(int)
        
        print("⏳ Calculating player scoring history...")
        # Calculate previous season stats for each player
        df = self.add_player_scoring_history(df)
        
        # Create basic context features
        print("⏳ Creating basic contextual features...")
        # Period context
        df['period_context'] = df['period'].apply(lambda x: 
            'first_period' if x == 1 else 
            'second_period' if x == 2 else 
            'third_period' if x == 3 else 
            'overtime')
        
        # Enhanced time pressure context
        df['time_pressure'] = df.apply(lambda row: 
            'final_minutes' if row['period'] == 3 and row['time_remaining_period'] <= 120 else
            'overtime' if row['period'] > 3 else
            'period_end_tired' if row['time_remaining_period'] <= 300 else
            'period_start_fresh' if row['time_remaining_period'] >= 900 else
            'regular_time', axis=1)
        
        # Period timing context with fatigue
        df['period_timing'] = df.apply(lambda row:
            'period_start_fresh' if row['period_minutes'] <= 2 else
            'period_end_tired' if row['period_minutes'] >= 18 else
            'period_middle', axis=1)
        
        # Enhanced pressure situations
        df['final_two_minutes'] = (
            (df['period'] == 3) & 
            (df['time_remaining_period'] <= 120)
        ).astype(int)
        df['overtime_shot'] = (df['period'] > 3).astype(int)
        df['period_start_shot'] = (df['period_minutes'] <= 2).astype(int)
        df['period_end_shot'] = (df['period_minutes'] >= 18).astype(int)
        df['high_fatigue_shot'] = (df['time_remaining_period'] <= 300).astype(int)
        df['fresh_legs_shot'] = (df['time_remaining_period'] >= 900).astype(int)
        
        # Spatial aggregation using absolute x (single net)
        print("⏳ Applying spatial aggregation...")
        x_bins = pd.cut(df['x_abs'], bins=20, labels=False)
        y_bins = pd.cut(df['y'], bins=10, labels=False)
        df['spatial_grid_x'] = x_bins
        df['spatial_grid_y'] = y_bins
        df['spatial_grid'] = x_bins * 10 + y_bins
        
        print("⏳ Encoding categorical features...")
        # Encode categorical context features
        categorical_features = ['period_context', 'time_pressure', 'period_timing', 
                               'special_teams', 'period_fatigue', 'game_fatigue', 
                               'player_scoring_tier', 'shot_type']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].fillna('unknown'))
        
        print(f"✅ Engineered enhanced contextual features from real data")
        print(f"📈 Goal rate: {df['is_goal'].mean():.1%}")
        print(f"🎯 Period distribution: {df['period_context'].value_counts().to_dict()}")
        print(f"⏰ Time pressure distribution: {df['time_pressure'].value_counts().to_dict()}")
        print(f"🏒 Special teams distribution: {df[['on_power_play', 'on_penalty_kill', 'even_strength']].sum().to_dict()}")
        print(f"😴 Fatigue distribution: {df['period_fatigue'].value_counts().to_dict()}")
        print(f"🎯 Player scoring tiers: {df['player_scoring_tier'].value_counts().to_dict()}")
        print(f"🏒 Using absolute x-axis: Single net analysis (right side only)")
        print(f"⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
        
        return df
    
    def add_player_scoring_history(self, df):
        """Add player scoring history from previous seasons."""
        print("⏳ Calculating player scoring history from previous seasons...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get previous season stats for all players
        # Note: Goals use 'scoringPlayerId' while shots use 'shootingPlayerId'
        previous_season_query = """
        WITH player_shots AS (
            SELECT 
                JSON_EXTRACT(details, '$.details.shootingPlayerId') as player_id,
                COUNT(*) as shots
            FROM events e
            JOIN games g ON e.gamePk = g.gamePk
            WHERE e.eventType = 'shot-on-goal'
            AND JSON_EXTRACT(details, '$.details.shootingPlayerId') IS NOT NULL
            AND g.gameDate < '2024-01-01'  -- Previous seasons
            AND g.gameDate >= '2019-01-01'  -- Expanded to include more seasons
            GROUP BY JSON_EXTRACT(details, '$.details.shootingPlayerId')
        ),
        player_goals AS (
            SELECT 
                JSON_EXTRACT(details, '$.details.scoringPlayerId') as player_id,
                COUNT(*) as goals
            FROM events e
            JOIN games g ON e.gamePk = g.gamePk
            WHERE e.eventType = 'goal'
            AND JSON_EXTRACT(details, '$.details.scoringPlayerId') IS NOT NULL
            AND g.gameDate < '2024-01-01'  -- Previous seasons
            AND g.gameDate >= '2019-01-01'  -- Expanded to include more seasons
            GROUP BY JSON_EXTRACT(details, '$.details.scoringPlayerId')
        )
        SELECT 
            s.player_id,
            s.shots as total_shots,
            COALESCE(g.goals, 0) as total_goals,
            CASE 
                WHEN s.shots > 0 THEN CAST(COALESCE(g.goals, 0) AS FLOAT) / s.shots
                ELSE 0.0
            END as shooting_pct
        FROM player_shots s
        LEFT JOIN player_goals g ON s.player_id = g.player_id
        WHERE s.shots >= 10  -- Minimum shots for meaningful stats
        """
        
        previous_stats = pd.read_sql_query(previous_season_query, conn)
        conn.close()
        
        print(f"📊 Found previous season stats for {len(previous_stats):,} players")
        
        # Create player scoring history features
        df['player_previous_goals'] = 0
        df['player_previous_shots'] = 0
        df['player_previous_shooting_pct'] = 0.0
        df['player_has_history'] = 0
        df['player_scoring_tier'] = 'unknown'
        
        # Map previous stats to current shots
        player_stats_dict = {}
        for _, row in previous_stats.iterrows():
            if pd.notna(row['player_id']):
                player_id = str(int(float(row['player_id'])))
                player_stats_dict[player_id] = {
                    'goals': row['total_goals'],
                    'shots': row['total_shots'],
                    'shooting_pct': row['shooting_pct']
                }
        
        # Apply player stats to current data
        for idx, row in df.iterrows():
            if pd.notna(row['shooting_player_id']):
                player_id = str(int(row['shooting_player_id']))
                if player_id in player_stats_dict:
                    stats = player_stats_dict[player_id]
                    df.at[idx, 'player_previous_goals'] = stats['goals']
                    df.at[idx, 'player_previous_shots'] = stats['shots']
                    df.at[idx, 'player_previous_shooting_pct'] = stats['shooting_pct']
                    df.at[idx, 'player_has_history'] = 1
                    
                    # Create scoring tiers
                    if stats['goals'] >= 30:
                        df.at[idx, 'player_scoring_tier'] = 'elite_scorer'
                    elif stats['goals'] >= 20:
                        df.at[idx, 'player_scoring_tier'] = 'good_scorer'
                    elif stats['goals'] >= 10:
                        df.at[idx, 'player_scoring_tier'] = 'average_scorer'
                    elif stats['goals'] > 0:
                        df.at[idx, 'player_scoring_tier'] = 'low_scorer'
                    else:
                        df.at[idx, 'player_scoring_tier'] = 'non_scorer'
                else:
                    # Player exists but no previous season data
                    df.at[idx, 'player_scoring_tier'] = 'no_history'
        
        print(f"✅ Added player scoring history features")
        print(f"📊 Players with history: {df['player_has_history'].sum():,} shots")
        print(f"🎯 Scoring tier distribution: {df['player_scoring_tier'].value_counts().to_dict()}")
        
        return df
    
    def prepare_enhanced_clustering_data(self, df):
        """Prepare data for enhanced context-aware clustering."""
        print("\n🎯 PREPARING ENHANCED CONTEXTUAL CLUSTERING DATA")
        print("="*60)
        print("⏳ Selecting enhanced clustering features...")
        
        # Enhanced feature set including all new contextual features
        clustering_features = [
            # Spatial features
            'distance_to_net', 'angle_to_net',
            'in_crease', 'in_slot', 'from_point', 'high_danger',
            'close_shot', 'medium_shot', 'long_shot',
            'sharp_angle', 'moderate_angle', 'straight_on',
            'spatial_grid_x', 'spatial_grid_y',
            
            # Basic time features
            'final_two_minutes', 'overtime_shot', 'period_start_shot', 'period_end_shot',
            
            # Enhanced fatigue features
            'high_fatigue_shot', 'fresh_legs_shot', 'time_remaining_period', 'time_remaining_game',
            
            # Special teams features
            'on_power_play', 'on_penalty_kill', 'even_strength',
            
            # Player scoring history
            'player_previous_goals', 'player_previous_shots', 'player_previous_shooting_pct', 'player_has_history',
            
            # Encoded categorical features
            'period_context_encoded', 'time_pressure_encoded', 'period_timing_encoded',
            'special_teams_encoded', 'period_fatigue_encoded', 'game_fatigue_encoded',
            'player_scoring_tier_encoded'
        ]
        
        # Filter features that actually exist
        available_features = [f for f in clustering_features if f in df.columns]
        print(f"📊 Available features: {len(available_features)} out of {len(clustering_features)} requested")
        
        # Retain x, y, x_abs for plotting and original categorical columns for analysis
        keep_cols = available_features + ['is_goal', 'x', 'y', 'x_abs', 'shooting_player_id', 
                                         'player_scoring_tier', 'period', 'shot_type']
        keep_cols = [col for col in keep_cols if col in df.columns]
        
        # Check for NaN values before dropping
        print(f"📊 Dataset before cleaning: {len(df):,} shots")
        print(f"🎯 Goals before cleaning: {df['is_goal'].sum():,} ({df['is_goal'].mean():.1%})")
        
        # Only drop rows where clustering features have NaN (not plotting features)
        clustering_features_only = [f for f in available_features if f in df.columns]
        df_clean = df[keep_cols].dropna(subset=clustering_features_only)
        
        print(f"📊 Clean dataset: {len(df_clean):,} shots")
        print(f"🎯 Goals in clean data: {df_clean['is_goal'].sum():,} ({df_clean['is_goal'].mean():.1%})")
        
        # Debug: Check if we still have goal data
        if df_clean['is_goal'].sum() == 0:
            print("⚠️  WARNING: No goals found in clean data - investigating...")
            print(f"   Original goals: {df['is_goal'].sum():,}")
            print(f"   NaN values in is_goal: {df['is_goal'].isna().sum()}")
            # Check specific columns for NaN
            for col in clustering_features_only[:5]:  # Check first 5 features
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"   NaN in {col}: {nan_count:,}")
        
        
        # Prepare features
        X = df_clean[available_features].values
        y = df_clean['is_goal'].values
        
        # Scale features
        print("⏳ Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"✅ Prepared {X_scaled.shape[1]} enhanced features for {X_scaled.shape[0]} shots")
        print(f"🔧 Enhanced features include: spatial, temporal, fatigue, special teams, player history")
        
        return X_scaled, y, df_clean, scaler, available_features
    
    def run_enhanced_clustering(self, X, n_clusters=6):
        """Run enhanced context-aware clustering with business constraints."""
        print(f"\n🔍 RUNNING ENHANCED CONTEXTUAL CLUSTERING")
        print("="*60)
        print(f"⏳ Using {n_clusters} clusters (business constraint: max 10)...")
        
        # Ensure we don't exceed business limit
        if n_clusters > 10:
            print(f"⚠️  Requested {n_clusters} clusters exceeds business limit, using 10 instead")
            n_clusters = 10
        
        algorithms = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters)
        }
        
        clustering_results = {}
        
        for name, algorithm in algorithms.items():
            print(f"⏳ Running {name}...")
            try:
                labels = algorithm.fit_predict(X)
                n_clusters_found = len(set(labels))
                
                # Calculate silhouette score
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                
                clustering_results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters_found,
                    'silhouette_score': silhouette,
                    'calinski_score': calinski
                }
                
                print(f"✅ {name}: {n_clusters_found} clusters, silhouette: {silhouette:.3f}, Calinski: {calinski:.0f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
        
        return clustering_results
    
    def analyze_enhanced_clusters(self, X, y, df_clean, clustering_results):
        """Analyze enhanced context-aware clustering results with minimum sample size filter."""
        print("\n📊 ANALYZING ENHANCED CONTEXTUAL CLUSTERS")
        print("="*60)
        
        # Find best algorithm
        best_algorithm = max(clustering_results.items(), 
                           key=lambda x: x[1]['silhouette_score'])[0]
        labels = clustering_results[best_algorithm]['labels']
        
        print(f"🏆 Best algorithm: {best_algorithm}")
        print(f"📈 Silhouette score: {clustering_results[best_algorithm]['silhouette_score']:.3f}")
        
        # Analyze each cluster with minimum sample size filter
        cluster_analysis = {}
        danger_classification = {}
        min_sample_size = 50  # Minimum shots per cluster for reliable analysis
        
        print("⏳ Analyzing individual enhanced clusters...")
        for cluster_id in set(labels):
            cluster_mask = labels == cluster_id
            cluster_data = df_clean[cluster_mask]
            
            # Skip clusters that are too small for reliable analysis
            size = len(cluster_data)
            if size < min_sample_size:
                print(f"   ⚠️  Cluster {cluster_id}: {size} shots (skipped - below minimum {min_sample_size})")
                continue
            
            # Calculate enhanced cluster statistics
            goal_rate = cluster_data['is_goal'].mean()
            avg_distance = cluster_data['distance_to_net'].mean()
            avg_angle = cluster_data['angle_to_net'].mean()
            high_danger_rate = cluster_data['high_danger'].mean()
            final_two_rate = cluster_data['final_two_minutes'].mean()
            
            # Enhanced contextual statistics
            power_play_rate = cluster_data['on_power_play'].mean() if 'on_power_play' in cluster_data.columns else 0
            penalty_kill_rate = cluster_data['on_penalty_kill'].mean() if 'on_penalty_kill' in cluster_data.columns else 0
            high_fatigue_rate = cluster_data['high_fatigue_shot'].mean() if 'high_fatigue_shot' in cluster_data.columns else 0
            
            # Calculate elite scorer rate with debug info
            if 'player_scoring_tier' in cluster_data.columns:
                tier_counts = cluster_data['player_scoring_tier'].value_counts()
                elite_count = tier_counts.get('elite_scorer', 0)
                elite_scorer_rate = elite_count / len(cluster_data) if len(cluster_data) > 0 else 0
            else:
                elite_scorer_rate = 0
            
            # Enhanced danger classification
            overall_goal_rate = df_clean['is_goal'].mean()
            
            # Apply enhanced business logic for danger classification
            if avg_distance <= 25 and goal_rate >= overall_goal_rate * 1.2:
                danger_type = 'High Danger'
            elif avg_distance > 75 or goal_rate <= overall_goal_rate * 0.7:
                danger_type = 'Low Danger'
            else:
                danger_type = 'Medium Danger'
            
            cluster_analysis[cluster_id] = {
                'size': size,
                'goal_rate': goal_rate,
                'avg_distance': avg_distance,
                'avg_angle': avg_angle,
                'high_danger_rate': high_danger_rate,
                'final_two_rate': final_two_rate,
                'power_play_rate': power_play_rate,
                'penalty_kill_rate': penalty_kill_rate,
                'high_fatigue_rate': high_fatigue_rate,
                'elite_scorer_rate': elite_scorer_rate
            }
            
            danger_classification[cluster_id] = danger_type
            
            print(f"   Cluster {cluster_id}: {size:,} shots, {goal_rate:.1%} goal rate, {danger_type}")
            print(f"      Special teams: PP {power_play_rate:.1%}, PK {penalty_kill_rate:.1%}")
            print(f"      Fatigue: {high_fatigue_rate:.1%}, Elite scorers: {elite_scorer_rate:.1%}")
        
        print(f"\n✅ Analyzed {len(cluster_analysis)} enhanced clusters meeting minimum sample size")
        
        return cluster_analysis, danger_classification
    
    def export_comprehensive_cluster_data(self, df_clean, labels, cluster_analysis, danger_classification, features):
        """Export comprehensive cluster data for narrative creation."""
        print("\n📝 EXPORTING COMPREHENSIVE CLUSTER DATA")
        print("="*60)
        print("⏳ Generating detailed cluster statistics...")
        
        # Create comprehensive cluster report
        report_lines = []
        report_lines.append("# Enhanced Context-Aware NHL Shot Clustering: Comprehensive Cluster Analysis")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("## Analysis Overview")
        report_lines.append(f"- **Total Shots Analyzed**: {len(df_clean):,}")
        report_lines.append(f"- **Total Goals**: {df_clean['is_goal'].sum():,}")
        report_lines.append(f"- **Overall Goal Rate**: {df_clean['is_goal'].mean():.1%}")
        report_lines.append(f"- **Features Used**: {len(features)} enhanced contextual features")
        report_lines.append(f"- **Clusters Identified**: {len(cluster_analysis)} meaningful clusters")
        report_lines.append("")
        
        # Add feature summary
        report_lines.append("## Features Used in Analysis")
        feature_categories = {
            'Spatial': ['distance_to_net', 'angle_to_net', 'in_crease', 'in_slot', 'from_point', 'high_danger', 'close_shot', 'medium_shot', 'long_shot', 'sharp_angle', 'moderate_angle', 'straight_on', 'spatial_grid_x', 'spatial_grid_y'],
            'Time & Fatigue': ['final_two_minutes', 'overtime_shot', 'period_start_shot', 'period_end_shot', 'high_fatigue_shot', 'fresh_legs_shot', 'time_remaining_period', 'time_remaining_game'],
            'Special Teams': ['on_power_play', 'on_penalty_kill', 'even_strength'],
            'Player History': ['player_previous_goals', 'player_previous_shots', 'player_previous_shooting_pct', 'player_has_history'],
            'Context Encoded': [f for f in features if f.endswith('_encoded')]
        }
        
        for category, feature_list in feature_categories.items():
            used_features = [f for f in feature_list if f in features]
            if used_features:
                report_lines.append(f"- **{category}**: {len(used_features)} features")
        report_lines.append("")
        
        # Detailed cluster analysis
        report_lines.append("## Detailed Cluster Analysis")
        report_lines.append("")
        
        for cluster_id in sorted(cluster_analysis.keys()):
            analysis = cluster_analysis[cluster_id]
            danger = danger_classification[cluster_id]
            
            # Get cluster data
            cluster_mask = labels == cluster_id
            cluster_data = df_clean[cluster_mask]
            
            report_lines.append(f"### Cluster {cluster_id}: {danger}")
            report_lines.append("-" * 40)
            report_lines.append("")
            
            # Basic statistics
            report_lines.append("#### Basic Statistics")
            report_lines.append(f"- **Shot Count**: {analysis['size']:,} shots ({analysis['size']/len(df_clean)*100:.1f}% of total)")
            report_lines.append(f"- **Goal Rate**: {analysis['goal_rate']:.1%} (vs {df_clean['is_goal'].mean():.1%} overall)")
            report_lines.append(f"- **Goals Scored**: {int(analysis['goal_rate'] * analysis['size'])} goals")
            report_lines.append(f"- **Danger Classification**: {danger}")
            report_lines.append("")
            
            # Spatial characteristics
            report_lines.append("#### Spatial Characteristics")
            report_lines.append(f"- **Average Distance to Net**: {analysis['avg_distance']:.1f} feet")
            report_lines.append(f"- **Average Angle to Net**: {analysis['avg_angle']:.1f} degrees")
            report_lines.append(f"- **High Danger Zone Rate**: {analysis['high_danger_rate']:.1%}")
            
            # Zone breakdown
            if 'in_crease' in cluster_data.columns:
                crease_rate = cluster_data['in_crease'].mean()
                slot_rate = cluster_data['in_slot'].mean()
                point_rate = cluster_data['from_point'].mean()
                report_lines.append(f"- **In Crease**: {crease_rate:.1%}")
                report_lines.append(f"- **In Slot**: {slot_rate:.1%}")
                report_lines.append(f"- **From Point**: {point_rate:.1%}")
            report_lines.append("")
            
            # Special teams breakdown
            report_lines.append("#### Special Teams Analysis")
            report_lines.append(f"- **Power Play Rate**: {analysis['power_play_rate']:.1%}")
            report_lines.append(f"- **Penalty Kill Rate**: {analysis['penalty_kill_rate']:.1%}")
            even_strength_rate = 1 - analysis['power_play_rate'] - analysis['penalty_kill_rate']
            report_lines.append(f"- **Even Strength Rate**: {even_strength_rate:.1%}")
            
            # Special teams goal rates
            if analysis['power_play_rate'] > 0:
                pp_data = cluster_data[cluster_data['on_power_play'] == 1]
                pp_goal_rate = pp_data['is_goal'].mean() if len(pp_data) > 0 else 0
                report_lines.append(f"- **Power Play Goal Rate**: {pp_goal_rate:.1%}")
            
            if analysis['penalty_kill_rate'] > 0:
                pk_data = cluster_data[cluster_data['on_penalty_kill'] == 1]
                pk_goal_rate = pk_data['is_goal'].mean() if len(pk_data) > 0 else 0
                report_lines.append(f"- **Penalty Kill Goal Rate**: {pk_goal_rate:.1%}")
            report_lines.append("")
            
            # Fatigue and timing analysis
            report_lines.append("#### Fatigue & Timing Analysis")
            report_lines.append(f"- **High Fatigue Shot Rate**: {analysis['high_fatigue_rate']:.1%}")
            report_lines.append(f"- **Final Two Minutes Rate**: {analysis['final_two_rate']:.1%}")
            
            if 'fresh_legs_shot' in cluster_data.columns:
                fresh_legs_rate = cluster_data['fresh_legs_shot'].mean()
                report_lines.append(f"- **Fresh Legs Shot Rate**: {fresh_legs_rate:.1%}")
            
            if 'overtime_shot' in cluster_data.columns:
                overtime_rate = cluster_data['overtime_shot'].mean()
                report_lines.append(f"- **Overtime Shot Rate**: {overtime_rate:.1%}")
            
            # Period breakdown
            if 'period' in cluster_data.columns:
                period_dist = cluster_data['period'].value_counts().sort_index()
                total_period_shots = period_dist.sum()
                report_lines.append("- **Period Distribution**:")
                for period, count in period_dist.items():
                    period_name = f"Period {period}" if period <= 3 else "Overtime"
                    report_lines.append(f"  - {period_name}: {count:,} shots ({count/total_period_shots:.1%})")
            report_lines.append("")
            
            # Player scoring history analysis
            report_lines.append("#### Player Scoring History")
            report_lines.append(f"- **Elite Scorer Rate**: {analysis['elite_scorer_rate']:.1%}")
            
            if 'player_scoring_tier' in cluster_data.columns:
                scoring_dist = cluster_data['player_scoring_tier'].value_counts()
                total_with_tiers = scoring_dist.sum()
                report_lines.append("- **Scoring Tier Distribution**:")
                tier_order = ['elite_scorer', 'good_scorer', 'average_scorer', 'low_scorer', 'non_scorer', 'no_history', 'unknown']
                for tier in tier_order:
                    if tier in scoring_dist:
                        count = scoring_dist[tier]
                        report_lines.append(f"  - {tier.replace('_', ' ').title()}: {count:,} shots ({count/total_with_tiers:.1%})")
            
            if 'player_has_history' in cluster_data.columns:
                history_rate = cluster_data['player_has_history'].mean()
                report_lines.append(f"- **Players with Previous Season Data**: {history_rate:.1%}")
                
                if history_rate > 0:
                    history_data = cluster_data[cluster_data['player_has_history'] == 1]
                    avg_prev_goals = history_data['player_previous_goals'].mean()
                    avg_prev_shots = history_data['player_previous_shots'].mean()
                    avg_prev_pct = history_data['player_previous_shooting_pct'].mean()
                    report_lines.append(f"- **Average Previous Season Goals**: {avg_prev_goals:.1f}")
                    report_lines.append(f"- **Average Previous Season Shots**: {avg_prev_shots:.1f}")
                    report_lines.append(f"- **Average Previous Season Shooting %**: {avg_prev_pct:.1%}")
            report_lines.append("")
            
            # Shot type analysis
            if 'shot_type' in cluster_data.columns:
                shot_types = cluster_data['shot_type'].value_counts()
                if len(shot_types) > 0:
                    report_lines.append("#### Shot Type Analysis")
                    total_typed_shots = shot_types.sum()
                    for shot_type, count in shot_types.head(5).items():  # Top 5 shot types
                        if pd.notna(shot_type):
                            report_lines.append(f"- **{shot_type.title()}**: {count:,} shots ({count/total_typed_shots:.1%})")
                    report_lines.append("")
            
            # Performance comparison
            report_lines.append("#### Performance vs Overall Average")
            overall_goal_rate = df_clean['is_goal'].mean()
            performance_ratio = analysis['goal_rate'] / overall_goal_rate if overall_goal_rate > 0 else 0
            report_lines.append(f"- **Goal Rate vs Average**: {performance_ratio:.2f}x overall rate")
            
            if performance_ratio > 1.2:
                report_lines.append("- **Performance**: Significantly above average (High Danger)")
            elif performance_ratio > 0.8:
                report_lines.append("- **Performance**: Around average (Medium Danger)")
            else:
                report_lines.append("- **Performance**: Below average (Low Danger)")
            report_lines.append("")
            report_lines.append("")
        
        # Summary statistics
        report_lines.append("## Summary Statistics")
        report_lines.append("")
        
        # Danger level summary
        danger_summary = {}
        for cluster_id, danger in danger_classification.items():
            if danger not in danger_summary:
                danger_summary[danger] = {'clusters': 0, 'shots': 0, 'goals': 0}
            danger_summary[danger]['clusters'] += 1
            danger_summary[danger]['shots'] += cluster_analysis[cluster_id]['size']
            danger_summary[danger]['goals'] += int(cluster_analysis[cluster_id]['goal_rate'] * cluster_analysis[cluster_id]['size'])
        
        report_lines.append("### By Danger Level")
        for danger_level in ['High Danger', 'Medium Danger', 'Low Danger']:
            if danger_level in danger_summary:
                stats = danger_summary[danger_level]
                goal_rate = stats['goals'] / stats['shots'] if stats['shots'] > 0 else 0
                report_lines.append(f"- **{danger_level}**: {stats['clusters']} clusters, {stats['shots']:,} shots, {goal_rate:.1%} goal rate")
        
        # Write to file
        filename = 'enhanced_cluster_comprehensive_analysis.md'
        with open(filename, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Comprehensive cluster analysis exported to '{filename}'")
        print(f"📊 Exported detailed statistics for {len(cluster_analysis)} clusters")
        print(f"📝 Ready for narrative creation and business insights")
        
        return filename
    
    def create_enhanced_visualization(self, df_clean, labels, danger_classification, cluster_analysis):
        """Create enhanced context-aware visualization."""
        print("\n🏒 CREATING ENHANCED CONTEXT-AWARE VISUALIZATION")
        print("="*60)
        print("⏳ Creating enhanced visualization...")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Ice rink with enhanced clusters
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Only show clusters that meet minimum sample size
        valid_clusters = [c for c in set(labels) if c in danger_classification]
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_clusters)))
        
        for i, cluster_id in enumerate(valid_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = df_clean[cluster_mask]
            
            ax1.scatter(cluster_data['x_abs'], cluster_data['y'], 
                       c=[colors[i]], alpha=0.6, s=25, 
                       label=f'Cluster {cluster_id}')
        
        # Draw corrected rink (half ice from center to goal)
        ax1.plot([0, 89], [42.5, 42.5], 'k-', linewidth=3)
        ax1.plot([0, 89], [-42.5, -42.5], 'k-', linewidth=3)
        ax1.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=3)
        ax1.plot([89, 89], [-42.5, 42.5], 'k-', linewidth=3)
        
        # Enhanced rink features
        net_x, goal_line_x, blue_line_x = 89, 85, 25
        ax1.axvline(x=0, color='red', linestyle='-', alpha=0.9, linewidth=4)
        ax1.axvline(x=blue_line_x, color='blue', linestyle='-', alpha=0.9, linewidth=3)
        ax1.plot([goal_line_x, goal_line_x], [-42.5, 42.5], 'r-', linewidth=4)
        ax1.plot([net_x, net_x], [-3, 3], 'r-', linewidth=10)
        
        ax1.set_xlim(-5, 95)
        ax1.set_ylim(-50, 50)
        ax1.set_title('Enhanced Context-Aware Shot Clusters', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Distance from Center Line (feet)')
        ax1.set_ylabel('Y Coordinate (feet)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Special teams analysis
        ax2 = fig.add_subplot(gs[0, 2])
        if cluster_analysis:
            special_teams_data = []
            cluster_ids = []
            for cluster_id, analysis in cluster_analysis.items():
                special_teams_data.append([
                    analysis['power_play_rate'],
                    analysis['penalty_kill_rate'],
                    1 - analysis['power_play_rate'] - analysis['penalty_kill_rate']  # Even strength
                ])
                cluster_ids.append(f"C{cluster_id}")
            
            special_teams_df = pd.DataFrame(
                special_teams_data,
                columns=['Power Play', 'Penalty Kill', 'Even Strength'],
                index=cluster_ids
            )
            special_teams_df.plot(kind='bar', stacked=True, ax=ax2)
            ax2.set_title('Special Teams Distribution by Cluster', fontweight='bold')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Rate')
            ax2.legend(title='Situation')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Fatigue patterns
        ax3 = fig.add_subplot(gs[1, 0])
        if 'high_fatigue_shot' in df_clean.columns:
            fatigue_by_cluster = []
            for cluster_id in valid_clusters:
                cluster_mask = labels == cluster_id
                fatigue_rate = df_clean[cluster_mask]['high_fatigue_shot'].mean()
                fatigue_by_cluster.append(fatigue_rate)
            
            ax3.bar(range(len(valid_clusters)), fatigue_by_cluster, color=colors)
            ax3.set_title('High Fatigue Shot Rate by Cluster', fontweight='bold')
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('High Fatigue Rate')
            ax3.set_xticks(range(len(valid_clusters)))
            ax3.set_xticklabels([f'C{c}' for c in valid_clusters])
        
        # Plot 4: Player scoring tier analysis
        ax4 = fig.add_subplot(gs[1, 1])
        if 'player_scoring_tier' in df_clean.columns and cluster_analysis:
            # Get actual player tier data for each cluster
            tier_data = []
            cluster_labels = []
            for cluster_id in valid_clusters:
                cluster_mask = labels == cluster_id
                cluster_data = df_clean[cluster_mask]
                
                # Count different scoring tiers
                tier_counts = cluster_data['player_scoring_tier'].value_counts()
                total_shots = len(cluster_data)
                
                # Focus on meaningful tiers
                elite_rate = tier_counts.get('elite_scorer', 0) / total_shots
                good_rate = tier_counts.get('good_scorer', 0) / total_shots
                avg_rate = tier_counts.get('average_scorer', 0) / total_shots
                
                tier_data.append([elite_rate, good_rate, avg_rate])
                cluster_labels.append(f'C{cluster_id}')
            
            if tier_data:
                tier_df = pd.DataFrame(tier_data, columns=['Elite', 'Good', 'Average'], index=cluster_labels)
                tier_df.plot(kind='bar', stacked=True, ax=ax4, color=['gold', 'silver', '#CD7F32'])
                ax4.set_title('Player Scoring Tiers by Cluster', fontweight='bold')
                ax4.set_xlabel('Cluster')
                ax4.set_ylabel('Rate')
                ax4.legend(title='Scorer Tier')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No scoring tier data\navailable', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Player Scoring Tiers by Cluster', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Player scoring data\nnot available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Player Scoring Tiers by Cluster', fontweight='bold')
        
        # Plot 5: Goal rates with enhanced context
        ax5 = fig.add_subplot(gs[1, 2])
        if cluster_analysis and valid_clusters:
            goal_rates = [cluster_analysis[c]['goal_rate'] for c in valid_clusters]
            danger_colors = ['red' if danger_classification[c] == 'High Danger' else 
                           'orange' if danger_classification[c] == 'Medium Danger' else 'lightblue' 
                           for c in valid_clusters]
            
            bars = ax5.bar(range(len(valid_clusters)), goal_rates, color=danger_colors, alpha=0.7)
            overall_rate = df_clean['is_goal'].mean()
            ax5.axhline(y=overall_rate, color='black', linestyle='--', 
                       label=f'Overall: {overall_rate:.1%}')
            
            # Add value labels
            for i, (bar, rate) in enumerate(zip(bars, goal_rates)):
                height = bar.get_height()
                if height > 0:  # Only add label if there's a meaningful height
                    ax5.text(bar.get_x() + bar.get_width()/2., height + max(0.001, overall_rate * 0.02),
                            f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax5.set_title('Goal Rates by Enhanced Cluster', fontweight='bold')
            ax5.set_xlabel('Cluster')
            ax5.set_ylabel('Goal Rate')
            ax5.set_xticks(range(len(valid_clusters)))
            ax5.set_xticklabels([f'C{c}' for c in valid_clusters])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No cluster data\navailable for\ngoal rate analysis', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Goal Rates by Enhanced Cluster', fontweight='bold')
        
        # Plot 6: Enhanced summary statistics (make it more readable)
        ax6 = fig.add_subplot(gs[2, :])
        if cluster_analysis and valid_clusters:
            # Create a more structured summary
            summary_lines = []
            summary_lines.append("ENHANCED CONTEXTUAL CLUSTER ANALYSIS SUMMARY")
            summary_lines.append("=" * 60)
            summary_lines.append("")
            
            for i, cluster_id in enumerate(sorted(valid_clusters)):
                analysis = cluster_analysis[cluster_id]
                danger = danger_classification[cluster_id]
                
                line1 = f"Cluster {cluster_id} ({danger}): {analysis['size']:,} shots ({analysis['goal_rate']:.1%} goals)"
                line2 = f"  Distance: {analysis['avg_distance']:.1f}ft | PP: {analysis['power_play_rate']:.1%} | PK: {analysis['penalty_kill_rate']:.1%}"
                line3 = f"  Fatigue: {analysis['high_fatigue_rate']:.1%} | Elite: {analysis['elite_scorer_rate']:.1%} | Final 2min: {analysis['final_two_rate']:.1%}"
                
                summary_lines.extend([line1, line2, line3, ""])
            
            summary_text = '\n'.join(summary_lines)
            ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.axis('off')
        else:
            ax6.text(0.5, 0.5, 'No cluster analysis data available for summary', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax6.transAxes, fontsize=12)
            ax6.axis('off')
        
        plt.suptitle('Enhanced Context-Aware NHL Shot Clustering Analysis\n'
                    'Special Teams • Fatigue Patterns • Player Scoring History • Game Context', 
                    fontsize=18, fontweight='bold')
        
        plt.savefig('enhanced_context_aware_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Enhanced context-aware visualization saved as 'enhanced_context_aware_clustering.png'")
        print("🔧 Enhanced features: Special teams, fatigue patterns, player scoring history")
        print("🏒 Corrected visualization: Half ice from center line to goal")
        print("⚠️  Caveat: Analysis assumes ice symmetry - both sides treated equally")
        print("📊 Filtered clusters by minimum sample size for reliable analysis")
        print("🚫 Empty net goals excluded from analysis")
    
    def run_complete_enhanced_analysis(self):
        """Run the complete enhanced context-aware clustering analysis."""
        print("\n📊 RUNNING COMPLETE ENHANCED ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        df = self.load_enhanced_contextual_data()
        df = self.engineer_enhanced_contextual_features(df)
        X_scaled, y, df_clean, scaler, features = self.prepare_enhanced_clustering_data(df)
        
        # Run clustering
        clustering_results = self.run_enhanced_clustering(X_scaled, n_clusters=6)
        
        # Analyze results
        cluster_analysis, danger_classification = self.analyze_enhanced_clusters(
            X_scaled, y, df_clean, clustering_results)
        
        # Create visualizations
        best_algorithm = max(clustering_results.items(), 
                           key=lambda x: x[1]['silhouette_score'])[0]
        labels = clustering_results[best_algorithm]['labels']
        
        self.create_enhanced_visualization(df_clean, labels, danger_classification, cluster_analysis)
        
        # Export comprehensive cluster data
        export_filename = self.export_comprehensive_cluster_data(
            df_clean, labels, cluster_analysis, danger_classification, features)
        
        print(f"\n🎉 ENHANCED CONTEXTUAL CLUSTERING ANALYSIS COMPLETE!")
        print("="*70)
        print(f"✅ Analyzed {len(df_clean):,} real NHL shots with enhanced context")
        print(f"✅ Features: {len(features)} enhanced contextual features")
        print(f"✅ Identified {len(cluster_analysis)} meaningful shot clusters")
        print(f"✅ Enhanced context: Special teams, fatigue, player history")
        print(f"📝 Comprehensive data exported to: {export_filename}")
        print(f"⚠️  Caveat: Analysis assumes ice symmetry - both sides treated equally")
        print(f"🏒 Business-ready clusters with practical interpretation")

def main():
    """Main execution function."""
    analyzer = EnhancedContextualNHLClustering(sample_ratio=1.0)
    analyzer.run_complete_enhanced_analysis()

if __name__ == "__main__":
    main() 