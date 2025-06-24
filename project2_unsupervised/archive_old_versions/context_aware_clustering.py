#!/usr/bin/env python3
"""
Context-Aware NHL Shot Clustering Analysis
Uses real NHL data with actual game context features
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Add data pipeline to path
sys.path.append('../data_pipeline/src')

def load_contextual_shot_data(db_path='../data_pipeline/nhl_stats.db', sample_ratio=0.25):
    """Load real NHL shot data with actual game context features and stratified sampling from most recent season."""
    print("🏒 LOADING REAL NHL CONTEXTUAL SHOT DATA")
    print("="*60)
    print("⏳ Connecting to database...")
    
    conn = sqlite3.connect(db_path)
    
    # Load shot events from most recent season only, excluding empty net goals
    print("⏳ Executing database query...")
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
    AND g.gameDate >= '2024-01-01'  -- Most recent season only
    AND (e.details IS NULL OR e.details NOT LIKE '%Empty Net%')  -- Exclude empty net goals
    ORDER BY g.gameDate, e.gamePk, e.period, e.periodTime
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"📊 Loaded {len(df):,} total shot events from most recent season (excluding empty net)")
    print(f"�� Goals: {(df['eventType'] == 'goal').sum():,}")
    print(f"🏒 Shots on goal: {(df['eventType'] == 'shot-on-goal').sum():,}")
    print(f"📅 Games: {df['gamePk'].nunique():,}")
    print(f"📈 Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
    
    # Create target variable
    df['is_goal'] = (df['eventType'] == 'goal').astype(int)
    
    # Stratified sampling to maintain goal ratio
    print(f"\n📊 Applying stratified sampling ({sample_ratio*100:.0f}% of data)...")
    print("⏳ Splitting data by goal status...")
    
    # Split by goal status to maintain ratio
    goals = df[df['is_goal'] == 1]
    non_goals = df[df['is_goal'] == 0]
    
    # Sample each group
    n_goals_sample = int(len(goals) * sample_ratio)
    n_non_goals_sample = int(len(non_goals) * sample_ratio)
    
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

def engineer_contextual_features(df):
    """Engineer contextual features from real NHL data with absolute x-axis (single net)."""
    print("\n🔧 ENGINEERING CONTEXTUAL FEATURES")
    print("="*50)
    print("⏳ Processing spatial coordinates...")
    
    # Use absolute x-axis for ice symmetry (both sides of ice are equivalent)
    # This captures only half the rink (right side) to increase data density per point
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
    
    print("⏳ Processing time features...")
    # Time features from real data
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    df['period_minutes'] = df['periodTime'].str.split(':').str[0].astype(float)
    df['period_seconds'] = df['periodTime'].str.split(':').str[1].astype(float)
    df['total_seconds'] = (df['period'] - 1) * 1200 + df['period_minutes'] * 60 + df['period_seconds']
    df['time_remaining'] = 1200 - (df['period_minutes'] * 60 + df['period_seconds'])
    
    # Create context features from basic data
    print("⏳ Creating contextual features...")
    # Period context
    df['period_context'] = df['period'].apply(lambda x: 
        'first_period' if x == 1 else 
        'second_period' if x == 2 else 
        'third_period' if x == 3 else 
        'overtime')
    
    # Time pressure context
    df['time_pressure'] = df.apply(lambda row: 
        'final_minutes' if row['period'] == 3 and row['time_remaining'] <= 120 else
        'overtime' if row['period'] > 3 else
        'regular_time', axis=1)
    
    # Period timing context
    df['period_timing'] = df['period_minutes'].apply(lambda x:
        'period_start' if x <= 2 else
        'period_end' if x >= 18 else
        'period_middle')
    
    # Real pressure situations
    df['final_two_minutes'] = (
        (df['period'] == 3) & 
        (df['time_remaining'] <= 120)
    ).astype(int)
    df['overtime_shot'] = (df['period'] > 3).astype(int)
    df['period_start_shot'] = (df['period_minutes'] <= 2).astype(int)
    df['period_end_shot'] = (df['period_minutes'] >= 18).astype(int)
    
    # Spatial aggregation using absolute x (single net)
    print("⏳ Applying spatial aggregation...")
    x_bins = pd.cut(df['x_abs'], bins=20, labels=False)
    y_bins = pd.cut(df['y'], bins=10, labels=False)
    df['spatial_grid_x'] = x_bins
    df['spatial_grid_y'] = y_bins
    df['spatial_grid'] = x_bins * 10 + y_bins
    
    print("⏳ Encoding categorical features...")
    # Encode categorical context features
    le_period = LabelEncoder()
    le_time_pressure = LabelEncoder()
    le_period_timing = LabelEncoder()
    
    df['period_context_encoded'] = le_period.fit_transform(df['period_context'])
    df['time_pressure_encoded'] = le_time_pressure.fit_transform(df['time_pressure'])
    df['period_timing_encoded'] = le_period_timing.fit_transform(df['period_timing'])
    
    print(f"✅ Engineered contextual features from real data")
    print(f"📈 Goal rate: {df['is_goal'].mean():.1%}")
    print(f"🎯 Period distribution: {df['period_context'].value_counts().to_dict()}")
    print(f"⏰ Time pressure distribution: {df['time_pressure'].value_counts().to_dict()}")
    print(f"🏒 Using absolute x-axis: Single net analysis (right side only)")
    print(f"⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
    print(f"📊 Data density: Increased by using single net approach")
    
    return df

def prepare_contextual_clustering_data(df):
    """Prepare data for context-aware clustering."""
    print("\n🎯 PREPARING CONTEXTUAL CLUSTERING DATA")
    print("="*50)
    print("⏳ Selecting clustering features...")
    
    # Select features including real context
    clustering_features = [
        'distance_to_net', 'angle_to_net',
        'in_crease', 'in_slot', 'from_point', 'high_danger',
        'close_shot', 'medium_shot', 'long_shot',
        'sharp_angle', 'moderate_angle', 'straight_on',
        'spatial_grid_x', 'spatial_grid_y',
        'final_two_minutes', 'overtime_shot', 'period_start_shot', 'period_end_shot',
        'period_context_encoded', 'time_pressure_encoded', 'period_timing_encoded'
    ]
    
    # Retain x, y, and x_abs for plotting
    keep_cols = clustering_features + ['is_goal', 'x', 'y', 'x_abs']
    keep_cols = [col for col in keep_cols if col in df.columns or col in ['x', 'y', 'x_abs', 'is_goal']]
    df_clean = df[keep_cols].dropna()
    
    print(f"📊 Clean dataset: {len(df_clean):,} shots")
    print(f"🎯 Goals in clean data: {df_clean['is_goal'].sum():,} ({df_clean['is_goal'].mean():.1%})")
    
    # Prepare features
    X = df_clean[clustering_features].values
    y = df_clean['is_goal'].values
    
    # Scale features
    print("⏳ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} shots")
    
    return X_scaled, y, df_clean, scaler

def run_contextual_clustering(X, n_clusters=4):
    """Run context-aware clustering algorithms."""
    print("\n🔍 RUNNING CONTEXTUAL CLUSTERING ALGORITHMS")
    print("="*50)
    
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
            
            clustering_results[name] = {
                'labels': labels,
                'n_clusters': n_clusters_found,
                'silhouette_score': silhouette
            }
            
            print(f"✅ {name}: {n_clusters_found} clusters, silhouette: {silhouette:.3f}")
            
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
    
    return clustering_results

def analyze_contextual_clusters(X, y, df_clean, clustering_results):
    """Analyze context-aware clustering results."""
    print("\n📊 ANALYZING CONTEXTUAL CLUSTERS")
    print("="*50)
    
    # Find best algorithm
    best_algorithm = max(clustering_results.items(), 
                       key=lambda x: x[1]['silhouette_score'])[0]
    labels = clustering_results[best_algorithm]['labels']
    
    print(f"🏆 Best algorithm: {best_algorithm}")
    print(f"📈 Silhouette score: {clustering_results[best_algorithm]['silhouette_score']:.3f}")
    
    # Analyze each cluster
    cluster_analysis = {}
    danger_classification = {}
    
    print("⏳ Analyzing individual clusters...")
    for cluster_id in set(labels):
        cluster_mask = labels == cluster_id
        cluster_data = df_clean[cluster_mask]
        
        # Calculate cluster statistics
        size = len(cluster_data)
        goal_rate = cluster_data['is_goal'].mean()
        avg_distance = cluster_data['distance_to_net'].mean()
        avg_angle = cluster_data['angle_to_net'].mean()
        high_danger_rate = cluster_data['high_danger'].mean()
        final_two_rate = cluster_data['final_two_minutes'].mean()
        
        # Classify danger level
        if goal_rate > df_clean['is_goal'].mean() * 1.2 and avg_distance < 25:
            danger_type = 'High Danger'
        elif goal_rate < df_clean['is_goal'].mean() * 0.8 or avg_distance > 35:
            danger_type = 'Low Danger'
        else:
            danger_type = 'Medium Danger'
        
        cluster_analysis[cluster_id] = {
            'size': size,
            'goal_rate': goal_rate,
            'avg_distance': avg_distance,
            'avg_angle': avg_angle,
            'high_danger_rate': high_danger_rate,
            'final_two_rate': final_two_rate
        }
        
        danger_classification[cluster_id] = danger_type
        
        print(f"   Cluster {cluster_id}: {size:,} shots, {goal_rate:.1%} goal rate, {danger_type}")
    
    return cluster_analysis, danger_classification

def create_contextual_visualization(df_clean, labels, danger_classification):
    """Create context-aware clustering visualization with improved net design."""
    print(f"\n🏒 CREATING CONTEXTUAL CLUSTERING VISUALIZATION")
    print("="*50)
    print("⏳ Creating visualization...")
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: All shots colored by cluster (using absolute x-axis)
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels))))
    
    for cluster_id in set(labels):
        cluster_mask = labels == cluster_id
        cluster_data = df_clean[cluster_mask]
        
        ax1.scatter(cluster_data['x_abs'], cluster_data['y'], 
                   c=[colors[cluster_id]], alpha=0.6, s=20, 
                   label=f'Cluster {cluster_id}')
    
    # Draw rink outline (single net, right side only)
    ax1.plot([0, 100], [42.5, 42.5], 'k-', linewidth=2)
    ax1.plot([0, 100], [-42.5, -42.5], 'k-', linewidth=2)
    ax1.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=2)
    ax1.plot([100, 100], [-42.5, 42.5], 'k-', linewidth=2)
    
    # Draw realistic net (goal line 4 feet from back, 10 feet from end board)
    net_x = 89
    goal_line_x = 85
    
    # Draw goal line (red line)
    ax1.plot([goal_line_x, goal_line_x], [-42.5, 42.5], 'r-', linewidth=3, label='Goal Line')
    
    # Draw realistic net (rectangular with posts and crossbar)
    ax1.plot([net_x, net_x], [-3, 3], 'r-', linewidth=8, label='Net')
    ax1.plot([net_x-0.5, net_x+0.5], [-3, -3], 'r-', linewidth=6)
    ax1.plot([net_x-0.5, net_x+0.5], [3, 3], 'r-', linewidth=6)
    ax1.plot([net_x-0.5, net_x-0.5], [-3, 3], 'r-', linewidth=6)
    ax1.plot([net_x+0.5, net_x+0.5], [-3, 3], 'r-', linewidth=6)
    
    # Net mesh pattern
    for i in range(-2, 3):
        ax1.plot([net_x-0.4, net_x+0.4], [i, i], 'r-', linewidth=1, alpha=0.3)
    
    # Add center line and blue lines
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=25, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=75, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-50, 50)
    ax1.set_title('Contextual Clusters on Ice Rink (Absolute X-Axis, Single Net)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance from Center (feet)')
    ax1.set_ylabel('Y Coordinate (feet)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: High vs Low danger zones
    high_danger_mask = np.array([danger_classification.get(l, 'Low Danger') == 'High Danger' 
                                for l in labels])
    
    ax2.scatter(df_clean[~high_danger_mask]['x_abs'], df_clean[~high_danger_mask]['y'], 
               c='lightblue', alpha=0.6, s=20, label='Low Danger')
    ax2.scatter(df_clean[high_danger_mask]['x_abs'], df_clean[high_danger_mask]['y'], 
               c='red', alpha=0.6, s=20, label='High Danger')
    
    # Draw rink outline (single net, right side only)
    ax2.plot([0, 100], [42.5, 42.5], 'k-', linewidth=2)
    ax2.plot([0, 100], [-42.5, -42.5], 'k-', linewidth=2)
    ax2.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=2)
    ax2.plot([100, 100], [-42.5, 42.5], 'k-', linewidth=2)
    
    # Draw realistic net (same as above)
    ax2.plot([goal_line_x, goal_line_x], [-42.5, 42.5], 'r-', linewidth=3, label='Goal Line')
    ax2.plot([net_x, net_x], [-3, 3], 'r-', linewidth=8, label='Net')
    ax2.plot([net_x-0.5, net_x+0.5], [-3, -3], 'r-', linewidth=6)
    ax2.plot([net_x-0.5, net_x+0.5], [3, 3], 'r-', linewidth=6)
    ax2.plot([net_x-0.5, net_x-0.5], [-3, 3], 'r-', linewidth=6)
    ax2.plot([net_x+0.5, net_x+0.5], [-3, 3], 'r-', linewidth=6)
    
    # Net mesh pattern
    for i in range(-2, 3):
        ax2.plot([net_x-0.4, net_x+0.4], [i, i], 'r-', linewidth=1, alpha=0.3)
    
    # Add center line and blue lines
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axvline(x=25, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axvline(x=75, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-50, 50)
    ax2.set_title('Contextual High vs Low Danger Zones (Single Net)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance from Center (feet)')
    ax2.set_ylabel('Y Coordinate (feet)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Context features distribution
    context_features = ['final_two_minutes', 'overtime_shot', 'period_start_shot', 'period_end_shot']
    context_names = ['Final 2 Min', 'Overtime', 'Period Start', 'Period End']
    
    x_pos = np.arange(len(context_features))
    high_danger_rates = []
    low_danger_rates = []
    
    for feature in context_features:
        high_danger_rate = df_clean[high_danger_mask][feature].mean()
        low_danger_rate = df_clean[~high_danger_mask][feature].mean()
        high_danger_rates.append(high_danger_rate)
        low_danger_rates.append(low_danger_rate)
    
    width = 0.35
    ax3.bar(x_pos - width/2, high_danger_rates, width, label='High Danger', color='red', alpha=0.7)
    ax3.bar(x_pos + width/2, low_danger_rates, width, label='Low Danger', color='lightblue', alpha=0.7)
    
    ax3.set_xlabel('Context Features')
    ax3.set_ylabel('Feature Rate')
    ax3.set_title('Context Feature Distribution by Danger Level', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(context_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Goal rates by cluster
    cluster_ids = list(range(len(set(labels))))
    goal_rates = []
    
    for cluster_id in cluster_ids:
        cluster_mask = labels == cluster_id
        goal_rate = df_clean[cluster_mask]['is_goal'].mean()
        goal_rates.append(goal_rate)
    
    colors_clusters = ['red' if danger_classification.get(i, 'Low') == 'High Danger' else 'lightblue' 
                      for i in cluster_ids]
    
    bars = ax4.bar(cluster_ids, goal_rates, color=colors_clusters, alpha=0.7)
    ax4.axhline(y=df_clean['is_goal'].mean(), color='black', linestyle='--', 
                label=f'Overall Rate: {df_clean["is_goal"].mean():.1%}')
    
    # Add value labels on bars
    for bar, rate in zip(bars, goal_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Goal Rate')
    ax4.set_title('Goal Rates by Contextual Cluster', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('NHL Context-Aware Shot Clustering: Real Data Analysis (Most Recent Season, 25% Sample, Single Net)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('context_aware_clustering_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Contextual clustering visualization saved as 'context_aware_clustering_real_data.png'")

def main():
    """Main context-aware clustering analysis pipeline."""
    print("🏒 CONTEXT-AWARE NHL SHOT CLUSTERING ANALYSIS")
    print("="*70)
    print("Using real NHL data with actual game context features (most recent season, 25% sample)")
    print("⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
    
    # Load real contextual data with stratified sampling
    df = load_contextual_shot_data(sample_ratio=0.25)
    
    # Engineer contextual features
    df = engineer_contextual_features(df)
    
    # Prepare clustering data
    X, y, df_clean, scaler = prepare_contextual_clustering_data(df)
    
    # Run contextual clustering
    clustering_results = run_contextual_clustering(X, n_clusters=4)
    
    # Analyze clusters
    cluster_analysis, danger_classification = analyze_contextual_clusters(X, y, df_clean, clustering_results)
    
    # Create visualizations
    best_algorithm = max(clustering_results.items(), 
                       key=lambda x: x[1]['silhouette_score'])[0]
    labels = clustering_results[best_algorithm]['labels']
    create_contextual_visualization(df_clean, labels, danger_classification)
    
    print(f"\n🎉 CONTEXTUAL CLUSTERING ANALYSIS COMPLETE!")
    print("="*70)
    print(f"✅ Analyzed {len(df):,} real NHL shots with context (most recent season, 25% sample)")
    print(f"✅ Identified {len(cluster_analysis)} contextual shot clusters")
    print(f"✅ Classified high vs low danger zones with context")
    print(f"✅ Generated comprehensive visualizations")
    print(f"⚠️  Caveat: Analysis assumes ice symmetry - both sides treated equally")
    print(f"🏒 Single net approach: Increased data density for better clustering")

if __name__ == '__main__':
    main()
