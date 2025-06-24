#!/usr/bin/env python3
"""
Individual Corporate Charts for NHL Shot Clustering
Generates DoorDash-style corporate visualizations using real NHL data
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Add data pipeline to path
sys.path.append('../data_pipeline/src')

def load_real_shot_data(db_path='../data_pipeline/nhl_stats.db', sample_ratio=0.25):
    """Load real shot data from NHL database with stratified sampling from most recent season."""
    print("🏒 LOADING REAL NHL SHOT DATA")
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
    print(f"🎯 Goals: {(df['eventType'] == 'goal').sum():,}")
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

def engineer_spatial_features(df):
    """Engineer spatial features for clustering with absolute x-axis (single net)."""
    print("\n🔧 ENGINEERING SPATIAL FEATURES")
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
    # Time features
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    df['period_minutes'] = df['periodTime'].str.split(':').str[0].astype(float)
    df['period_seconds'] = df['periodTime'].str.split(':').str[1].astype(float)
    df['total_seconds'] = (df['period'] - 1) * 1200 + df['period_minutes'] * 60 + df['period_seconds']
    
    # Pressure situations
    df['final_two_minutes'] = (
        (df['period'] == 3) & 
        ((1200 - (df['period_minutes'] * 60 + df['period_seconds'])) <= 120)
    ).astype(int)
    df['overtime_shot'] = (df['period'] > 3).astype(int)
    
    # Spatial aggregation using absolute x (single net)
    print("⏳ Applying spatial aggregation...")
    x_bins = pd.cut(df['x_abs'], bins=20, labels=False)
    y_bins = pd.cut(df['y'], bins=10, labels=False)
    df['spatial_grid_x'] = x_bins
    df['spatial_grid_y'] = y_bins
    df['spatial_grid'] = x_bins * 10 + y_bins
    
    # Create danger zones based on distance and angle
    df['danger_zone'] = pd.cut(
        df['distance_to_net'], 
        bins=[0, 10, 20, 35, 50, 100], 
        labels=['Crease', 'High Danger', 'Medium Danger', 'Low Danger', 'Very Low Danger']
    )
    
    print(f"✅ Engineered spatial features from real data")
    print(f"📈 Goal rate: {df['is_goal'].mean():.1%}")
    print(f"🏒 Using absolute x-axis: Single net analysis (right side only)")
    print(f"⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
    print(f"📊 Data density: Increased by using single net approach")
    
    return df

def run_real_clustering(df):
    """Run clustering analysis on real NHL data."""
    print("\n🔍 RUNNING REAL NHL CLUSTERING")
    print("="*50)
    print("⏳ Preparing clustering data...")
    
    # Select features for clustering
    clustering_features = [
        'distance_to_net', 'angle_to_net',
        'in_crease', 'in_slot', 'from_point', 'high_danger',
        'close_shot', 'medium_shot', 'long_shot',
        'sharp_angle', 'moderate_angle', 'straight_on',
        'spatial_grid_x', 'spatial_grid_y',
        'final_two_minutes', 'overtime_shot'
    ]
    
    # Keep original coordinates for visualization
    keep_cols = clustering_features + ['is_goal', 'x', 'y', 'x_abs']
    df_clean = df[keep_cols].dropna()
    
    print(f"📊 Clean dataset: {len(df_clean):,} shots")
    print(f"🎯 Goals in clean data: {df_clean['is_goal'].sum():,} ({df_clean['is_goal'].mean():.1%})")
    
    # Prepare features for clustering
    X = df_clean[clustering_features].values
    y = df_clean['is_goal'].values
    
    # Scale features
    print("⏳ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} shots")
    
    # Run clustering
    print("⏳ Running DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    # Analyze clusters
    print("⏳ Analyzing clusters...")
    cluster_analysis = {}
    danger_classification = {}
    
    for cluster_id in set(labels):
        if cluster_id == -1:  # Skip noise
            continue
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
    
    print(f"✅ Identified {len(cluster_analysis)} clusters")
    
    return df_clean, labels, cluster_analysis, danger_classification

def create_ice_rink_chart(df_clean, labels, danger_classification, filename):
    """Create ice rink visualization with improved net design."""
    plt.figure(figsize=(12, 8))
    
    # Plot shots colored by cluster (using absolute x-axis)
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels))))
    
    for cluster_id in set(labels):
        cluster_mask = labels == cluster_id
        cluster_data = df_clean[cluster_mask]
        
        plt.scatter(cluster_data['x_abs'], cluster_data['y'], 
                   c=[colors[cluster_id]], alpha=0.6, s=15, 
                   label=f'Cluster {cluster_id}')
    
    # Draw rink outline
    plt.plot([0, 100], [42.5, 42.5], 'k-', linewidth=2)
    plt.plot([0, 100], [-42.5, -42.5], 'k-', linewidth=2)
    plt.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=2)
    plt.plot([100, 100], [-42.5, 42.5], 'k-', linewidth=2)
    
    # Draw improved nets
    # Left net
    plt.plot([0, 0], [-4, 4], 'r-', linewidth=6, label='Net')
    plt.plot([-2, 2], [-4, -4], 'r-', linewidth=4)
    plt.plot([-2, 2], [4, 4], 'r-', linewidth=4)
    plt.plot([-2, -2], [-4, 4], 'r-', linewidth=4)
    plt.plot([2, 2], [-4, 4], 'r-', linewidth=4)
    
    # Right net
    plt.plot([100, 100], [-4, 4], 'r-', linewidth=6)
    plt.plot([98, 102], [-4, -4], 'r-', linewidth=4)
    plt.plot([98, 102], [4, 4], 'r-', linewidth=4)
    plt.plot([98, 98], [-4, 4], 'r-', linewidth=4)
    plt.plot([102, 102], [-4, 4], 'r-', linewidth=4)
    
    # Add center line and blue lines
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=25, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    plt.axvline(x=75, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.xlim(-5, 105)
    plt.ylim(-50, 50)
    plt.title('NHL Shot Clusters on Ice Rink (Absolute X-Axis)', fontsize=16, fontweight='bold')
    plt.xlabel('Distance from Center (feet)')
    plt.ylabel('Y Coordinate (feet)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_danger_zone_chart(df_clean, labels, danger_classification, filename):
    """Create high vs low danger zones chart."""
    plt.figure(figsize=(12, 8))
    
    # High vs Low danger zones
    high_danger_mask = np.array([danger_classification.get(l, 'Low Danger') == 'High Danger' 
                                for l in labels])
    
    plt.scatter(df_clean[~high_danger_mask]['x_abs'], df_clean[~high_danger_mask]['y'], 
               c='lightblue', alpha=0.6, s=15, label='Low Danger')
    plt.scatter(df_clean[high_danger_mask]['x_abs'], df_clean[high_danger_mask]['y'], 
               c='red', alpha=0.6, s=15, label='High Danger')
    
    # Draw rink outline
    plt.plot([0, 100], [42.5, 42.5], 'k-', linewidth=2)
    plt.plot([0, 100], [-42.5, -42.5], 'k-', linewidth=2)
    plt.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=2)
    plt.plot([100, 100], [-42.5, 42.5], 'k-', linewidth=2)
    
    # Draw improved nets
    # Left net
    plt.plot([0, 0], [-4, 4], 'r-', linewidth=6, label='Net')
    plt.plot([-2, 2], [-4, -4], 'r-', linewidth=4)
    plt.plot([-2, 2], [4, 4], 'r-', linewidth=4)
    plt.plot([-2, -2], [-4, 4], 'r-', linewidth=4)
    plt.plot([2, 2], [-4, 4], 'r-', linewidth=4)
    
    # Right net
    plt.plot([100, 100], [-4, 4], 'r-', linewidth=6)
    plt.plot([98, 102], [-4, -4], 'r-', linewidth=4)
    plt.plot([98, 102], [4, 4], 'r-', linewidth=4)
    plt.plot([98, 98], [-4, 4], 'r-', linewidth=4)
    plt.plot([102, 102], [-4, 4], 'r-', linewidth=4)
    
    # Add center line and blue lines
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=25, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    plt.axvline(x=75, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.xlim(-5, 105)
    plt.ylim(-50, 50)
    plt.title('High vs Low Danger Zones (Absolute X-Axis)', fontsize=16, fontweight='bold')
    plt.xlabel('Distance from Center (feet)')
    plt.ylabel('Y Coordinate (feet)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_goal_rate_chart(cluster_analysis, danger_classification, filename):
    """Create goal rates by cluster chart."""
    plt.figure(figsize=(10, 6))
    
    cluster_ids = list(cluster_analysis.keys())
    goal_rates = [cluster_analysis[c]['goal_rate'] for c in cluster_ids]
    
    colors = ['red' if danger_classification.get(i, 'Low') == 'High Danger' else 'lightblue' 
              for i in cluster_ids]
    
    bars = plt.bar(cluster_ids, goal_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.axhline(y=np.mean(goal_rates), color='black', linestyle='--', 
                label=f'Overall Rate: {np.mean(goal_rates):.1%}')
    
    # Add value labels on bars
    for bar, rate in zip(bars, goal_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Goal Rate')
    plt.title('Goal Rates by Cluster', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_distance_analysis_chart(df_clean, filename):
    """Create distance analysis chart."""
    plt.figure(figsize=(10, 6))
    
    # Distance categories
    distance_cats = ['Close Shot', 'Medium Shot', 'Long Shot']
    distance_rates = [
        df_clean[df_clean['close_shot'] == 1]['is_goal'].mean(),
        df_clean[df_clean['medium_shot'] == 1]['is_goal'].mean(),
        df_clean[df_clean['long_shot'] == 1]['is_goal'].mean()
    ]
    
    colors = ['red', 'orange', 'lightblue']
    bars = plt.bar(distance_cats, distance_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.axhline(y=df_clean['is_goal'].mean(), color='black', linestyle='--', 
                label=f'Overall Rate: {df_clean["is_goal"].mean():.1%}')
    
    # Add value labels on bars
    for bar, rate in zip(bars, distance_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Shot Distance Category')
    plt.ylabel('Goal Rate')
    plt.title('Goal Rates by Shot Distance', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_angle_analysis_chart(df_clean, filename):
    """Create angle analysis chart."""
    plt.figure(figsize=(10, 6))
    
    # Angle categories
    angle_cats = ['Straight On', 'Moderate Angle', 'Sharp Angle']
    angle_rates = [
        df_clean[df_clean['straight_on'] == 1]['is_goal'].mean(),
        df_clean[df_clean['moderate_angle'] == 1]['is_goal'].mean(),
        df_clean[df_clean['sharp_angle'] == 1]['is_goal'].mean()
    ]
    
    colors = ['red', 'orange', 'lightblue']
    bars = plt.bar(angle_cats, angle_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.axhline(y=df_clean['is_goal'].mean(), color='black', linestyle='--', 
                label=f'Overall Rate: {df_clean["is_goal"].mean():.1%}')
    
    # Add value labels on bars
    for bar, rate in zip(bars, angle_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Shot Angle Category')
    plt.ylabel('Goal Rate')
    plt.title('Goal Rates by Shot Angle', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_time_pressure_chart(df_clean, filename):
    """Create time pressure analysis chart."""
    plt.figure(figsize=(10, 6))
    
    # Time pressure features
    time_features = ['Regular Time', 'Final 2 Minutes', 'Overtime']
    time_rates = [
        df_clean[df_clean['final_two_minutes'] == 0]['is_goal'].mean(),
        df_clean[df_clean['final_two_minutes'] == 1]['is_goal'].mean(),
        df_clean[df_clean['overtime_shot'] == 1]['is_goal'].mean()
    ]
    
    colors = ['lightblue', 'red', 'orange']
    bars = plt.bar(time_features, time_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.axhline(y=df_clean['is_goal'].mean(), color='black', linestyle='--', 
                label=f'Overall Rate: {df_clean["is_goal"].mean():.1%}')
    
    # Add value labels on bars
    for bar, rate in zip(bars, time_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Time Pressure Category')
    plt.ylabel('Goal Rate')
    plt.title('Goal Rates by Time Pressure', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_cluster_size_chart(cluster_analysis, filename):
    """Create cluster size distribution chart."""
    plt.figure(figsize=(10, 6))
    
    cluster_ids = list(cluster_analysis.keys())
    cluster_sizes = [cluster_analysis[c]['size'] for c in cluster_ids]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    bars = plt.bar(cluster_ids, cluster_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Shots')
    plt.title('Cluster Size Distribution', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def create_performance_summary_chart(df_clean, cluster_analysis, danger_classification, filename):
    """Create performance summary chart."""
    plt.figure(figsize=(12, 8))
    
    # Calculate performance metrics
    total_shots = len(df_clean)
    total_goals = df_clean['is_goal'].sum()
    overall_goal_rate = total_goals / total_shots
    
    # High vs low danger statistics
    high_danger_mask = df_clean['high_danger'] == 1
    high_danger_shots = df_clean[high_danger_mask]
    low_danger_shots = df_clean[~high_danger_mask]
    
    high_danger_goal_rate = high_danger_shots['is_goal'].mean()
    low_danger_goal_rate = low_danger_shots['is_goal'].mean()
    
    # Create summary metrics
    metrics = ['Overall', 'High Danger', 'Low Danger']
    rates = [overall_goal_rate, high_danger_goal_rate, low_danger_goal_rate]
    counts = [total_shots, len(high_danger_shots), len(low_danger_shots)]
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Goal rates
    colors = ['black', 'red', 'lightblue']
    bars1 = ax1.bar(metrics, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Goal Rate')
    ax1.set_title('Goal Rates by Danger Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Shot counts
    bars2 = ax2.bar(metrics, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Number of Shots')
    ax2.set_title('Shot Distribution by Danger Level', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('NHL Shot Analysis Performance Summary (25% Stratified Sample)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✅ Saved {filename}")

def main():
    """Main function to generate all corporate charts."""
    print("🏒 GENERATING CORPORATE NHL CLUSTERING CHARTS")
    print("="*70)
    print("Creating individual charts with real NHL data (most recent season, 25% sample)")
    print("⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
    
    # Set DoorDash-style styling
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'none'
    plt.rcParams['savefig.transparent'] = True
    
    # Load and process real data
    df = load_real_shot_data(sample_ratio=0.25)
    df = engineer_spatial_features(df)
    df_clean, labels, cluster_analysis, danger_classification = run_real_clustering(df)
    
    # Create output directory
    os.makedirs('corporate_charts', exist_ok=True)
    
    # Generate all charts
    print(f"\n📊 GENERATING CORPORATE CHARTS")
    print("="*50)
    
    create_ice_rink_chart(df_clean, labels, danger_classification, 'corporate_charts/01_ice_rink_clusters.png')
    create_danger_zone_chart(df_clean, labels, danger_classification, 'corporate_charts/02_danger_zones.png')
    create_goal_rate_chart(cluster_analysis, danger_classification, 'corporate_charts/03_goal_rates.png')
    create_distance_analysis_chart(df_clean, 'corporate_charts/04_distance_analysis.png')
    create_angle_analysis_chart(df_clean, 'corporate_charts/05_angle_analysis.png')
    create_time_pressure_chart(df_clean, 'corporate_charts/06_time_pressure.png')
    create_cluster_size_chart(cluster_analysis, 'corporate_charts/07_cluster_sizes.png')
    create_performance_summary_chart(df_clean, cluster_analysis, danger_classification, 'corporate_charts/08_performance_summary.png')
    
    print(f"\n🎉 CORPORATE CHARTS GENERATION COMPLETE!")
    print("="*70)
    print(f"✅ Generated 8 corporate charts with real NHL data")
    print(f"✅ Used most recent season data with 25% stratified sampling")
    print(f"✅ Applied absolute x-axis for ice symmetry (single net approach)")
    print(f"✅ Improved net visualization design")
    print(f"✅ All charts saved with transparent backgrounds")
    print(f"⚠️  Caveat: Analysis assumes ice symmetry - both sides treated equally")
    print(f"🏒 Single net approach: Increased data density for better clustering")

if __name__ == '__main__':
    main() 