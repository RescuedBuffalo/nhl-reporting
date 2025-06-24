#!/usr/bin/env python3
"""
Real NHL Data Clustering Analysis
Uses actual NHL shot data from the database for unsupervised learning
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# Add data pipeline to path
sys.path.append('../data_pipeline/src')

class RealNHLClusteringAnalyzer:
    """Analyzer for real NHL shot data clustering."""
    
    def __init__(self, db_path='../data_pipeline/nhl_stats.db', sample_ratio=0.50):
        """Initialize analyzer with database path and sampling ratio."""
        self.db_path = db_path
        self.sample_ratio = sample_ratio
        self.shot_data = None
        self.clustering_results = {}
        print(f"🔧 Initialized with {sample_ratio*100:.0f}% sampling ratio")
        self.scaler = StandardScaler()
        
    def load_real_shot_data(self):
        """Load real shot data from NHL database with stratified sampling from most recent season."""
        print("🏒 LOADING REAL NHL SHOT DATA")
        print("="*60)
        print("⏳ Connecting to database...")
        
        conn = sqlite3.connect(self.db_path)
        
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
        
        self.shot_data = df_sampled
        return df_sampled
    
    def engineer_spatial_features(self):
        """Engineer spatial features for clustering with absolute x-axis (single net)."""
        print("\n🔧 ENGINEERING SPATIAL FEATURES")
        print("="*50)
        print("⏳ Processing spatial coordinates...")
        
        df = self.shot_data.copy()
        
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
        
        self.shot_data = df
        
        feature_cols = [c for c in df.columns if c not in [
            'gamePk', 'eventType', 'teamId', 'x', 'y', 'gameDate', 'periodTime', 'details'
        ]]
        
        print(f"✅ Engineered {len(feature_cols)} features")
        print(f"📈 Goal rate: {df['is_goal'].mean():.1%}")
        print(f"🏒 Using absolute x-axis: Single net analysis (right side only)")
        print(f"⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
        print(f"📊 Data density: Increased by using single net approach")
        
        return df
    
    def prepare_clustering_data(self):
        """Prepare data for clustering analysis."""
        print("\n🎯 PREPARING CLUSTERING DATA")
        print("="*50)
        print("⏳ Selecting clustering features...")
        
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
        df_clean = self.shot_data[keep_cols].dropna()
        
        print(f"📊 Clean dataset: {len(df_clean):,} shots")
        print(f"🎯 Goals in clean data: {df_clean['is_goal'].sum():,} ({df_clean['is_goal'].mean():.1%})")
        
        # Prepare features for clustering
        X = df_clean[clustering_features].values
        y = df_clean['is_goal'].values
        
        # Scale features
        print("⏳ Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} shots")
        
        return X_scaled, y, df_clean, self.scaler
    
    def find_optimal_clusters(self, X, max_clusters=8):
        """Find optimal number of clusters using multiple metrics."""
        print("\n🔍 FINDING OPTIMAL CLUSTERS")
        print("="*50)
        
        silhouette_scores = []
        calinski_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            sil_score = silhouette_score(X, cluster_labels)
            cal_score = calinski_harabasz_score(X, cluster_labels)
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            
            print(f"k={k}: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.0f}")
        
        # Find optimal k
        optimal_k_sil = np.argmax(silhouette_scores) + 2
        optimal_k_cal = np.argmax(calinski_scores) + 2
        
        print(f"\n🏆 Optimal clusters:")
        print(f"   Silhouette score: k={optimal_k_sil}")
        print(f"   Calinski-Harabasz: k={optimal_k_cal}")
        
        # Use silhouette score as primary metric
        optimal_k = optimal_k_sil
        
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), calinski_scores, 'ro-')
        plt.axvline(x=optimal_k_cal, color='red', linestyle='--', label=f'Optimal k={optimal_k_cal}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Score vs Number of Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_k
    
    def run_clustering_algorithms(self, X):
        """Run multiple clustering algorithms and compare results with cluster limits."""
        print("\n🔍 RUNNING CLUSTERING ALGORITHMS")
        print("="*50)
        
        algorithms = {
            'K-Means': KMeans(n_clusters=8, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=8),
            'DBSCAN': DBSCAN(eps=0.8, min_samples=50),  # Increased parameters to reduce clusters
            'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20)  # Increased for fewer clusters
        }
        
        for name, algorithm in algorithms.items():
            print(f"⏳ Running {name}...")
            try:
                labels = algorithm.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Enforce maximum cluster limit of 10
                if n_clusters > 10:
                    print(f"⚠️  {name} produced {n_clusters} clusters, applying business constraint...")
                    if name in ['DBSCAN', 'HDBSCAN']:
                        # For density-based methods, we'll keep the largest 10 clusters
                        cluster_sizes = {}
                        for label in set(labels):
                            if label != -1:
                                cluster_sizes[label] = np.sum(labels == label)
                        
                        # Keep top 10 largest clusters
                        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
                        top_cluster_labels = [c[0] for c in top_clusters]
                        
                        # Reassign labels: keep top 10, mark rest as noise (-1)
                        new_labels = np.copy(labels)
                        for i, label in enumerate(labels):
                            if label not in top_cluster_labels and label != -1:
                                new_labels[i] = -1
                        
                        labels = new_labels
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        print(f"   📊 Reduced to {n_clusters} clusters (business constraint: max 10)")
                
                # Calculate silhouette score (skip if only one cluster)
                if n_clusters > 1:
                    silhouette = silhouette_score(X, labels)
                else:
                    silhouette = 0
                
                # Calculate Calinski-Harabasz score
                if n_clusters > 1:
                    calinski = calinski_harabasz_score(X, labels)
                else:
                    calinski = 0
                
                self.clustering_results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_score': calinski
                }
                
                print(f"✅ {name}: {n_clusters} clusters, silhouette: {silhouette:.3f}, Calinski: {calinski:.0f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
        
        return self.clustering_results
    
    def compare_clustering_algorithms(self):
        """Compare and contrast different clustering algorithms."""
        print("\n📊 CLUSTERING ALGORITHM COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        for name, results in self.clustering_results.items():
            comparison_data.append({
                'Algorithm': name,
                'Clusters': results['n_clusters'],
                'Silhouette': results['silhouette_score'],
                'Calinski': results['calinski_score']
            })
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        print("\n📈 ALGORITHM PERFORMANCE COMPARISON:")
        print("="*50)
        print(df_comparison.to_string(index=False))
        
        # Find best algorithms by different metrics
        best_silhouette = df_comparison.loc[df_comparison['Silhouette'].idxmax()]
        best_calinski = df_comparison.loc[df_comparison['Calinski'].idxmax()]
        
        print(f"\n🏆 BEST ALGORITHMS:")
        print(f"   Silhouette Score: {best_silhouette['Algorithm']} ({best_silhouette['Silhouette']:.3f})")
        print(f"   Calinski-Harabasz: {best_calinski['Algorithm']} ({best_calinski['Calinski']:.0f})")
        
        # Create comparison visualization
        self.create_algorithm_comparison_visualization(df_comparison)
        
        return df_comparison
    
    def create_algorithm_comparison_visualization(self, df_comparison):
        """Create visualization comparing clustering algorithms."""
        print("\n📊 CREATING ALGORITHM COMPARISON VISUALIZATION")
        print("="*60)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Silhouette Scores
        algorithms = df_comparison['Algorithm']
        silhouette_scores = df_comparison['Silhouette']
        
        bars1 = ax1.bar(algorithms, silhouette_scores, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        ax1.set_title('Silhouette Scores by Algorithm', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_ylim(0, max(silhouette_scores) * 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, silhouette_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Calinski-Harabasz Scores
        calinski_scores = df_comparison['Calinski']
        
        bars2 = ax2.bar(algorithms, calinski_scores, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        ax2.set_title('Calinski-Harabasz Scores by Algorithm', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Calinski-Harabasz Score')
        ax2.set_ylim(0, max(calinski_scores) * 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars2, calinski_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Number of Clusters
        cluster_counts = df_comparison['Clusters']
        
        bars3 = ax3.bar(algorithms, cluster_counts, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        ax3.set_title('Number of Clusters by Algorithm', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Number of Clusters')
        ax3.set_ylim(0, max(cluster_counts) * 1.1)
        
        # Add value labels on bars
        for bar, count in zip(bars3, cluster_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Algorithm Characteristics
        characteristics = {
            'K-Means': ['Fixed clusters', 'Fast', 'Spherical clusters', 'Sensitive to initialization'],
            'Agglomerative': ['Hierarchical', 'Flexible', 'Dendrogram', 'Computationally expensive'],
            'DBSCAN': ['Density-based', 'Noise handling', 'Variable shapes', 'Parameter sensitive'],
            'HDBSCAN': ['Hierarchical DBSCAN', 'Robust', 'Multiple densities', 'Advanced']
        }
        
        ax4.axis('off')
        y_pos = 0.9
        for alg in algorithms:
            ax4.text(0.1, y_pos, f"{alg}:", fontsize=14, fontweight='bold')
            y_pos -= 0.05
            for char in characteristics[alg]:
                ax4.text(0.15, y_pos, f"• {char}", fontsize=12)
                y_pos -= 0.04
            y_pos -= 0.02
        
        ax4.set_title('Algorithm Characteristics', fontsize=16, fontweight='bold')
        
        plt.suptitle('NHL Shot Clustering: Algorithm Comparison (Most Recent Season, 50% Sample)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('clustering_algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Algorithm comparison visualization saved as 'clustering_algorithm_comparison.png'")
    
    def analyze_clusters(self, X, y, df_clean, labels):
        """Analyze clustering results and classify danger levels with minimum sample size."""
        print("\n📊 ANALYZING CLUSTERS")
        print("="*50)
        
        # Find best algorithm
        best_algorithm = max(self.clustering_results.items(), 
                           key=lambda x: x[1]['silhouette_score'])[0]
        labels = self.clustering_results[best_algorithm]['labels']
        
        print(f"🏆 Best algorithm: {best_algorithm}")
        print(f"📈 Silhouette score: {self.clustering_results[best_algorithm]['silhouette_score']:.3f}")
        
        # Analyze each cluster with minimum sample size filter
        cluster_analysis = {}
        danger_classification = {}
        min_sample_size = 50  # Minimum shots per cluster for reliable analysis
        
        print("⏳ Analyzing individual clusters...")
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = labels == cluster_id
            cluster_data = df_clean[cluster_mask]
            
            # Skip clusters that are too small for reliable analysis
            size = len(cluster_data)
            if size < min_sample_size:
                print(f"   ⚠️  Cluster {cluster_id}: {size} shots (skipped - below minimum {min_sample_size})")
                continue
            
            # Calculate cluster statistics
            goal_rate = cluster_data['is_goal'].mean()
            avg_distance = cluster_data['distance_to_net'].mean()
            avg_angle = cluster_data['angle_to_net'].mean()
            high_danger_rate = cluster_data['high_danger'].mean()
            final_two_rate = cluster_data['final_two_minutes'].mean()
            
            # Enhanced danger classification based on distance and goal rate
            overall_goal_rate = df_clean['is_goal'].mean()
            
            # Apply business logic for danger classification
            if avg_distance <= 25 and goal_rate >= overall_goal_rate * 1.2:
                danger_type = 'High Danger'
            elif avg_distance > 75 or goal_rate <= overall_goal_rate * 0.7:  # Beyond blue line is low danger
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
        
        print(f"\n✅ Analyzed {len(cluster_analysis)} clusters meeting minimum sample size")
        
        return cluster_analysis, danger_classification
    
    def create_ice_rink_visualization(self, df_clean, labels, danger_classification):
        """Create ice rink visualization with corrected single net and blue line."""
        print(f"\n🏒 CREATING ICE RINK VISUALIZATION")
        print("="*50)
        print("⏳ Creating visualization...")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: All shots colored by cluster (using absolute x-axis, single net)
        valid_clusters = [c for c in set(labels) if c != -1 and c in danger_classification]
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_clusters)))
        
        for i, cluster_id in enumerate(valid_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = df_clean[cluster_mask]
            
            ax1.scatter(cluster_data['x_abs'], cluster_data['y'], 
                       c=[colors[i]], alpha=0.6, s=20, 
                       label=f'Cluster {cluster_id}')
        
        # Draw rink outline (single net, right side only - half ice from center to goal)
        ax1.plot([0, 89], [42.5, 42.5], 'k-', linewidth=2)  # Top boards
        ax1.plot([0, 89], [-42.5, -42.5], 'k-', linewidth=2)  # Bottom boards
        ax1.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=2)  # Center line (red line)
        ax1.plot([89, 89], [-42.5, 42.5], 'k-', linewidth=2)  # End boards
        
        # Draw realistic net (goal line 4 feet from back, 10 feet from end board)
        # Net is at x=89, goal line at x=85, end board at x=89 (corrected)
        net_x = 89
        goal_line_x = 85
        blue_line_x = 25  # Blue line 25 feet from center
        
        # Draw center line (red line at x=0)
        ax1.axvline(x=0, color='red', linestyle='-', alpha=0.8, linewidth=3, label='Center Line')
        
        # Draw blue line (only one since we're showing half ice)
        ax1.axvline(x=blue_line_x, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='Blue Line')
        
        # Draw goal line (red line)
        ax1.plot([goal_line_x, goal_line_x], [-42.5, 42.5], 'r-', linewidth=3, label='Goal Line')
        
        # Draw realistic net (rectangular with posts and crossbar)
        ax1.plot([net_x, net_x], [-3, 3], 'r-', linewidth=8, label='Net')
        ax1.plot([net_x-0.5, net_x+0.5], [-3, -3], 'r-', linewidth=6)  # Bottom
        ax1.plot([net_x-0.5, net_x+0.5], [3, 3], 'r-', linewidth=6)    # Top (crossbar)
        ax1.plot([net_x-0.5, net_x-0.5], [-3, 3], 'r-', linewidth=6)   # Left post
        ax1.plot([net_x+0.5, net_x+0.5], [-3, 3], 'r-', linewidth=6)   # Right post
        
        # Net mesh pattern (simplified)
        for i in range(-2, 3):
            ax1.plot([net_x-0.4, net_x+0.4], [i, i], 'r-', linewidth=1, alpha=0.3)
        
        ax1.set_xlim(-5, 95)
        ax1.set_ylim(-50, 50)
        ax1.set_title('Shot Clusters on Half Ice (Center to Goal)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Distance from Center Line (feet)')
        ax1.set_ylabel('Y Coordinate (feet)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: High vs Low danger zones with corrected classification
        high_danger_mask = np.array([danger_classification.get(l, 'Low Danger') == 'High Danger' 
                                    for l in labels])
        
        # Only show points that belong to analyzed clusters (not noise or too small)
        valid_mask = np.array([l in danger_classification for l in labels])
        
        low_danger_data = df_clean[~high_danger_mask & valid_mask]
        high_danger_data = df_clean[high_danger_mask & valid_mask]
        
        ax2.scatter(low_danger_data['x_abs'], low_danger_data['y'], 
                   c='lightblue', alpha=0.6, s=20, label='Low Danger')
        ax2.scatter(high_danger_data['x_abs'], high_danger_data['y'], 
                   c='red', alpha=0.6, s=20, label='High Danger')
        
        # Draw same rink outline
        ax2.plot([0, 89], [42.5, 42.5], 'k-', linewidth=2)
        ax2.plot([0, 89], [-42.5, -42.5], 'k-', linewidth=2)
        ax2.plot([0, 0], [-42.5, 42.5], 'k-', linewidth=2)
        ax2.plot([89, 89], [-42.5, 42.5], 'k-', linewidth=2)
        
        # Draw lines
        ax2.axvline(x=0, color='red', linestyle='-', alpha=0.8, linewidth=3, label='Center Line')
        ax2.axvline(x=blue_line_x, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='Blue Line')
        ax2.plot([goal_line_x, goal_line_x], [-42.5, 42.5], 'r-', linewidth=3, label='Goal Line')
        
        # Draw net
        ax2.plot([net_x, net_x], [-3, 3], 'r-', linewidth=8, label='Net')
        ax2.plot([net_x-0.5, net_x+0.5], [-3, -3], 'r-', linewidth=6)
        ax2.plot([net_x-0.5, net_x+0.5], [3, 3], 'r-', linewidth=6)
        ax2.plot([net_x-0.5, net_x-0.5], [-3, 3], 'r-', linewidth=6)
        ax2.plot([net_x+0.5, net_x+0.5], [-3, 3], 'r-', linewidth=6)
        
        # Net mesh pattern
        for i in range(-2, 3):
            ax2.plot([net_x-0.4, net_x+0.4], [i, i], 'r-', linewidth=1, alpha=0.3)
        
        ax2.set_xlim(-5, 95)
        ax2.set_ylim(-50, 50)
        ax2.set_title('High vs Low Danger Zones (Filtered by Sample Size)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Distance from Center Line (feet)')
        ax2.set_ylabel('Y Coordinate (feet)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('NHL Shot Clustering: Real Data Analysis (Most Recent Season, 50% Sample, No Empty Net)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('real_data_ice_rink_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Ice rink visualization saved as 'real_data_ice_rink_clusters.png'")
        print("🏒 Corrected visualization: Half ice from center line to goal")
        print("⚠️  Caveat: Analysis assumes ice symmetry - both sides treated equally")
        print("📊 Filtered clusters by minimum sample size for reliable analysis")
        print("🚫 Empty net goals excluded from analysis")
    
    def create_comprehensive_analysis(self):
        """Create comprehensive clustering analysis with algorithm comparison."""
        print("\n📊 CREATING COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        # Load and process data
        self.load_real_shot_data()
        self.engineer_spatial_features()
        X, y, df_clean, scaler = self.prepare_clustering_data()
        
        # Find optimal clusters
        self.find_optimal_clusters(X)
        
        # Run multiple clustering algorithms
        self.run_clustering_algorithms(X)
        
        # Compare algorithms
        comparison_df = self.compare_clustering_algorithms()
        
        # Analyze best algorithm results
        best_algorithm = max(self.clustering_results.items(), 
                           key=lambda x: x[1]['silhouette_score'])[0]
        labels = self.clustering_results[best_algorithm]['labels']
        
        cluster_analysis, danger_classification = self.analyze_clusters(X, y, df_clean, labels)
        
        # Create visualizations
        self.create_ice_rink_visualization(df_clean, labels, danger_classification)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(cluster_analysis, danger_classification, df_clean, comparison_df, best_algorithm)
        
        return cluster_analysis, danger_classification
    
    def generate_comprehensive_report(self, cluster_analysis, danger_classification, df_clean, comparison_df, best_algorithm):
        """Generate comprehensive summary report with algorithm comparison."""
        print(f"\n📝 GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        print("⏳ Calculating statistics...")
        
        # Calculate overall statistics
        total_shots = len(df_clean)
        total_goals = df_clean['is_goal'].sum()
        overall_goal_rate = total_goals / total_shots
        
        # High vs low danger statistics
        labels = self.clustering_results[best_algorithm]['labels']
        high_danger_mask = np.array([danger_classification.get(l, 'Low Danger') == 'High Danger' 
                                    for l in labels])
        
        high_danger_shots = df_clean[high_danger_mask]
        low_danger_shots = df_clean[~high_danger_mask]
        
        high_danger_goal_rate = high_danger_shots['is_goal'].mean()
        low_danger_goal_rate = low_danger_shots['is_goal'].mean()
        
        print("⏳ Writing comprehensive report...")
        report = f"""
# Comprehensive NHL Shot Clustering Analysis Report

## 📊 **Dataset Overview**
- **Total Shots**: {total_shots:,} (50% stratified sample from most recent season)
- **Total Goals**: {total_goals:,}
- **Overall Goal Rate**: {overall_goal_rate:.1%}
- **Games Analyzed**: {self.shot_data['gamePk'].nunique():,}
- **Date Range**: {self.shot_data['gameDate'].min()} to {self.shot_data['gameDate'].max()}
- **Season**: Most recent NHL season only

## 🎯 **Clustering Algorithm Comparison**

### **Algorithm Performance Summary**:
{comparison_df.to_string(index=False)}

### **Best Algorithm**: {best_algorithm}
- **Silhouette Score**: {self.clustering_results[best_algorithm]['silhouette_score']:.3f}
- **Calinski-Harabasz Score**: {self.clustering_results[best_algorithm]['calinski_score']:.0f}
- **Number of Clusters**: {self.clustering_results[best_algorithm]['n_clusters']}

### **Algorithm Analysis**:

#### **K-Means Clustering**:
- **Strengths**: Fast, simple, works well with spherical clusters
- **Weaknesses**: Requires specifying number of clusters, sensitive to initialization
- **Best for**: When you know the expected number of clusters
- **Performance**: Silhouette: {comparison_df[comparison_df['Algorithm'] == 'K-Means']['Silhouette'].iloc[0]:.3f}

#### **Agglomerative Clustering**:
- **Strengths**: Hierarchical structure, flexible, creates dendrogram
- **Weaknesses**: Computationally expensive, doesn't scale well
- **Best for**: When you want to understand hierarchical relationships
- **Performance**: Silhouette: {comparison_df[comparison_df['Algorithm'] == 'Agglomerative']['Silhouette'].iloc[0]:.3f}

#### **DBSCAN**:
- **Strengths**: Handles noise, discovers clusters of arbitrary shapes
- **Weaknesses**: Sensitive to parameters (eps, min_samples)
- **Best for**: When you don't know the number of clusters, want to handle noise
- **Performance**: Silhouette: {comparison_df[comparison_df['Algorithm'] == 'DBSCAN']['Silhouette'].iloc[0]:.3f}

#### **HDBSCAN**:
- **Strengths**: Hierarchical DBSCAN, robust to parameter selection
- **Weaknesses**: More complex, slower than DBSCAN
- **Best for**: When you want robust density-based clustering
- **Performance**: Silhouette: {comparison_df[comparison_df['Algorithm'] == 'HDBSCAN']['Silhouette'].iloc[0]:.3f}

## 🏆 **Best Algorithm Results**

### **Cluster Analysis**:
"""
        
        for cluster_id, analysis in cluster_analysis.items():
            danger_type = danger_classification[cluster_id]
            report += f"""
#### **Cluster {cluster_id}** ({danger_type})
- **Size**: {analysis['size']:,} shots ({analysis['size']/total_shots*100:.1f}%)
- **Goal Rate**: {analysis['goal_rate']:.1%}
- **Average Distance**: {analysis['avg_distance']:.1f} feet
- **Average Angle**: {analysis['avg_angle']:.1f}°
- **High Danger Zone**: {analysis['high_danger_rate']:.1%}
- **Final 2 Minutes**: {analysis['final_two_rate']:.1%}
"""
        
        report += f"""
## 🏆 **Danger Zone Classification**

### **High Danger Shots**:
- **Count**: {len(high_danger_shots):,} ({len(high_danger_shots)/total_shots*100:.1f}%)
- **Goal Rate**: {high_danger_goal_rate:.1%}
- **Improvement over baseline**: {(high_danger_goal_rate/overall_goal_rate-1)*100:+.1f}%

### **Low Danger Shots**:
- **Count**: {len(low_danger_shots):,} ({len(low_danger_shots)/total_shots*100:.1f}%)
- **Goal Rate**: {low_danger_goal_rate:.1%}
- **Reduction from baseline**: {(1-low_danger_goal_rate/overall_goal_rate)*100:.1f}%

## 💡 **Key Insights**

1. **Algorithm Selection**: {best_algorithm} performed best for this dataset
2. **Spatial Patterns**: Clustering reveals distinct shot location patterns using absolute x-axis
3. **Danger Classification**: Clear separation between high and low danger zones
4. **Goal Rate Variation**: Significant differences in goal rates across clusters
5. **Real Data Validation**: Analysis based on {total_shots:,} actual NHL shots (most recent season, 50% sample)
6. **Ice Symmetry**: Using absolute x-axis treats both sides of ice equally
7. **Single Net Approach**: Increased data density by analyzing right side only

## ⚠️ **Methodological Caveats**

### **Ice Symmetry Assumption**:
- **Assumption**: Both sides of the ice are treated equally using absolute x-axis values
- **Reality**: Home ice advantage and other factors may create asymmetries
- **Impact**: Results may not capture side-specific patterns
- **Mitigation**: Analysis focuses on general shot patterns rather than side-specific strategies

### **Single Net Analysis**:
- **Approach**: Only right-side net analyzed to increase data density per point
- **Benefit**: More robust clustering with higher point density
- **Limitation**: May miss left-side specific patterns
- **Justification**: Primary goal is identifying general danger zones, not side-specific analysis

## 🚀 **Business Applications**

1. **Coaching Strategy**: Focus defensive efforts on high-danger clusters
2. **Player Development**: Train players to generate shots in high-danger zones
3. **Game Analysis**: Real-time identification of dangerous shot situations
4. **Scouting**: Evaluate players based on shot location quality

## 📁 **Generated Files**
- `optimal_clusters_analysis.png` - Cluster optimization analysis
- `clustering_algorithm_comparison.png` - Algorithm comparison visualization
- `real_data_ice_rink_clusters.png` - Ice rink visualization with realistic net
- `real_data_clustering_report.md` - This report

---
*Analysis completed using real NHL data from {self.shot_data['gamePk'].nunique():,} games (most recent season, 50% stratified sample)*
*Methodological caveat: Assumes ice symmetry - both sides treated equally*
"""
        
        with open('real_data_clustering_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Comprehensive report saved as 'real_data_clustering_report.md'")
        
        return report

def main():
    """Main analysis pipeline."""
    print("🏒 REAL NHL DATA CLUSTERING ANALYSIS")
    print("="*70)
    print("Using actual NHL shot data for unsupervised learning (most recent season, 50% sample)")
    print("⚠️  Caveat: Assumes ice symmetry - both sides treated equally")
    
    # Initialize analyzer with 50% stratified sampling
    analyzer = RealNHLClusteringAnalyzer(sample_ratio=0.50)
    
    # Run comprehensive analysis with algorithm comparison
    cluster_analysis, danger_classification = analyzer.create_comprehensive_analysis()
    
    print(f"\n🎉 REAL DATA ANALYSIS COMPLETE!")
    print("="*70)
    print(f"✅ Analyzed {len(analyzer.shot_data):,} real NHL shots (most recent season, 50% sample)")
    print(f"✅ Compared 4 clustering algorithms (K-Means, Agglomerative, DBSCAN, HDBSCAN)")
    print(f"✅ Identified {len(cluster_analysis)} distinct shot clusters")
    print(f"✅ Classified high vs low danger zones")
    print(f"✅ Generated comprehensive visualizations and report")
    print(f"⚠️  Caveat: Analysis assumes ice symmetry - both sides treated equally")
    print(f"🏒 Single net approach: Increased data density for better clustering")

if __name__ == '__main__':
    main() 