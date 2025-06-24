#!/usr/bin/env python3
"""
NHL Shot Clustering Analysis - Parameter Sensitivity Analysis

Implementing solutions to address parameter sensitivity issues:
1. Grid search parameter optimization
2. Domain knowledge integration
3. Stability analysis across time periods
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Corporate styling
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_shot_data(db_path='../data_pipeline/nhl_stats.db'):
    """Load shot and goal data from SQLite database"""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        x, y, 
        eventType,
        CASE WHEN eventType = 'goal' THEN 1 ELSE 0 END as is_goal,
        period,
        teamId,
        playerId
    FROM events 
    WHERE eventType IN ('shot-on-goal', 'goal') 
    AND x IS NOT NULL AND y IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def create_spatial_bins(df, x_bins=20, y_bins=15):
    """Create spatial bins and calculate goal probabilities"""
    
    # Create bins
    x_edges = np.linspace(df['x'].min(), df['x'].max(), x_bins + 1)
    y_edges = np.linspace(df['y'].min(), df['y'].max(), y_bins + 1)
    
    # Assign bins to each shot
    df['x_bin'] = pd.cut(df['x'], bins=x_edges, labels=False)
    df['y_bin'] = pd.cut(df['y'], bins=y_edges, labels=False)
    
    # Calculate bin centers for visualization
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Aggregate by bins
    bin_stats = df.groupby(['x_bin', 'y_bin']).agg({
        'is_goal': ['count', 'sum', 'mean'],
        'x': 'mean',
        'y': 'mean'
    }).reset_index()
    
    # Flatten column names
    bin_stats.columns = ['x_bin', 'y_bin', 'shot_count', 'goal_count', 'goal_rate', 'x_center', 'y_center']
    
    # Filter out bins with too few shots (less than 5)
    bin_stats = bin_stats[bin_stats['shot_count'] >= 5]
    
    return bin_stats, x_centers, y_centers

def train_clustering_model(bin_stats, n_clusters=6):
    """Train clustering model and return performance metrics"""
    if len(bin_stats) < n_clusters:
        return 0.0, None, None
    
    scaler = StandardScaler()
    agg_features = bin_stats[['x_center', 'y_center', 'goal_rate']].values
    agg_features_scaled = scaler.fit_transform(agg_features)
    
    # Train K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(agg_features_scaled)
    
    # Calculate silhouette score
    silhouette = silhouette_score(agg_features_scaled, cluster_labels)
    
    return silhouette, kmeans, scaler

def solution_2_grid_search_optimization():
    """Solution 2.1: Comprehensive parameter grid search"""
    
    print("🔍 Grid Search Parameter Optimization")
    print("=" * 50)
    
    # Load data
    shot_data = load_shot_data()
    
    # Define parameter grid
    bin_configs = [
        (10, 10), (15, 10), (20, 10), (25, 10), (30, 10),
        (10, 15), (15, 15), (20, 15), (25, 15), (30, 15),
        (10, 20), (15, 20), (20, 20), (25, 20), (30, 20)
    ]
    
    cluster_configs = [3, 4, 5, 6, 7, 8, 9, 10]
    
    results = []
    total_combinations = len(bin_configs) * len(cluster_configs)
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, (x_bins, y_bins) in enumerate(bin_configs):
        for n_clusters in cluster_configs:
            try:
                # Create spatial bins
                bin_stats, _, _ = create_spatial_bins(shot_data, x_bins, y_bins)
                
                # Train clustering model
                silhouette_score_val, _, _ = train_clustering_model(bin_stats, n_clusters)
                
                results.append({
                    'x_bins': x_bins,
                    'y_bins': y_bins,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_score_val,
                    'total_bins': x_bins * y_bins,
                    'bin_resolution_x': 200 / x_bins,  # NHL rink is ~200' wide
                    'bin_resolution_y': 85 / y_bins    # NHL rink is ~85' wide
                })
                
                print(f"  Progress: {len(results)}/{total_combinations} - "
                      f"({x_bins}x{y_bins}, {n_clusters} clusters): {silhouette_score_val:.4f}")
                
            except Exception as e:
                print(f"  Error with ({x_bins}x{y_bins}, {n_clusters} clusters): {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal parameters
    optimal_idx = results_df['silhouette_score'].idxmax()
    optimal_params = results_df.loc[optimal_idx]
    
    print(f"\nGrid Search Results:")
    print(f"  Total configurations tested: {len(results_df)}")
    print(f"  Best silhouette score: {optimal_params['silhouette_score']:.4f}")
    print(f"  Optimal parameters:")
    print(f"    X bins: {optimal_params['x_bins']}")
    print(f"    Y bins: {optimal_params['y_bins']}")
    print(f"    Clusters: {optimal_params['n_clusters']}")
    print(f"    Total bins: {optimal_params['total_bins']}")
    print(f"    Resolution: {optimal_params['bin_resolution_x']:.1f}' x {optimal_params['bin_resolution_y']:.1f}'")
    
    # Create parameter sensitivity visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Silhouette score by x_bins
    x_bin_performance = results_df.groupby('x_bins')['silhouette_score'].mean()
    ax1.plot(x_bin_performance.index, x_bin_performance.values, marker='o', linewidth=2)
    ax1.set_xlabel('X Bins', fontweight='bold')
    ax1.set_ylabel('Mean Silhouette Score', fontweight='bold')
    ax1.set_title('Performance by X Bins', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Silhouette score by y_bins
    y_bin_performance = results_df.groupby('y_bins')['silhouette_score'].mean()
    ax2.plot(y_bin_performance.index, y_bin_performance.values, marker='s', linewidth=2, color='orange')
    ax2.set_xlabel('Y Bins', fontweight='bold')
    ax2.set_ylabel('Mean Silhouette Score', fontweight='bold')
    ax2.set_title('Performance by Y Bins', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Silhouette score by n_clusters
    cluster_performance = results_df.groupby('n_clusters')['silhouette_score'].mean()
    ax3.plot(cluster_performance.index, cluster_performance.values, marker='^', linewidth=2, color='green')
    ax3.set_xlabel('Number of Clusters', fontweight='bold')
    ax3.set_ylabel('Mean Silhouette Score', fontweight='bold')
    ax3.set_title('Performance by Number of Clusters', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of x_bins vs y_bins
    pivot_table = results_df.groupby(['x_bins', 'y_bins'])['silhouette_score'].mean().unstack()
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_xlabel('Y Bins', fontweight='bold')
    ax4.set_ylabel('X Bins', fontweight='bold')
    ax4.set_title('Silhouette Score Heatmap', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_grid_search.png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()
    
    return results_df, optimal_params

def solution_2_domain_knowledge_integration():
    """Solution 2.2: Integrate hockey domain knowledge for parameter selection"""
    
    print("\n🏒 Domain Knowledge Integration")
    print("=" * 50)
    
    # NHL rink dimensions and standard zones
    rink_dimensions = {
        'width': 200,  # feet
        'length': 85,  # feet
        'goal_width': 6,  # feet
        'goal_depth': 4  # feet
    }
    
    # Standard hockey zones based on domain knowledge
    zone_definitions = {
        'slot': {
            'description': 'High-danger area in front of net',
            'x_range': (-10, 10),
            'y_range': (-10, 10),
            'optimal_resolution': 5  # 5' resolution for precise slot analysis
        },
        'point': {
            'description': 'Blue line area for point shots',
            'x_range': (-20, 20),
            'y_range': (-20, 20),
            'optimal_resolution': 10  # 10' resolution for point analysis
        },
        'circle': {
            'description': 'Face-off circle areas',
            'x_range': (-30, 30),
            'y_range': (-30, 30),
            'optimal_resolution': 15  # 15' resolution for circle analysis
        },
        'perimeter': {
            'description': 'Outer areas of offensive zone',
            'x_range': (-100, 100),
            'y_range': (-42.5, 42.5),
            'optimal_resolution': 20  # 20' resolution for perimeter
        }
    }
    
    print("Hockey Domain Knowledge Analysis:")
    for zone, details in zone_definitions.items():
        print(f"  {zone.title()} Zone:")
        print(f"    Description: {details['description']}")
        print(f"    Range: {details['x_range']} x {details['y_range']}")
        print(f"    Optimal Resolution: {details['optimal_resolution']}'")
    
    # Calculate domain-informed optimal parameters
    optimal_x_bins = int(rink_dimensions['width'] / 10)  # 10' resolution
    optimal_y_bins = int(rink_dimensions['length'] / 10)  # 10' resolution
    
    print(f"\nDomain-Informed Parameter Selection:")
    print(f"  NHL Rink Dimensions: {rink_dimensions['width']}' x {rink_dimensions['length']}'")
    print(f"  Optimal X Bins: {optimal_x_bins} (resolution: {rink_dimensions['width']/optimal_x_bins:.1f}')")
    print(f"  Optimal Y Bins: {optimal_y_bins} (resolution: {rink_dimensions['length']/optimal_y_bins:.1f}')")
    print(f"  Total Spatial Bins: {optimal_x_bins * optimal_y_bins}")
    
    # Validate against grid search results
    print(f"\nValidation Against Grid Search:")
    print(f"  Domain-informed X bins: {optimal_x_bins}")
    print(f"  Domain-informed Y bins: {optimal_y_bins}")
    
    # Create domain knowledge visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw rink outline
    rink_corners = np.array([[-100, -42.5], [100, -42.5], [100, 42.5], [-100, 42.5], [-100, -42.5]])
    ax.plot(rink_corners[:, 0], rink_corners[:, 1], 'k-', linewidth=3, label='Rink Boundary')
    
    # Draw zones with different colors
    colors = ['red', 'orange', 'yellow', 'lightblue']
    for i, (zone, details) in enumerate(zone_definitions.items()):
        x_min, x_max = details['x_range']
        y_min, y_max = details['y_range']
        
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                           fill=True, alpha=0.3, color=colors[i], label=f'{zone.title()} Zone')
        ax.add_patch(rect)
    
    # Add goal creases
    goal_crease_left = plt.Circle((-89, 0), 4, fill=False, color='red', linewidth=2)
    goal_crease_right = plt.Circle((89, 0), 4, fill=False, color='red', linewidth=2)
    ax.add_patch(goal_crease_left)
    ax.add_patch(goal_crease_right)
    
    # Add center line and blue lines
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=-25, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=25, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlim(-105, 105)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_title('NHL Rink Zones: Domain Knowledge Integration', fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate (feet)', fontweight='bold')
    ax.set_ylabel('Y Coordinate (feet)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('domain_knowledge_integration.png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()
    
    return zone_definitions, optimal_x_bins, optimal_y_bins

def solution_2_stability_analysis():
    """Solution 2.3: Test parameter stability across different datasets"""
    
    print("\n📈 Parameter Stability Analysis")
    print("=" * 50)
    
    # Load data
    shot_data = load_shot_data()
    
    # Split data by time periods (simulate different seasons/time periods)
    total_shots = len(shot_data)
    split_points = [total_shots // 4, total_shots // 2, 3 * total_shots // 4]
    
    time_periods = {
        'period_1': shot_data.iloc[:split_points[0]],
        'period_2': shot_data.iloc[split_points[0]:split_points[1]],
        'period_3': shot_data.iloc[split_points[1]:split_points[2]],
        'period_4': shot_data.iloc[split_points[2]:]
    }
    
    # Test different parameter configurations
    test_configs = [
        (20, 15, 6),  # Original configuration
        (15, 15, 6),  # Alternative 1
        (25, 20, 6),  # Alternative 2
        (20, 10, 6),  # Alternative 3
    ]
    
    stability_results = {}
    
    print("Testing parameter stability across time periods...")
    
    for period_name, period_data in time_periods.items():
        print(f"\n  {period_name}: {len(period_data)} shots")
        stability_results[period_name] = {}
        
        for x_bins, y_bins, n_clusters in test_configs:
            try:
                # Create spatial bins
                bin_stats, _, _ = create_spatial_bins(period_data, x_bins, y_bins)
                
                # Train clustering model
                silhouette_score_val, _, _ = train_clustering_model(bin_stats, n_clusters)
                
                config_name = f"{x_bins}x{y_bins}_{n_clusters}clusters"
                stability_results[period_name][config_name] = silhouette_score_val
                
                print(f"    {config_name}: {silhouette_score_val:.4f}")
                
            except Exception as e:
                print(f"    {config_name}: Error - {e}")
                stability_results[period_name][config_name] = 0.0
    
    # Analyze stability
    print(f"\nStability Analysis Results:")
    
    # Calculate stability metrics for each configuration
    config_stability = {}
    for config_name in test_configs:
        config_key = f"{config_name[0]}x{config_name[1]}_{config_name[2]}clusters"
        scores = [stability_results[period][config_key] for period in time_periods.keys()]
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv_score = std_score / mean_score if mean_score > 0 else float('inf')  # Coefficient of variation
        
        config_stability[config_key] = {
            'mean': mean_score,
            'std': std_score,
            'cv': cv_score,
            'scores': scores
        }
        
        print(f"  {config_key}:")
        print(f"    Mean: {mean_score:.4f}")
        print(f"    Std: {std_score:.4f}")
        print(f"    CV: {cv_score:.4f} (lower = more stable)")
    
    # Find most stable configuration
    most_stable_config = min(config_stability.keys(), 
                           key=lambda x: config_stability[x]['cv'])
    
    print(f"\nMost Stable Configuration: {most_stable_config}")
    print(f"  CV: {config_stability[most_stable_config]['cv']:.4f}")
    
    # Create stability visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance by time period
    config_names = list(config_stability.keys())
    x_pos = np.arange(len(time_periods))
    width = 0.2
    
    for i, config_name in enumerate(config_names):
        scores = config_stability[config_name]['scores']
        ax1.bar(x_pos + i*width, scores, width, label=config_name, alpha=0.8)
    
    ax1.set_xlabel('Time Period', fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontweight='bold')
    ax1.set_title('Performance Stability Across Time Periods', fontweight='bold', pad=15)
    ax1.set_xticks(x_pos + width * 1.5)
    ax1.set_xticklabels(list(time_periods.keys()))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Coefficient of variation comparison
    config_names = list(config_stability.keys())
    cv_values = [config_stability[config]['cv'] for config in config_names]
    
    bars = ax2.bar(config_names, cv_values, alpha=0.8, color=['red' if config == most_stable_config else 'gray' for config in config_names])
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation', fontweight='bold')
    ax2.set_title('Parameter Stability (Lower CV = More Stable)', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('parameter_stability_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()
    
    return stability_results, config_stability, most_stable_config

def main():
    """Run all parameter sensitivity analysis solutions"""
    
    print("🔍 NHL Shot Clustering: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    # Solution 2.1: Grid search optimization
    results_df, optimal_params = solution_2_grid_search_optimization()
    
    # Solution 2.2: Domain knowledge integration
    zone_definitions, optimal_x_bins, optimal_y_bins = solution_2_domain_knowledge_integration()
    
    # Solution 2.3: Stability analysis
    stability_results, config_stability, most_stable_config = solution_2_stability_analysis()
    
    print("\n✅ Parameter Sensitivity Analysis Complete!")
    print("\n📁 Generated Files:")
    print("   - parameter_sensitivity_grid_search.png")
    print("   - domain_knowledge_integration.png")
    print("   - parameter_stability_analysis.png")
    
    print(f"\n📊 Key Results:")
    print(f"   - Grid search tested {len(results_df)} configurations")
    print(f"   - Domain-informed parameters: {optimal_x_bins}x{optimal_y_bins}")
    print(f"   - Most stable configuration: {most_stable_config}")
    print(f"   - Optimal silhouette score: {optimal_params['silhouette_score']:.4f}")

if __name__ == "__main__":
    main() 