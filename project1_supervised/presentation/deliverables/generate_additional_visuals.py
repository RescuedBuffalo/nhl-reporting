#!/usr/bin/env python3
"""
NHL xG Additional Presentation Visualizations Generator
Creates histogram, bar chart, and feature overview visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import sqlite3
import json
from pathlib import Path

# Set style and parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory
output_dir = Path("visualizations/presentation_charts")
output_dir.mkdir(parents=True, exist_ok=True)

# Color scheme
BLUE = '#0A66C2'
GREEN = '#2E7D32'
RED = '#EF5350'
LIGHT_GREEN = '#66BB6A'
GREY = '#BDBDBD'

def load_shot_data():
    """Load shot data from database"""
    try:
        conn = sqlite3.connect('nhl_stats.db')
        query = """
        SELECT 
            e.x, e.y, e.eventType, e.details
        FROM events e
        WHERE e.eventType IN ('goal', 'shot-on-goal')
        AND e.x IS NOT NULL 
        AND e.y IS NOT NULL
        AND e.details IS NOT NULL
        ORDER BY e.gamePk, e.eventIdx
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_shot_distance_histogram():
    """1. Histogram – Shot Distances"""
    print("Creating Shot Distance Histogram...")
    
    # Load data
    df = load_shot_data()
    
    if df is not None:
        # Calculate distance from goal (assuming goal at x=89, y=0)
        goal_x, goal_y = 89, 0
        df['distance'] = np.sqrt((df['x'] - goal_x)**2 + (df['y'] - goal_y)**2)
        distances = df['distance'].values
        print(f"Loaded {len(distances):,} shots from database")
    else:
        # Generate mock data if database not available
        print("Database not available, generating mock distance data...")
        np.random.seed(42)
        # Create realistic shot distance distribution (skewed toward close shots)
        close_shots = np.random.exponential(15, 3000)  # Many close shots
        medium_shots = np.random.normal(30, 10, 1500)  # Some medium shots
        long_shots = np.random.normal(50, 15, 500)     # Few long shots
        distances = np.concatenate([close_shots, medium_shots, long_shots])
        distances = distances[distances > 0]  # Remove negative distances
        distances = distances[distances < 100]  # Remove unrealistic distances
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with 1-foot bins
    bins = np.arange(0, int(np.max(distances)) + 2, 1)
    n, bins_used, patches = ax.hist(distances, bins=bins, color=BLUE, alpha=0.7, 
                                   edgecolor='black', linewidth=0.5)
    
    # Highlight the skew toward low distances
    # Color bars differently based on distance
    for i, patch in enumerate(patches):
        if bins_used[i] < 20:  # Close shots
            patch.set_facecolor(GREEN)
            patch.set_alpha(0.8)
        elif bins_used[i] < 40:  # Medium shots
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.7)
        else:  # Long shots
            patch.set_facecolor(GREY)
            patch.set_alpha(0.6)
    
    # Add statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    
    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_dist:.1f} ft')
    ax.axvline(median_dist, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_dist:.1f} ft')
    
    ax.set_xlabel('Distance from Goal (feet)', fontweight='bold')
    ax.set_ylabel('Number of Shots', fontweight='bold')
    ax.set_title('Shot Distance Distribution', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Add annotation about skew
    ax.text(0.6, 0.8, f'Skewed toward close shots\n{np.sum(distances < 30)/len(distances)*100:.1f}% within 30 ft', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shot_distance_histogram.png', bbox_inches='tight')
    plt.close()
    print("✅ Saved: shot_distance_histogram.png")

def create_shot_type_distribution():
    """2. Bar Chart – Shot Type Distribution"""
    print("Creating Shot Type Distribution...")
    
    # Load data
    df = load_shot_data()
    
    if df is not None:
        # Extract shot types from details JSON
        shot_types = []
        for details_str in df['details']:
            try:
                if pd.notna(details_str):
                    details = json.loads(details_str)
                    shot_type = details.get('shotType', 'Unknown')
                    shot_types.append(shot_type)
                else:
                    shot_types.append('Unknown')
            except:
                shot_types.append('Unknown')
        
        df['shot_type'] = shot_types
        type_counts = df['shot_type'].value_counts()
        print(f"Found {len(type_counts)} different shot types")
        
    else:
        # Generate mock data if database not available
        print("Database not available, generating mock shot type data...")
        shot_types = ['Wrist Shot', 'Slap Shot', 'Snap Shot', 'Backhand', 'Tip-In', 
                     'Deflection', 'Wrap-around', 'Unknown']
        # Realistic distribution (wrist shots most common)
        counts = [8500, 2200, 3100, 1800, 1200, 900, 400, 600]
        type_counts = pd.Series(counts, index=shot_types)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by count (descending)
    type_counts_sorted = type_counts.sort_values(ascending=True)  # Ascending for horizontal bars
    
    # Create color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(type_counts_sorted)))
    
    bars = ax.barh(range(len(type_counts_sorted)), type_counts_sorted.values, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, type_counts_sorted.values)):
        width = bar.get_width()
        ax.text(width + max(type_counts_sorted.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(type_counts_sorted)))
    ax.set_yticklabels(type_counts_sorted.index)
    ax.set_xlabel('Number of Shots', fontweight='bold')
    ax.set_ylabel('Shot Type', fontweight='bold')
    ax.set_title('Shot Types Across Dataset', fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add total shots annotation
    total_shots = type_counts_sorted.sum()
    ax.text(0.7, 0.95, f'Total Shots: {total_shots:,}', transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shot_type_distribution.png', bbox_inches='tight')
    plt.close()
    print("✅ Saved: shot_type_distribution.png")

def create_feature_table_overview():
    """3. Table or Visual Grid – Feature Categories"""
    print("Creating Feature Categories Overview...")
    
    # Define feature categories and examples
    categories = {
        'Geometry': ['Distance to goal', 'Shot angle', 'X,Y coordinates'],
        'Shot Context': ['Shot type', 'Rebound flag', 'Empty net'],
        'Game State': ['Score differential', 'Man advantage', 'Period'],
        'Player Info': ['Shooter position', 'Player ID', 'Handedness'],
        'Rebound Behavior': ['Time since last shot', 'Shot sequence', 'Rebound type'],
        'Time Context': ['Period time', 'Game time', 'Recent pressure']
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up 2x3 grid
    rows, cols = 2, 3
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows
    
    # Colors for each category
    category_colors = [BLUE, GREEN, RED, '#FF9800', '#9C27B0', '#607D8B']
    
    # Draw grid boxes
    for i, (category, features) in enumerate(categories.items()):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = col * cell_width
        y = 1 - (row + 1) * cell_height  # Flip y-axis
        
        # Create rectangle
        rect = patches.FancyBboxPatch((x + 0.02, y + 0.05), 
                                     cell_width - 0.04, cell_height - 0.1,
                                     boxstyle="round,pad=0.02",
                                     facecolor=category_colors[i], 
                                     alpha=0.3,
                                     edgecolor=category_colors[i], 
                                     linewidth=2)
        ax.add_patch(rect)
        
        # Add category title
        ax.text(x + cell_width/2, y + cell_height - 0.15, category,
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=category_colors[i])
        
        # Add feature examples
        feature_text = '\n'.join([f'• {feature}' for feature in features])
        ax.text(x + cell_width/2, y + cell_height/2 - 0.05, feature_text,
                ha='center', va='center', fontsize=10,
                color='black', linespacing=1.5)
    
    # Set up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Feature Engineering Overview', fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Add total feature count
    total_features = sum(len(features) for features in categories.values())
    ax.text(0.5, 0.05, f'Total: {total_features} engineered features across 6 categories',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_table_overview.png', bbox_inches='tight')
    plt.close()
    print("✅ Saved: feature_table_overview.png")

def main():
    """Generate all additional presentation visualizations"""
    print("🎨 Generating Additional NHL xG Presentation Visualizations")
    print("=" * 60)
    
    # Create all visualizations
    create_shot_distance_histogram()
    create_shot_type_distribution()
    create_feature_table_overview()
    
    print("\n" + "=" * 60)
    print("🏆 ADDITIONAL VISUALIZATIONS COMPLETE!")
    print(f"📁 Saved to: {output_dir}")
    print("\nAll generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"   - {file.name}")

if __name__ == "__main__":
    main() 