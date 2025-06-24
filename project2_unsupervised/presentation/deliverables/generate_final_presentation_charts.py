#!/usr/bin/env python3
"""
Generate Final Presentation Charts for Enhanced Context-Aware NHL Shot Clustering
Based on the latest enhanced analysis with corrected player scoring tiers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import sqlite3

# Set professional style
plt.style.use('default')
sns.set_palette("husl")

class PresentationChartGenerator:
    def __init__(self):
        self.colors = {
            'high_danger': '#FF4444',
            'medium_danger': '#FFA500', 
            'low_danger': '#4444FF',
            'elite': '#FFD700',
            'good': '#C0C0C0',
            'average': '#CD7F32'
        }
        
        # Latest cluster data from enhanced analysis
        self.cluster_data = {
            0: {'name': 'Clutch Time Power Plays', 'shots': 1809, 'goal_rate': 0.202, 'danger': 'Medium', 'elite_rate': 0.081, 'pp_rate': 0.284, 'fatigue': 1.0},
            1: {'name': 'Fresh Legs Perimeter', 'shots': 7787, 'goal_rate': 0.100, 'danger': 'Medium', 'elite_rate': 0.057, 'pp_rate': 0.062, 'fatigue': 0.0},
            2: {'name': 'Point Shot Barrage', 'shots': 17080, 'goal_rate': 0.039, 'danger': 'Low', 'elite_rate': 0.047, 'pp_rate': 0.078, 'fatigue': 0.26},
            3: {'name': 'High-Traffic Slot', 'shots': 9991, 'goal_rate': 0.177, 'danger': 'High', 'elite_rate': 0.078, 'pp_rate': 0.104, 'fatigue': 0.264},
            4: {'name': 'Overtime Desperation', 'shots': 271, 'goal_rate': 0.339, 'danger': 'High', 'elite_rate': 0.070, 'pp_rate': 0.0, 'fatigue': 0.015},
            5: {'name': 'Balanced Attack', 'shots': 14433, 'goal_rate': 0.114, 'danger': 'Medium', 'elite_rate': 0.075, 'pp_rate': 0.094, 'fatigue': 0.266}
        }
        
        self.total_shots = 51371
        self.overall_goal_rate = 0.104

    def create_chart_1_cluster_overview(self):
        """Chart 1: Cluster Overview - Shot Distribution and Goal Rates"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Shot volume by cluster
        clusters = list(self.cluster_data.keys())
        cluster_names = [f"C{i}\n{self.cluster_data[i]['name'][:15]}..." for i in clusters]
        shot_counts = [self.cluster_data[i]['shots'] for i in clusters]
        colors = [self.colors['high_danger'] if self.cluster_data[i]['danger'] == 'High' 
                 else self.colors['medium_danger'] if self.cluster_data[i]['danger'] == 'Medium'
                 else self.colors['low_danger'] for i in clusters]
        
        bars1 = ax1.bar(cluster_names, shot_counts, color=colors, alpha=0.8)
        ax1.set_title('Shot Volume by Cluster', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Number of Shots', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars1, shot_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 200,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Right: Goal rates by cluster
        goal_rates = [self.cluster_data[i]['goal_rate'] for i in clusters]
        bars2 = ax2.bar(cluster_names, goal_rates, color=colors, alpha=0.8)
        ax2.axhline(y=self.overall_goal_rate, color='black', linestyle='--', 
                   label=f'Overall: {self.overall_goal_rate:.1%}')
        ax2.set_title('Goal Rates by Cluster', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Goal Rate', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Add value labels
        for bar, rate in zip(bars2, goal_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('final_deliverables/chart_1_cluster_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_2_elite_scorer_deployment(self):
        """Chart 2: Elite Scorer Strategic Deployment"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        clusters = list(self.cluster_data.keys())
        cluster_names = [f"C{i}: {self.cluster_data[i]['name']}" for i in clusters]
        elite_rates = [self.cluster_data[i]['elite_rate'] for i in clusters]
        
        # Create gradient colors based on elite scorer rate
        colors = plt.cm.YlOrRd([rate for rate in elite_rates])
        
        bars = ax.bar(cluster_names, elite_rates, color=colors, alpha=0.8)
        ax.set_title('Elite Scorer Deployment by Cluster\n"Stars Shine Brightest in Clutch Moments"', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Elite Scorer Rate (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Add value labels and insights
        for i, (bar, rate) in enumerate(zip(bars, elite_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight key insights
            if i == 0:  # Clutch Time Power Plays
                ax.annotate('Highest Elite\nDeployment', 
                           xy=(bar.get_x() + bar.get_width()/2., height),
                           xytext=(bar.get_x() + bar.get_width()/2., height + 0.02),
                           ha='center', fontweight='bold', color='red',
                           arrowprops=dict(arrowstyle='->', color='red'))
            elif i == 2:  # Point Shots
                ax.annotate('Lowest Elite\nRate', 
                           xy=(bar.get_x() + bar.get_width()/2., height),
                           xytext=(bar.get_x() + bar.get_width()/2., height + 0.015),
                           ha='center', fontweight='bold', color='blue',
                           arrowprops=dict(arrowstyle='->', color='blue'))
        
        plt.tight_layout()
        plt.savefig('final_deliverables/chart_2_elite_scorer_deployment.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_3_fatigue_paradox(self):
        """Chart 3: The Fatigue Paradox Discovery"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group clusters by fatigue level
        fatigue_groups = {
            'No Fatigue (0%)': [1],
            'Low Fatigue (1.5%)': [4],
            'Moderate Fatigue (26%)': [2, 3, 5],
            'High Fatigue (100%)': [0]
        }
        
        group_names = []
        avg_goal_rates = []
        colors_list = []
        
        for group_name, cluster_ids in fatigue_groups.items():
            group_names.append(group_name)
            if len(cluster_ids) == 1:
                rate = self.cluster_data[cluster_ids[0]]['goal_rate']
                if cluster_ids[0] == 0:  # High fatigue
                    colors_list.append('#FF4444')
                elif cluster_ids[0] == 1:  # No fatigue
                    colors_list.append('#4444FF')
                else:  # Low fatigue
                    colors_list.append('#44FF44')
            else:
                # Average for moderate fatigue group
                total_shots = sum(self.cluster_data[cid]['shots'] for cid in cluster_ids)
                weighted_rate = sum(self.cluster_data[cid]['goal_rate'] * self.cluster_data[cid]['shots'] 
                                  for cid in cluster_ids) / total_shots
                rate = weighted_rate
                colors_list.append('#FFA500')
            avg_goal_rates.append(rate)
        
        bars = ax.bar(group_names, avg_goal_rates, color=colors_list, alpha=0.8)
        ax.axhline(y=self.overall_goal_rate, color='black', linestyle='--', 
                  label=f'Overall Average: {self.overall_goal_rate:.1%}')
        
        ax.set_title('The Fatigue Paradox\n"Tired Players Make Better Shots"', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Goal Rate', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # Add value labels and insights
        for bar, rate in zip(bars, avg_goal_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add insight text
        ax.text(0.5, 0.95, 'Key Insight: 100% fatigue shots have 2x higher goal rate than fresh legs',
               transform=ax.transAxes, ha='center', va='top', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('final_deliverables/chart_3_fatigue_paradox.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_4_overtime_dominance(self):
        """Chart 4: Overtime Creates Different Game"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Overtime cluster vs others
        categories = ['Overtime\n(Cluster 4)', 'All Other\nClusters']
        goal_rates = [self.cluster_data[4]['goal_rate'], 
                     sum(self.cluster_data[i]['goal_rate'] * self.cluster_data[i]['shots'] 
                         for i in [0,1,2,3,5]) / sum(self.cluster_data[i]['shots'] for i in [0,1,2,3,5])]
        
        colors = ['#FF0000', '#0066CC']
        bars1 = ax1.bar(categories, goal_rates, color=colors, alpha=0.8)
        ax1.set_title('Overtime vs Regular Play Goal Rates', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Goal Rate', fontsize=12)
        
        for bar, rate in zip(bars1, goal_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add multiplier annotation
        multiplier = goal_rates[0] / goal_rates[1]
        ax1.text(0.5, 0.8, f'{multiplier:.1f}x Higher\nGoal Rate', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
        
        # Right: Cluster 4 characteristics
        characteristics = ['Goal Rate', 'Fresh Legs %', 'Overtime %', 'Elite Scorer %']
        values = [self.cluster_data[4]['goal_rate'], 0.948, 0.934, self.cluster_data[4]['elite_rate']]
        
        bars2 = ax2.bar(characteristics, values, color=['red', 'green', 'blue', 'gold'], alpha=0.8)
        ax2.set_title('Cluster 4: Overtime Desperation Profile', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Rate/Percentage', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('final_deliverables/chart_4_overtime_dominance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_5_business_impact_matrix(self):
        """Chart 5: Business Impact Matrix"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create scatter plot with goal rate vs shot volume
        clusters = list(self.cluster_data.keys())
        x_values = [self.cluster_data[i]['shots']/1000 for i in clusters]  # In thousands
        y_values = [self.cluster_data[i]['goal_rate'] for i in clusters]
        
        # Size by elite scorer rate
        sizes = [self.cluster_data[i]['elite_rate'] * 2000 for i in clusters]
        
        # Color by danger level
        colors = [self.colors['high_danger'] if self.cluster_data[i]['danger'] == 'High'
                 else self.colors['medium_danger'] if self.cluster_data[i]['danger'] == 'Medium'
                 else self.colors['low_danger'] for i in clusters]
        
        scatter = ax.scatter(x_values, y_values, s=sizes, c=colors, alpha=0.7, edgecolors='black')
        
        # Add cluster labels
        for i, cluster_id in enumerate(clusters):
            ax.annotate(f"C{cluster_id}: {self.cluster_data[cluster_id]['name'][:12]}...", 
                       (x_values[i], y_values[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add quadrant lines
        ax.axhline(y=self.overall_goal_rate, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=self.total_shots/len(clusters)/1000, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.02, 0.98, 'Low Volume\nHigh Quality', transform=ax.transAxes, 
               ha='left', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(0.98, 0.98, 'High Volume\nHigh Quality', transform=ax.transAxes, 
               ha='right', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.7))
        ax.text(0.02, 0.02, 'Low Volume\nLow Quality', transform=ax.transAxes, 
               ha='left', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.text(0.98, 0.02, 'High Volume\nLow Quality', transform=ax.transAxes, 
               ha='right', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_xlabel('Shot Volume (Thousands)', fontsize=12)
        ax.set_ylabel('Goal Rate', fontsize=12)
        ax.set_title('Business Impact Matrix: Volume vs Quality vs Elite Deployment\n(Bubble size = Elite Scorer Rate)', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['high_danger'], 
                                     markersize=10, label='High Danger'),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['medium_danger'], 
                                     markersize=10, label='Medium Danger'),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['low_danger'], 
                                     markersize=10, label='Low Danger')]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig('final_deliverables/chart_5_business_impact_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_6_actionable_insights_summary(self):
        """Chart 6: Actionable Insights Summary Dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Key metric cards
        metrics = [
            ("Total Shots Analyzed", "51,371", "Complete 2024 season"),
            ("Elite Scorer Advantage", "72%", "Higher in clutch vs point shots"),
            ("Fatigue Effect", "2.0x", "Tired players goal rate multiplier"),
            ("Overtime Multiplier", "3.3x", "vs regular play goal rate"),
            ("Cluster Effectiveness", "19.4%", "High danger goal rate"),
            ("Business Ready", "6", "Actionable shot archetypes")
        ]
        
        for i, (title, value, subtitle) in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Create metric card
            ax.text(0.5, 0.7, value, ha='center', va='center', fontsize=24, fontweight='bold',
                   transform=ax.transAxes, color='darkblue')
            ax.text(0.5, 0.4, title, ha='center', va='center', fontsize=12, fontweight='bold',
                   transform=ax.transAxes)
            ax.text(0.5, 0.2, subtitle, ha='center', va='center', fontsize=10,
                   transform=ax.transAxes, style='italic')
            
            # Style the card
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor('darkblue')
        
        plt.suptitle('Enhanced Context-Aware NHL Shot Clustering: Key Insights Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig('final_deliverables/chart_6_actionable_insights_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_charts(self):
        """Generate all presentation charts"""
        print("🎨 GENERATING FINAL PRESENTATION CHARTS")
        print("="*60)
        
        charts = [
            ("Chart 1: Cluster Overview", self.create_chart_1_cluster_overview),
            ("Chart 2: Elite Scorer Deployment", self.create_chart_2_elite_scorer_deployment),
            ("Chart 3: Fatigue Paradox", self.create_chart_3_fatigue_paradox),
            ("Chart 4: Overtime Dominance", self.create_chart_4_overtime_dominance),
            ("Chart 5: Business Impact Matrix", self.create_chart_5_business_impact_matrix),
            ("Chart 6: Actionable Insights Summary", self.create_chart_6_actionable_insights_summary)
        ]
        
        for chart_name, chart_function in charts:
            print(f"⏳ Creating {chart_name}...")
            chart_function()
            print(f"✅ {chart_name} completed")
        
        print(f"\n🎉 ALL PRESENTATION CHARTS GENERATED!")
        print(f"📊 6 professional charts ready for presentation")
        print(f"📁 Saved to: final_deliverables/")

if __name__ == "__main__":
    generator = PresentationChartGenerator()
    generator.generate_all_charts() 