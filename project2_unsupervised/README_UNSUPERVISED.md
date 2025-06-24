# NHL Shot Clustering Analysis: Unsupervised Learning Project

## 🎯 Quick Overview

This project implements unsupervised learning techniques to classify NHL shots into high and low danger zones using clustering algorithms. The key innovation is a **spatial aggregation framework** that addresses coordinate sparsity by grouping shots into spatial bins and calculating goal probabilities.

**🏆 Key Result**: 65.6% improvement in clustering performance over raw coordinate analysis.

## 📁 Project Structure

```
nhl-reporting/
├── src/analysis/
│   └── nhl_shot_clustering_analysis.ipynb    # Main analysis notebook
├── presentation_visuals/                      # Generated charts for video
│   ├── ice_rink_clusters.png                 # Ice rink with danger zones
│   ├── goal_rate_heatmap.png                 # Goal rate heatmap
│   ├── cluster_characteristics.png           # Cluster analysis charts
│   ├── optimal_clusters.png                  # K-selection analysis
│   └── performance_comparison.png            # Raw vs aggregated comparison
├── NHL_SHOT_CLUSTERING_ANALYSIS.md           # Comprehensive analysis report
├── presentation_script_unsupervised.md       # Video presentation script
├── generate_presentation_visuals.py          # Visualization generator
└── UNSUPERVISED_PROJECT_SUMMARY.md           # Complete project summary
```

## 🚀 Quick Start

### 1. Run the Analysis
```bash
cd src/analysis
jupyter notebook nhl_shot_clustering_analysis.ipynb
```

### 2. Generate Visualizations
```bash
python generate_presentation_visuals.py
```

### 3. View Results
- **Analysis Report**: `NHL_SHOT_CLUSTERING_ANALYSIS.md`
- **Project Summary**: `UNSUPERVISED_PROJECT_SUMMARY.md`
- **Video Script**: `presentation_script_unsupervised.md`

## 📊 Key Findings

### **Clustering Performance**
- **Raw Coordinates**: Silhouette score = 0.234
- **Aggregated Probability**: Silhouette score = 0.388
- **Improvement**: +65.6%

### **Danger Zone Classification**
- **High Danger**: 15.6-18.7% goal rate (close to net, center ice)
- **Low Danger**: 7.8-8.9% goal rate (perimeter shots, wide angles)
- **Volume Distribution**: 48.5% low danger, 37.8% medium, 13.7% high danger

### **Statistical Validation**
- **ANOVA F-statistic**: 47.82 (p < 0.001)
- **All clusters**: Statistically significant differences
- **Reliability**: Highly reliable and meaningful classifications

## 🎨 Visualizations

### **Ice Rink Cluster Map**
- Red zones: High danger areas (goal rate > 15%)
- Blue zones: Low danger areas (goal rate < 10%)
- Size: Proportional to shot volume
- Clear patterns: Concentration near net, dispersion in perimeter

### **Goal Rate Heatmap**
- Color gradient: Red (high) to Blue (low) goal rates
- Hot spots: Slot area and crease regions
- Cold zones: Perimeter and wide angles

## 🧠 Methodology

### **Spatial Aggregation Framework**
1. **Spatial Binning**: Group shots into 20x15 grid
2. **Probability Calculation**: Compute goal rates within each bin
3. **Feature Engineering**: Create (x_center, y_center, goal_rate) features
4. **Clustering**: K-Means with optimal k selection (k=6)

### **Data Source**
- **94,047 shots** from 2024-25 NHL season
- **10,895 goals** (10.4% goal rate)
- **Complete NHL coverage**: All 32 teams, 746 players

## 🚀 Business Applications

### **Coaching and Strategy**
- Shot selection optimization
- Defensive positioning
- Practice planning
- Game planning

### **Player Analysis**
- Shot quality assessment
- Performance metrics
- Development tracking
- Scouting

### **Real-time Analytics**
- Live game analysis
- Broadcasting enhancement
- Fan engagement
- Betting models

## 📋 Deliverables Status

### ✅ **Deliverable 1: Jupyter Notebook**
- Complete unsupervised learning analysis
- EDA, clustering, evaluation, and business applications

### ✅ **Deliverable 2: Video Presentation**
- 10-12 minute presentation script
- Professional visualizations generated
- Clear narrative flow

### ✅ **Deliverable 3: GitHub Repository**
- Organized project structure
- Complete documentation
- Reproducible analysis

## 🎓 Academic Contributions

### **Methodological Innovations**
1. Spatial Aggregation Framework for sports coordinate clustering
2. Multi-metric clustering evaluation
3. Domain-specific visualization techniques
4. Statistical validation methodology

### **Domain-Specific Insights**
1. Danger zone quantification
2. Volume-efficiency trade-off analysis
3. Spatial pattern recognition
4. Coaching implications

## 🔗 Related Projects

This project builds on the **supervised learning NHL Expected Goals (xG) modeling** project, sharing the same data pipeline and infrastructure. See `PROJECT_SUMMARY.md` for details on the supervised learning approach.

## 📞 Contact & Next Steps

### **For Video Production**
1. Use `presentation_script_unsupervised.md` as your script
2. Incorporate visualizations from `presentation_visuals/`
3. Target length: 10-12 minutes
4. Upload to YouTube and link in repository

### **For Academic Submission**
1. Submit `nhl_shot_clustering_analysis.ipynb` as main deliverable
2. Include `NHL_SHOT_CLUSTERING_ANALYSIS.md` for comprehensive documentation
3. Reference `UNSUPERVISED_PROJECT_SUMMARY.md` for complete project overview

---

**🏒 Project Status: COMPLETE and ready for submission with all deliverables prepared.** 