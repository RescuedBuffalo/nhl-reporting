# NHL Analytics Portfolio

A comprehensive collection of machine learning projects analyzing NHL data, demonstrating progression from supervised to unsupervised learning techniques.

## 📁 Repository Structure

This repository contains three distinct projects sharing a common data pipeline:

```
nhl-reporting/
├── data_pipeline/           # Shared data infrastructure
│   ├── src/                # Data collection and processing
│   ├── nhl_stats.db        # NHL database (95MB)
│   └── requirements.txt    # Dependencies
├── project1_supervised/     # NHL Expected Goals (xG) Modeling
│   ├── results/            # Model outputs and analysis
│   ├── report-images/      # Visualizations
│   ├── PROJECT_SUMMARY.md  # Comprehensive project report
│   └── ...                 # Other project files
├── project2_unsupervised/   # NHL Shot Clustering Analysis
│   ├── presentation_visuals/ # Generated charts
│   ├── NHL_SHOT_CLUSTERING_ANALYSIS.md
│   └── ...                 # Other project files
├── project3_future/         # Future project (TBD)
└── README.md               # This file
```

## 🎯 Project Overview

### **Project 1: Supervised Learning - NHL Expected Goals (xG) Modeling**
- **Objective**: Predict shot success probability using historical NHL data
- **Technique**: Supervised learning with temporal validation
- **Key Achievement**: 6.6x more data, competitive model performance
- **Status**: ✅ Complete

### **Project 2: Unsupervised Learning - NHL Shot Clustering Analysis**
- **Objective**: Classify high and low danger shot zones using clustering
- **Technique**: Spatial aggregation framework with K-means clustering
- **Key Achievement**: 65.6% improvement in clustering performance
- **Status**: ✅ Complete

### **Project 3: Future Project**
- **Objective**: TBD (potentially reinforcement learning, time series analysis, or deep learning)
- **Status**: 🔄 Planning

## 🛠️ Shared Data Pipeline

The `data_pipeline/` directory contains the infrastructure used by all projects:

- **NHL API Integration**: Real-time data collection from api-web.nhle.com
- **Database**: SQLite with 1,404 games, 98,453 events, 96,408 shots
- **Data Processing**: Coordinate normalization, feature engineering
- **Quality Assurance**: Bias elimination, data validation

## 🚀 Quick Start

### **For Project 1 (Supervised Learning)**
```bash
cd project1_supervised
# See PROJECT_SUMMARY.md for detailed instructions
```

### **For Project 2 (Unsupervised Learning)**
```bash
cd project2_unsupervised
python generate_presentation_visuals.py
# See README_UNSUPERVISED.md for detailed instructions
```

### **For Data Pipeline**
```bash
cd data_pipeline
# See src/ for data collection and processing scripts
```

## 📊 Key Metrics

| Project | Technique | Key Metric | Improvement |
|---------|-----------|------------|-------------|
| **Project 1** | Supervised Learning | Model Performance | 6.6x more data |
| **Project 2** | Unsupervised Learning | Clustering Quality | +65.6% silhouette score |
| **Project 3** | TBD | TBD | TBD |

## 🎓 Academic Context

This portfolio demonstrates progression through different machine learning paradigms:

1. **Supervised Learning**: Predictive modeling with business constraints
2. **Unsupervised Learning**: Pattern discovery and clustering
3. **Future**: Advanced techniques (reinforcement learning, deep learning, etc.)

Each project builds on the previous while exploring new methodologies and applications.

## 📋 Submission Guidelines

Each project is designed to be submitted independently:

- **Project 1**: Focus on supervised learning methodology and business applications
- **Project 2**: Focus on unsupervised learning and spatial analytics
- **Project 3**: TBD based on course requirements

## 🔗 Data Source

All projects use the same NHL data source:
- **API**: api-web.nhle.com/v1 (current NHL API)
- **Coverage**: 2024-25 season, all 32 teams, 746 players
- **Events**: 185,256 total events (94,047 shots, 10,895 goals)

## 📞 Contact

For questions about specific projects, see the individual project directories and their respective README files.

---

**🏒 Portfolio Status**: Projects 1 & 2 complete, Project 3 in planning phase.
