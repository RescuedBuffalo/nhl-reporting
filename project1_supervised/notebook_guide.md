# NHL Expected Goals Analysis - Notebook Guide

## 📖 Overview

This guide provides step-by-step instructions for running the `NHL_Expected_Goals_Analysis.ipynb` notebook and understanding each section of the analysis.

## 🔧 Prerequisites

Before running the notebook, ensure you have:

1. **Python Environment**: Python 3.8+ with Jupyter installed
2. **Dependencies**: All packages from `requirements.txt` installed
3. **Data**: `nhl_stats.db` SQLite database in the project directory
4. **Access**: Data pipeline modules in `../data_pipeline/src/`

## 📋 Notebook Sections

### 1. Project Setup and Data Loading
**Purpose**: Import libraries and set up the analysis environment

**What it does**:
- Imports all required Python libraries
- Sets matplotlib styling for consistent visualizations
- Adds data pipeline modules to Python path
- Configures warning filters

**Expected output**: Library version information and success messages

**Runtime**: ~10 seconds

---

### 2. Data Loading and Initial Exploration
**Purpose**: Load NHL shot data from the SQLite database

**What it does**:
- Connects to `nhl_stats.db` SQLite database
- Loads shot events, player positions, and team information
- Displays basic dataset statistics
- Performs initial data quality checks

**Expected output**:
- Data loading confirmation messages
- Dataset overview (number of shots, goals, games)
- Data quality metrics (missing values, goal rate)

**Runtime**: ~30 seconds

**⚠️ Note**: If you see errors here, ensure `nhl_stats.db` is in the correct location

---

### 3. Exploratory Data Analysis (EDA)
**Purpose**: Create comprehensive visualizations to understand the data

**What it does**:
- Generates 6 key visualizations:
  1. Shot vs Goal distribution (pie chart)
  2. Shots by period (bar chart)
  3. Shot location heatmap
  4. Goal location heatmap
  5. Games over time
  6. Goal rate by team
- Calculates key statistics

**Expected output**:
- Large figure with 6 subplots
- Summary statistics about teams and performance

**Runtime**: ~45 seconds

---

### 4. Data Processing and Feature Engineering
**Purpose**: Process raw data and create predictive features

**What it does**:
- Uses existing `NHLxGAnalyzer` from data pipeline
- Engineers 18+ features across multiple categories:
  - Geometric: distance_to_net, angle_to_net
  - Zone: in_crease, in_slot, from_point
  - Shot type: is_wrist_shot, is_slap_shot, etc.
  - Position: is_forward, is_defenseman
  - Temporal: potential_rebound, final_two_minutes
- Creates feature correlation heatmap
- Shows feature correlation with target variable

**Expected output**:
- Feature engineering progress messages
- Feature set definitions (Basic, Zone Enhanced, etc.)
- Correlation heatmap visualization
- Top 10 feature correlations with goals

**Runtime**: ~1-2 minutes

---

### 5. Model Training and Comparison
**Purpose**: Train and evaluate multiple machine learning models

**What it does**:
- Defines 4 models: Logistic Regression, Random Forest, Decision Tree, SVM
- Uses temporal split (80/20) for realistic evaluation
- Trains each model with proper scaling (for SVM)
- Calculates multiple metrics: ROC-AUC, Average Precision, Cross-validation
- Creates performance comparison table

**Expected output**:
- Training progress for each model
- Performance metrics for each model
- Model performance summary table
- Identification of best performing model

**Runtime**: ~2-3 minutes

**📊 Expected Performance**: Random Forest typically performs best with ~0.83 ROC-AUC

---

### 6. Model Evaluation and Visualization
**Purpose**: Create comprehensive evaluation visualizations

**What it does**:
- Generates 4 evaluation plots:
  1. ROC curves for all models
  2. Precision-Recall curves
  3. Performance comparison bar chart
  4. Prediction distribution for best model
- Produces detailed classification report
- Shows confusion matrix
- Displays feature importance (for tree-based models)

**Expected output**:
- Large evaluation figure with 4 subplots
- Detailed classification report
- Confusion matrix
- Top 10 feature importances

**Runtime**: ~1 minute

---

### 7. Business Analysis and Constraints
**Purpose**: Analyze business constraints for production deployment

**What it does**:
- Tests different probability thresholds (0.1 to 0.9)
- Calculates business metrics:
  - Miss rate (α): proportion of goals missed
  - Review rate (β): proportion of shots flagged
- Finds optimal threshold meeting constraints (α ≤ 25%, β ≤ 40%)
- Creates constraint space visualization

**Expected output**:
- Business constraint analysis results
- Optimal threshold identification (if feasible)
- Constraint space plots
- Production readiness assessment

**Runtime**: ~45 seconds

**🎯 Goal**: Find threshold that balances accuracy with operational efficiency

---

### 8. Conclusions and Recommendations
**Purpose**: Summarize findings and provide actionable recommendations

**What it does**:
- Summarizes model performance results
- Lists key insights from analysis
- Assesses business readiness for deployment
- Provides specific recommendations for:
  - Immediate deployment steps
  - Data improvements
  - Model enhancements
  - Business integration

**Expected output**:
- Comprehensive summary report
- Structured recommendations
- Success criteria checklist
- Final deployment recommendation

**Runtime**: ~5 seconds

## 🚀 Running the Notebook

### Option 1: Run All Cells
```bash
# Start Jupyter
jupyter notebook NHL_Expected_Goals_Analysis.ipynb

# In Jupyter: Cell → Run All
```

### Option 2: Step-by-Step Execution
1. Run each cell individually using `Shift + Enter`
2. Wait for each cell to complete before proceeding
3. Review outputs and visualizations at each step

## ⏱️ Total Runtime
- **Full notebook**: ~8-12 minutes
- **Key bottlenecks**: Model training (section 5) and feature engineering (section 4)

## 🔍 Troubleshooting

### Common Issues

**1. Database Connection Error**
```
Error: no such table: events
```
**Solution**: Ensure `nhl_stats.db` is in the project directory

**2. Module Import Error**
```
ModuleNotFoundError: No module named 'models.nhl_xg_core'
```
**Solution**: Verify data pipeline is accessible at `../data_pipeline/src/`

**3. Memory Issues**
```
MemoryError during model training
```
**Solution**: Close other applications or use a subset of data for testing

**4. Visualization Issues**
```
Plots not displaying correctly
```
**Solution**: Ensure matplotlib backend is properly configured: `%matplotlib inline`

## 📊 Expected Results

By the end of the notebook, you should have:

1. **Data Understanding**: Clear picture of NHL shot data characteristics
2. **Model Performance**: Comparison of 4 different algorithms
3. **Best Model**: Random Forest with ~0.83 ROC-AUC
4. **Business Analysis**: Threshold optimization for production use
5. **Recommendations**: Clear next steps for deployment

## 💡 Tips for Success

1. **Run sequentially**: Each section builds on previous results
2. **Check outputs**: Verify each cell produces expected results
3. **Save regularly**: Use `Ctrl+S` to save progress
4. **Clear outputs**: If re-running, consider clearing outputs first
5. **Monitor memory**: Close notebook when done to free resources

## 📝 Notes for Academic Review

This notebook demonstrates:
- **Supervised Learning Best Practices**: Proper train/test splitting, cross-validation
- **Business Integration**: Real-world constraint analysis
- **Model Interpretability**: Feature importance and business metrics
- **Production Readiness**: Performance validation and deployment considerations 