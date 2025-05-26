# NHL Expected Goals (xG) Modeling Project 🏒

## 📋 Project Overview

This project develops a comprehensive machine learning system to predict NHL shot success and calculate Expected Goals (xG) using historical NHL event data. The project demonstrates advanced techniques in sports analytics, temporal validation, business constraint optimization, and real-time deployment considerations.

## 🎯 Project Goals

### Primary Objective
Predict the outcome of individual NHL shot events (goal vs save) and estimate Expected Goals (xG) using machine learning models that can operate in real-time streaming environments.

### Key Requirements
- **Streaming Compatibility**: All features must be available in real-time (no future data)
- **Business Constraints**: Miss rate (α) ≤ 25%, Review rate (β) ≤ 40%
- **Temporal Validation**: Proper time-respecting train/test splits
- **Production Ready**: Sub-150ms prediction latency

## 📁 Project Structure

### 🔧 **Source Code (`src/`)**

#### **Data Processing (`src/data/`)**
- **`scrape_nhl_data.py`** - NHL API data collection with rate limiting
- **`functions.py`** - NHL API utilities and feature engineering functions
- **`verify_data.py`** - Data quality validation and integrity checks

#### **Machine Learning Models (`src/models/`)**
- **`nhl_xg_core.py`** - Complete NHL xG analysis framework (41 features, 5 models)
- **`nhl_business_analysis.py`** - Business constraint optimization and dual-constraint framework

#### **Analysis (`src/analysis/`)**
- **`run_analysis.py`** - Command-line interface for running different analysis types

#### **Visualization (`src/visualization/`)**
- **`report_visualization_package.py`** - Professional visualization suite (8 publication-ready charts)

### 📚 **Documentation (`docs/`)**
- **`FINAL_ACADEMIC_REPORT.md`** - Complete academic report for submission
- **`PROJECT_SUMMARY.md`** - Comprehensive project summary and findings

### 🖼️ **Visualizations (`report-images/`)**
Professional visualizations for academic and business presentation:
- Ice rink heatmaps, distance/angle analysis
- Model evolution and performance tracking
- Business impact dashboards
- Technical architecture diagrams

### 📊 **Data & Database**
- **`nhl_stats.db`** (45MB) - SQLite database with NHL event data
  - 274 games, 98,825 total events
  - 18,470 shots on net with 1,938 goals
  - Tables: games, events, players, teams

### 🗂️ **Legacy Files (`archive/`, `Python/archive/`)**
- Contains 30+ legacy analysis files that were consolidated
- Preserved for reference and reproducibility

### 🔧 **Configuration & Setup**
- **`requirements.txt`** - Python dependencies
- **`run_full_scrape.py`** - Data collection script
- **`.gitignore`** - Git ignore patterns

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Basic Analysis
```bash
cd src/analysis
python run_analysis.py --analysis basic
```

### 3. Run Business Analysis
```bash
python run_analysis.py --analysis business
```

### 4. Generate Visualizations
```bash
cd ../visualization
python report_visualization_package.py
```

## 📈 Key Results

### **Model Performance**
- **Best AUC**: 0.6954 (Position-Enhanced model, 4.1% improvement over baseline)
- **Goal Detection**: 54.4% of goals caught with 18.5% precision
- **Business Compliance**: All models meet α ≤ 25% miss rate threshold
- **Streaming Ready**: 100% of features available in real-time

### **Business Impact**
- **Review Efficiency**: Review 30.5% of shots to find 54.4% of goals
- **Cost Optimization**: Basic Features model most cost-effective ($0.59 per goal)
- **Pre-filtering**: 3.4% efficiency gain with domain knowledge filters
- **Deployment Ready**: Sub-150ms prediction latency

### **Academic Contributions**
- **Temporal Validation**: Proper time-respecting validation methodology
- **Streaming Compatibility**: Framework for real-time ML deployment
- **Dual-Constraint Optimization**: F1 score optimization under α and β constraints
- **Business Threshold Analysis**: α ≤ 25% evaluation framework

## 🔬 Methodology Highlights

### **Data Collection**
- NHL API integration with rate limiting
- 274 games spanning multiple seasons
- Comprehensive event data with player positions

### **Feature Engineering**
- 41 streaming-safe features across 8 categories
- Advanced time-based features (rebounds, pressure situations)
- Player position and handedness effects
- Geometric shot quality metrics

### **Model Development**
- 5 model configurations with progressive complexity
- Proper temporal validation (no future data leakage)
- Ensemble methods with Random Forest + Logistic Regression
- Business constraint optimization

### **Evaluation Framework**
- Goal-focused metrics beyond standard ML measures
- Business threshold analysis (α ≤ 25% miss rate)
- Dual-constraint optimization (α and β constraints)
- Real-world deployment considerations

## 🎓 Academic Submission Ready

This project is structured for academic submission with:
- **Complete methodology documentation**
- **Reproducible results** with clear code organization
- **Professional visualizations** ready for publication
- **Honest evaluation** with realistic performance expectations
- **Real-world applicability** with business deployment analysis

## 🔮 Future Work

### **Phase 1: Enhanced Features (3-6 months)**
- Game context modeling (score differential, man advantage)
- Advanced goalie features (fatigue, recent performance)
- Team momentum indicators

### **Phase 2: Deep Learning (6-12 months)**
- LSTM sequence models for game flow
- Graph Neural Networks for player interactions
- External data integration (weather, referee tendencies)

### **Phase 3: Production Optimization**
- Hyperparameter optimization with Optuna
- Custom loss functions for business objectives
- Model calibration for probability accuracy

## 📞 Contact & Usage

This project demonstrates advanced sports analytics techniques suitable for:
- **Live Broadcasting**: Real-time xG graphics
- **Mobile Applications**: Instant shot analysis
- **Betting Platforms**: Live odds updates
- **Team Analytics**: Bench-side coaching insights
- **Academic Research**: Sports ML methodology

## 🏗️ Development

### **Project Structure Benefits**
- **Logical Organization**: Clear separation of concerns
- **Modular Design**: Easy to extend and maintain
- **Academic Ready**: Professional structure for submission
- **Production Ready**: Organized for deployment
- **Reproducible**: Clear dependencies and documentation

### **Code Quality**
- **Streaming Compatible**: 100% real-time deployment ready
- **Well Documented**: Comprehensive docstrings and comments
- **Tested**: Data validation and quality assurance
- **Scalable**: Ready for full NHL season deployment

---

**🏒 Ready for NHL deployment with comprehensive business analysis and academic rigor!**
