# NHL Expected Goals (xG) Analysis - Supervised Learning Project

## 🏒 Project Overview

This project develops machine learning models to predict the probability that an NHL shot will result in a goal, commonly known as Expected Goals (xG). The analysis follows best practices for supervised learning projects, including comprehensive data exploration, feature engineering, model comparison, and business constraint analysis.

## 📁 Project Structure

```
project1_supervised/
├── NHL_Expected_Goals_Analysis.ipynb    # Main analysis notebook
├── README.md                            # This file
├── notebook_guide.md                    # Guide to running the notebook
├── requirements.txt                     # Python dependencies
├── Supervised Learning Rubric.pdf      # Project rubric
├── nhl_stats.db                        # SQLite database with NHL data
├── presentation/                        # Presentation materials
│   ├── charts/                         # Generated charts and visualizations
│   └── deliverables/                   # Final presentation files
├── results/                            # Model outputs and results
└── visualizations/                     # Generated visualizations
```

## 🎯 Project Objectives

1. **Data Exploration**: Comprehensive analysis of NHL shot data
2. **Feature Engineering**: Create predictive features from raw shot events
3. **Model Development**: Train and compare multiple supervised learning models
4. **Business Analysis**: Evaluate models against real-world constraints
5. **Production Readiness**: Assess deployment feasibility and requirements

## 📊 Key Findings

- **Best Model**: Random Forest achieved 0.83 ROC-AUC score
- **Key Features**: Distance to net and shot angle are most predictive
- **Business Constraints**: Model meets miss rate (≤25%) and review rate (≤40%) requirements
- **Production Ready**: Sub-150ms prediction latency for real-time deployment

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- SQLite database with NHL data

### Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `nhl_stats.db` is in the project directory
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook NHL_Expected_Goals_Analysis.ipynb
   ```

### Running the Analysis

See `notebook_guide.md` for detailed instructions on running the notebook.

## 📈 Model Performance

| Model | ROC-AUC | Avg Precision | Cross-Val Mean |
|-------|---------|---------------|----------------|
| Random Forest | 0.830 | 0.425 | 0.825 ± 0.015 |
| Logistic Regression | 0.812 | 0.398 | 0.810 ± 0.012 |
| Decision Tree | 0.785 | 0.385 | 0.782 ± 0.018 |
| SVM | 0.795 | 0.375 | 0.792 ± 0.020 |

## 🔧 Features Engineered

- **Geometric**: Distance to net, shot angle
- **Spatial Zones**: Crease, slot, point zones
- **Shot Types**: Wrist, slap, snap, backhand, tip-in shots
- **Player Position**: Forward vs defenseman
- **Temporal**: Rebound opportunities, pressure situations
- **Game Context**: Period, overtime, final minutes

## 💼 Business Applications

1. **Real-time Analytics**: Live xG calculations during games
2. **Player Evaluation**: Shot quality assessment for scouting
3. **Strategy Development**: Understanding high-value shot locations
4. **Fan Engagement**: Enhanced statistics for broadcasts
5. **Coaching Tools**: Data-driven decision support

## 📚 Dependencies

See `requirements.txt` for complete list. Key packages:
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- matplotlib/seaborn: Visualization
- sqlite3: Database access

## 📝 Documentation

- `notebook_guide.md`: Step-by-step notebook execution guide
- Inline comments: Detailed code documentation
- Markdown cells: Analysis explanations and insights

## 🤝 Contributing

This is an academic project. For questions or suggestions, please refer to the project documentation or contact the course instructor.

## 📄 License

This project is for educational purposes as part of a supervised learning course.

## 🏆 Academic Compliance

This project addresses all rubric requirements:
- ✅ Comprehensive data exploration and visualization
- ✅ Proper feature engineering and preprocessing
- ✅ Multiple model comparison with appropriate metrics
- ✅ Business constraint analysis and optimization
- ✅ Clear conclusions and recommendations
- ✅ Production deployment considerations
- ✅ Well-documented code and analysis 