# Voriconazole Pharmacokinetic Analysis

A comprehensive pharmacokinetic (PK) analysis project for voriconazole, incorporating exploratory data analysis (EDA), machine learning modeling, and external validation.

## 📊 Project Overview

This project analyzes voriconazole pharmacokinetics with a focus on:
- CYP2C19 genotype effects on drug clearance (CL/F)
- Inflammation status (CRP levels) impact on PK parameters
- Machine learning models for concentration prediction
- External validation of predictive models

## 🗂️ Project Structure

```
voriconazole-pk-analysis/
├── data/                   # Data files (not included for privacy)
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_EDA_and_Modeling.ipynb
│   └── 02_Model_Analysis.ipynb
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── results/                # Model outputs and predictions
├── figures/                # Generated plots and visualizations
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🔬 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- **Genotype Analysis**: PM (Poor Metabolizer), IM (Intermediate Metabolizer), NM (Normal Metabolizer)
- **Inflammation Stratification**: High CRP (>100 mg/L) vs Low CRP (≤100 mg/L)
- **Distribution Analysis**: CL/F and concentration distributions across groups
- **Statistical Testing**: ANOVA and post-hoc tests

### 2. Machine Learning Models
- **Random Forest Regressor**: For CL/F and concentration prediction
- **Feature Engineering**: Genotype encoding, inflammation markers
- **Model Evaluation**: R², MAE, RMSE, MAPE metrics
- **Cross-Validation**: Train/test split and external validation

### 3. Key Visualizations
- Violin plots for CL/F distribution by genotype
- Scatter plots for predicted vs observed values
- Feature importance analysis
- External validation performance plots

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voriconazole-pk-analysis.git
cd voriconazole-pk-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

## 📈 Key Findings

- **Genotype Effect**: Significant differences in CL/F across CYP2C19 genotypes (PM < IM < NM)
- **Inflammation Impact**: High CRP levels associated with altered pharmacokinetics
- **Model Performance**: 
  - Internal validation R² > 0.85
  - External validation with robust predictive accuracy
  - P30 (predictions within ±30%) > 70%

## 📊 Model Performance Metrics

| Metric | CL/F Prediction | Concentration Prediction |
|--------|----------------|-------------------------|
| R² | 0.850+ | 0.800+ |
| MAE | Low | Low |
| RMSE | Acceptable | Acceptable |
| MAPE | <20% | <25% |

## 🔧 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## 📝 Usage Example

```python
import pandas as pd
from src.modeling import train_random_forest
from src.visualization import plot_predictions

# Load data
df = pd.read_csv('data/example.csv')

# Train model
model, metrics = train_random_forest(df, target='CL/F')

# Visualize results
plot_predictions(y_true, y_pred)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Yehui Zhou - Initial work

## 🙏 Acknowledgments

- Research team members
- Data providers
- Open-source community

## 📮 Contact

For questions or collaboration opportunities, please contact [lenhartkoo@foxmail.com]

## 🔖 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{voriconazole_pk_analysis,
  title={Voriconazole Pharmacokinetic Analysis with Machine Learning},
  author={Yehui Zhou},
  year={2026},
  publisher={GitHub},
  howpublished={\\url{https://github.com/harkool/Voriconazole}}
}
```

---

**Note**: Patient data is not included in this repository to protect privacy. Sample data structure is provided in the documentation.
