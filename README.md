# Voriconazole Pharmacokinetic Analysis

A comprehensive pharmacokinetic (PK) analysis project for voriconazole, incorporating exploratory data analysis (EDA), machine learning modeling, and external validation.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This project analyzes voriconazole pharmacokinetics with a focus on:
- CYP2C19 genotype effects on drug clearance (CL/F)
- Inflammation status (CRP levels) impact on PK parameters
- Machine learning models for concentration prediction
- External validation of predictive models

## 🗂️ Project Structure

```
Voriconazole/
├── data/                   # Data files (not included for privacy)
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_processing.py  # Data preprocessing utilities
│   ├── modeling.py         # Machine learning models
│   └── visualization.py    # Plotting functions
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
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Harkool/Voriconazole.git
cd Voriconazole
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run example analysis:
```bash
python examples/example_workflow.py
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

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

## 📝 Usage Example

```python
from src.data_processing import load_data, encode_genotype, stratify_inflammation
from src.modeling import train_random_forest
from src.visualization import plot_predictions

# Load and prepare data
df = load_data('data/your_data.csv')
df = encode_genotype(df)
df = stratify_inflammation(df, cutoff=100)

# Train model
feature_cols = ['CYP2C19 genotype', 'CRP', 'Age', 'Weight']
results = train_random_forest(df[feature_cols], df['CL/F'])

# Visualize results
plot_predictions(
    results['y_test'], 
    results['y_test_pred'],
    metrics=results['metrics']['test'],
    save_path='figures/predictions.png'
)

# Print performance
print(f"Test R²: {results['metrics']['test']['R2']:.3f}")
print(f"Test MAE: {results['metrics']['test']['MAE']:.3f}")
```

## 📖 Documentation

- [User Guide](docs/USER_GUIDE.md) - Detailed usage instructions
- [API Reference](docs/API_REFERENCE.md) - Function documentation
- [Methodology](docs/METHODOLOGY.md) - Statistical methods

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**Liu Hao**
- Institution: China Pharmaceutical University
- Email: lenhartkoo@foxmail.com
- GitHub: [@Harkool](https://github.com/Harkool)

## 🙏 Acknowledgments

- China Pharmaceutical University
- Research team members
- Open-source community

## 📮 Contact

For questions or collaboration opportunities, please contact:
- Email: lenhartkoo@foxmail.com
- GitHub Issues: [Report issues](https://github.com/Harkool/Voriconazole/issues)

## 🔖 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{liu2025voriconazole,
  title={Voriconazole Pharmacokinetic Analysis with Machine Learning},
  author={Liu, Hao},
  year={2025},
  institution={China Pharmaceutical University},
  publisher={GitHub},
  howpublished={\url{https://github.com/Harkool/Voriconazole}}
}
```

---

**Note**: Patient data is not included in this repository to protect privacy. Sample data structure is provided in the documentation.
