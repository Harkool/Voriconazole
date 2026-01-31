import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import *
from src.modeling import *
from src.visualization import *


def run_complete_analysis(data_path, output_dir='results'):
    """
    Run complete PK analysis pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to input CSV file
    output_dir : str
        Directory for saving results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("VORICONAZOLE PHARMACOKINETIC ANALYSIS")
    print("=" * 70)
    
    # ========== 1. Data Loading ==========
    print("\n[1] Loading data...")
    df = load_data(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} variables")
    
    # ========== 2. Data Preprocessing ==========
    print("\n[2] Preprocessing data...")
    df = encode_genotype(df)
    df = stratify_inflammation(df, cutoff=100)
    
    print(f"✓ Genotype distribution:")
    for geno, count in df['Genotype_Label'].value_counts().items():
        print(f"  {geno}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"✓ Inflammation status:")
    for status, count in df['Inflammation_Status'].value_counts().items():
        print(f"  {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # ========== 3. Exploratory Data Analysis ==========
    print("\n[3] Performing exploratory data analysis...")
    
    # Summary statistics
    summary = get_summary_statistics(
        df,
        group_cols='Genotype_Label',
        value_cols=['CL/F', 'DV', 'CRP']
    )
    print("✓ Summary statistics calculated")
    
    # Visualizations
    print("✓ Generating EDA plots...")
    plot_distribution_by_genotype(
        df, 'CL/F',
        save_path=output_path / 'clf_distribution.png'
    )
    
    plot_inflammation_comparison(
        df, 'CL/F',
        save_path=output_path / 'inflammation_comparison.png'
    )
    
    # ========== 4. Feature Preparation ==========
    print("\n[4] Preparing features for modeling...")
    feature_cols = ['CYP2C19 genotype', 'CRP', 'Age', 'Weight']
    available_features = [col for col in feature_cols if col in df.columns]
    
    X, y = prepare_features(df, available_features, 'CL/F')
    print(f"✓ Features: {available_features}")
    print(f"✓ Training samples: {len(X)}")
    
    # ========== 5. Model Training ==========
    print("\n[5] Training Random Forest model...")
    results = train_random_forest(
        X, y,
        test_size=0.2,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    print("✓ Model trained successfully")
    print(f"\nTraining Performance:")
    for metric, value in results['metrics']['train'].items():
        print(f"  {metric:6s}: {value:.3f}")
    
    print(f"\nTest Performance:")
    for metric, value in results['metrics']['test'].items():
        print(f"  {metric:6s}: {value:.3f}")
    
    # ========== 6. Feature Importance ==========
    print("\n[6] Analyzing feature importance...")
    importance_df = get_feature_importance(results['model'], available_features)
    print("✓ Feature importance:")
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")
    
    plot_feature_importance(
        importance_df,
        save_path=output_path / 'feature_importance.png'
    )
    
    # ========== 7. Prediction Visualization ==========
    print("\n[7] Generating prediction plots...")
    
    plot_predictions(
        results['y_test'],
        results['y_test_pred'],
        title='Test Set Predictions',
        metrics=results['metrics']['test'],
        save_path=output_path / 'test_predictions.png'
    )
    
    # Residual analysis
    plot_residuals(
        results['y_test'],
        results['y_test_pred'],
        save_path=output_path / 'residual_analysis.png'
    )
    
    # ========== 8. Save Results ==========
    print("\n[8] Saving results...")
    
    # Save model
    import joblib
    joblib.dump(results['model'], output_path / 'rf_model.pkl')
    print(f"✓ Model saved to {output_path / 'rf_model.pkl'}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'Observed': results['y_test'],
        'Predicted': results['y_test_pred'],
        'Error': results['y_test'] - results['y_test_pred'],
        'Relative_Error': (results['y_test'] - results['y_test_pred']) / results['y_test']
    })
    pred_df.to_csv(output_path / 'predictions.csv', index=False)
    print(f"✓ Predictions saved to {output_path / 'predictions.csv'}")
    
    # Save metrics
    import json
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"✓ Metrics saved to {output_path / 'metrics.json'}")
    
    # Save summary
    summary.to_csv(output_path / 'summary_statistics.csv')
    print(f"✓ Summary statistics saved to {output_path / 'summary_statistics.csv'}")
    
    # ========== 9. Summary ==========
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  Test R²:   {results['metrics']['test']['R2']:.3f}")
    print(f"  Test MAE:  {results['metrics']['test']['MAE']:.3f}")
    print(f"  Test MAPE: {results['metrics']['test']['MAPE']:.2f}%")
    print(f"  Test P30:  {results['metrics']['test']['P30']:.2f}%")
    
    print(f"\nOutput files saved to: {output_path}")
    print(f"  - Model: rf_model.pkl")
    print(f"  - Predictions: predictions.csv")
    print(f"  - Metrics: metrics.json")
    print(f"  - Figures: *.png")
    
    print("\n" + "=" * 70)
    print("Contact: lenhartkoo@foxmail.com")
    print("GitHub: https://github.com/Harkool/Voriconazole")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = Path(__file__).parent.parent / 'data' / 'example.csv'
        print(f"No data file specified. Using sample data: {data_file}")
        print("Usage: python complete_analysis.py <path_to_your_data.csv>\n")
    
    try:
        results = run_complete_analysis(data_file)
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
