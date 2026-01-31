import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from src.data_processing import (
    encode_genotype, 
    stratify_inflammation,
    prepare_features,
    get_summary_statistics
)
from src.modeling import (
    train_random_forest,
    calculate_metrics,
    get_feature_importance
)
from src.visualization import (
    plot_distribution_by_genotype,
    plot_predictions,
    plot_feature_importance
)


def main():
    """Run the complete analysis workflow."""
    # 1. Load sample data
    print("\n[Step 1/5] Loading sample data...")
    data_path = Path(__file__).parent.parent / 'data' / 'example.csv'
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Columns: {', '.join(df.columns)}")
    except FileNotFoundError:
        print("✗ Sample data not found. Please check data/sample_data_structure.csv")
        return
    
    # 2. Data preprocessing
    print("\n[Step 2/5] Preprocessing data...")
    df = encode_genotype(df)
    df = stratify_inflammation(df, cutoff=100)
    print(f"✓ Genotypes: {df['Genotype_Label'].value_counts().to_dict()}")
    print(f"✓ Inflammation groups: {df['Inflammation_Status'].value_counts().to_dict()}")
    
    # 3. Summary statistics
    print("\n[Step 3/5] Calculating summary statistics...")
    summary = get_summary_statistics(
        df, 
        group_cols='Genotype_Label',
        value_cols=['CL/F', 'DV', 'CRP']
    )
    print("✓ Summary statistics by genotype:")
    print(summary)
    
    # 4. Model training (if enough data)
    if len(df) >= 5:
        print("\n[Step 4/5] Training Random Forest model...")
        feature_cols = ['CYP2C19 genotype', 'CRP']
        X, y = prepare_features(df, feature_cols, 'CL/F')
        
        try:
            results = train_random_forest(
                X, y,
                test_size=0.3,
                n_estimators=50,
                random_state=42
            )
            
            print("✓ Model trained successfully")
            print(f"\nTraining Performance:")
            print(f"  R²:   {results['metrics']['train']['R2']:.3f}")
            print(f"  MAE:  {results['metrics']['train']['MAE']:.3f}")
            print(f"  RMSE: {results['metrics']['train']['RMSE']:.3f}")
            
            print(f"\nTest Performance:")
            print(f"  R²:   {results['metrics']['test']['R2']:.3f}")
            print(f"  MAE:  {results['metrics']['test']['MAE']:.3f}")
            print(f"  RMSE: {results['metrics']['test']['RMSE']:.3f}")
            print(f"  MAPE: {results['metrics']['test']['MAPE']:.2f}%")
            print(f"  P30:  {results['metrics']['test']['P30']:.2f}%")
            
            # Feature importance
            importance_df = get_feature_importance(results['model'], feature_cols)
            print("\n✓ Feature Importance:")
            for _, row in importance_df.iterrows():
                print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")
                
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            print("  Note: Sample data is too small for reliable modeling")
    else:
        print("\n[Step 4/5] Skipping model training (insufficient data)")
        print("  Note: This is sample data. Use real data for modeling.")
    
    # 5. Next steps
    print("\n[Step 5/5] Workflow complete!")
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Replace sample data with your real dataset in data/")
    print("2. Adjust parameters in this script for your analysis")
    print("3. Run full analysis with larger dataset")
    print("4. Generate publication-quality figures")
    print("5. Save models and results")
    print("\nFor detailed usage, see docs/USER_GUIDE.md")
    print("=" * 70)
    print("\nContact: lenhartkoo@foxmail.com")
    print("GitHub: https://github.com/Harkool/Voriconazole")
    print("=" * 70)


if __name__ == "__main__":
    main()
