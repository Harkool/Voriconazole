"""
Data processing utilities for voriconazole PK analysis.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load voriconazole dataset from CSV.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    return df


def encode_genotype(df, genotype_col='CYP2C19 genotype'):
    """
    Encode CYP2C19 genotype with standardized labels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    genotype_col : str
        Name of genotype column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded genotype
    """
    geno_map = {
        1: 'NM',  # Normal Metabolizer
        2: 'IM',  # Intermediate Metabolizer
        3: 'PM'   # Poor Metabolizer
    }
    
    df = df.copy()
    df['Genotype_Label'] = df[genotype_col].map(geno_map)
    return df


def stratify_inflammation(df, crp_col='CRP', cutoff=100):
    """
    Stratify patients by inflammation status based on CRP levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    crp_col : str
        Name of CRP column
    cutoff : float
        CRP cutoff value (mg/L)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with inflammation status
    """
    df = df.copy()
    df['Inflammation_Status'] = df[crp_col].apply(
        lambda x: 'High CRP' if x > cutoff else 'Low CRP'
    )
    return df


def prepare_features(df, feature_cols, target_col):
    """
    Prepare feature matrix and target vector for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
        
    Returns:
    --------
    tuple
        (X, y) - feature matrix and target vector
    """
    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    return X, y


def calculate_cl_f(df, dose_col='Dose', concentration_col='DV', tau=12):
    """
    Calculate CL/F (apparent clearance) if not already present.
    
    CL/F = Dose / (C_ss * tau)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dose_col : str
        Column name for dose (mg)
    concentration_col : str
        Column name for steady-state concentration (mg/L)
    tau : float
        Dosing interval (hours)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with CL/F column
    """
    df = df.copy()
    if 'CL/F' not in df.columns:
        df['CL/F'] = df[dose_col] / (df[concentration_col] * tau)
    return df


def get_summary_statistics(df, group_cols, value_cols):
    """
    Calculate summary statistics by groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    group_cols : list or str
        Column(s) to group by
    value_cols : list
        Columns to summarize
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    summary = df.groupby(group_cols)[value_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    return summary


def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column to check for outliers
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.Series
        Boolean mask of outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return outliers


if __name__ == "__main__":
    # Example usage
    print("Data processing utilities for voriconazole PK analysis")
