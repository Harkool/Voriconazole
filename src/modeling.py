"""
Machine learning modeling utilities for voriconazole PK analysis.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_random_forest(X, y, test_size=0.2, random_state=42, **rf_params):
    """
    Train a Random Forest model for PK parameter prediction.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    **rf_params : dict
        Additional parameters for RandomForestRegressor
        
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and metrics
    """
    # Default RF parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': random_state
    }
    default_params.update(rf_params)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': calculate_metrics(y_train, y_train_pred),
        'test': calculate_metrics(y_test, y_test_pred),
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics
    }


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # P30 (Percentage within ±30%)
    relative_errors = np.abs((y_true - y_pred) / y_true)
    p30 = (relative_errors <= 0.3).sum() / len(y_true) * 100
    
    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'P30': p30
    }


def cross_validate_model(X, y, model=None, cv=5, random_state=42):
    """
    Perform cross-validation on the model.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    model : estimator object
        Model to cross-validate (default: RandomForestRegressor)
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std()
    }


def predict_concentration(model, X, dose, tau=12):
    """
    Predict voriconazole concentration from CL/F.
    
    C_ss = Dose / (CL/F * tau)
    
    Parameters:
    -----------
    model : trained model
        Model that predicts CL/F
    X : array-like
        Feature matrix
    dose : float or array-like
        Dose in mg
    tau : float
        Dosing interval in hours
        
    Returns:
    --------
    array-like
        Predicted concentrations
    """
    cl_f_pred = model.predict(X)
    concentration = dose / (cl_f_pred * tau)
    return concentration


def get_feature_importance(model, feature_names, top_n=None):
    """
    Get feature importance from trained model.
    
    Parameters:
    -----------
    model : trained model
        Model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Return top N features (None for all)
        
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    import pandas as pd
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    return importance_df


def optimize_threshold(y_true, y_pred, target_metric='p30', threshold_range=(0.2, 0.4)):
    """
    Optimize prediction threshold for classification tasks.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    target_metric : str
        Metric to optimize
    threshold_range : tuple
        Range of thresholds to test
        
    Returns:
    --------
    dict
        Optimal threshold and corresponding metric value
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
    scores = []
    
    for threshold in thresholds:
        relative_errors = np.abs((y_true - y_pred) / y_true)
        score = (relative_errors <= threshold).sum() / len(y_true) * 100
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    
    return {
        'optimal_threshold': thresholds[optimal_idx],
        'optimal_score': scores[optimal_idx],
        'all_thresholds': thresholds,
        'all_scores': scores
    }


if __name__ == "__main__":
    # Example usage
    print("Modeling utilities for voriconazole PK analysis")
