"""
Visualization utilities for voriconazole PK analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def setup_plot_style():
    """
    Set up consistent plotting style.
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


def plot_distribution_by_genotype(df, value_col, genotype_col='Genotype_Label', 
                                   figsize=(14, 6), save_path=None):
    """
    Plot distribution of a variable by genotype.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Column to plot
    genotype_col : str
        Genotype column name
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Violin plot
    sns.violinplot(
        x=genotype_col, y=value_col, data=df,
        order=['PM', 'IM', 'NM'], ax=axes[0]
    )
    axes[0].set_title(f'{value_col} Distribution by Genotype', fontweight='bold')
    axes[0].set_xlabel('CYP2C19 Genotype')
    axes[0].set_ylabel(value_col)
    
    # Box plot with swarm
    sns.boxplot(
        x=genotype_col, y=value_col, data=df,
        order=['PM', 'IM', 'NM'], ax=axes[1]
    )
    sns.swarmplot(
        x=genotype_col, y=value_col, data=df,
        order=['PM', 'IM', 'NM'], color='black', size=3, alpha=0.3, ax=axes[1]
    )
    axes[1].set_title(f'{value_col} by Genotype (Box + Swarm)', fontweight='bold')
    axes[1].set_xlabel('CYP2C19 Genotype')
    axes[1].set_ylabel(value_col)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(y_true, y_pred, title='Predicted vs Observed',
                     xlabel='Observed', ylabel='Predicted',
                     metrics=None, figsize=(8, 8), save_path=None):
    """
    Plot predicted vs observed values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    metrics : dict
        Dictionary of metrics to display
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=80, ax=ax)
    
    # Identity line
    max_val = max(y_true.max(), y_pred.max()) * 1.1
    min_val = min(y_true.min(), y_pred.min()) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')
    
    # ±30% error band
    x_range = np.linspace(min_val, max_val, 100)
    ax.fill_between(x_range, x_range * 0.7, x_range * 1.3, 
                    color='gray', alpha=0.2, label='±30% error')
    
    # Add metrics to title if provided
    if metrics:
        title_text = f"{title}\n"
        title_text += f"R² = {metrics.get('R2', 0):.3f}, "
        title_text += f"MAE = {metrics.get('MAE', 0):.3f}, "
        title_text += f"RMSE = {metrics.get('RMSE', 0):.3f}"
        ax.set_title(title_text, fontweight='bold')
    else:
        ax.set_title(title, fontweight='bold')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance_df, top_n=15, figsize=(10, 8), save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    setup_plot_style()
    
    # Select top N features
    plot_df = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Horizontal bar plot
    sns.barplot(
        data=plot_df,
        y='Feature',
        x='Importance',
        palette='viridis',
        ax=ax
    )
    
    ax.set_title(f'Top {top_n} Feature Importance', fontweight='bold', fontsize=14)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, figsize=(12, 5), save_path=None):
    """
    Plot residual analysis.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    setup_plot_style()
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_inflammation_comparison(df, value_col, inflammation_col='Inflammation_Status',
                                 genotype_col='Genotype_Label', figsize=(14, 6), save_path=None):
    """
    Plot comparison between inflammation groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Column to plot
    inflammation_col : str
        Inflammation status column
    genotype_col : str
        Genotype column
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # By inflammation status
    sns.violinplot(
        x=inflammation_col, y=value_col, data=df,
        palette='Set2', ax=axes[0]
    )
    axes[0].set_title(f'{value_col} by Inflammation Status', fontweight='bold')
    axes[0].set_xlabel('Inflammation Status')
    axes[0].set_ylabel(value_col)
    
    # By genotype and inflammation
    sns.boxplot(
        x=genotype_col, y=value_col, hue=inflammation_col,
        data=df, order=['PM', 'IM', 'NM'], ax=axes[1]
    )
    axes[1].set_title(f'{value_col} by Genotype and Inflammation', fontweight='bold')
    axes[1].set_xlabel('CYP2C19 Genotype')
    axes[1].set_ylabel(value_col)
    axes[1].legend(title='Inflammation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_heatmap(df, columns, figsize=(10, 8), save_path=None):
    """
    Plot correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to include in correlation
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    setup_plot_style()
    
    # Calculate correlation
    corr = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities for voriconazole PK analysis")
