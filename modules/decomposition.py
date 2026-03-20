"""
Time Series Decomposition Module
---------------------------------
Decomposes time series into trend, seasonal, and residual components.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np


def decompose_time_series(df, period=30):
    """
    Decompose the time series into trend, seasonal, and residual components.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Time series data with Modal_Price as index
    period : int
        Seasonal period (default: 30 days for monthly seasonality)
        
    Returns
    -------
    df : pandas.DataFrame
        Original dataframe (unchanged)
    plot_path : str
        Path to the saved decomposition plot
    """
    
    print("\nPerforming Time Series Decomposition...")
    
    # Ensure we have enough data for decomposition
    if len(df) < period * 2:
        print(f"Warning: Not enough data for decomposition (need at least {period * 2} records)")
        period = max(7, len(df) // 3)  # Fall back to weekly or smaller period
        print(f"Using period: {period}")
    
    try:
        # Perform seasonal decomposition
        # Using additive model: Y = Trend + Seasonal + Residual
        decomposition = seasonal_decompose(
            df["Modal_Price"], 
            model='additive', 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create the plot directory
        os.makedirs("static/plots", exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # Original data
        axes[0].plot(df.index, df["Modal_Price"], color='#2c3e50', linewidth=1.5)
        axes[0].set_title('Original Time Series', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price (₹)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Trend component
        axes[1].plot(df.index, decomposition.trend, color='#e74c3c', linewidth=2)
        axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend (₹)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal component
        axes[2].plot(df.index, decomposition.seasonal, color='#3498db', linewidth=1.5)
        axes[2].set_title(f'Seasonal Component (Period: {period} days)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Seasonal (₹)', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Residual component
        axes[3].plot(df.index, decomposition.resid, color='#27ae60', linewidth=1, alpha=0.7)
        axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[3].set_title('Residual Component (Noise)', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Residual (₹)', fontsize=10)
        axes[3].set_xlabel('Date', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = "static/plots/decomposition.png"
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print("✓ Time series decomposition complete")
        print(f"  - Trend strength: {_calculate_strength(decomposition.trend, df['Modal_Price']):.2%}")
        print(f"  - Seasonal strength: {_calculate_strength(decomposition.seasonal, df['Modal_Price']):.2%}")
        
    except Exception as e:
        print(f"Warning: Decomposition failed ({str(e)}). Creating simple visualization...")
        
        # Fallback: Create a simple plot showing just the original data
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df["Modal_Price"], color='#2c3e50', linewidth=1.5)
        ax.set_title('Time Series (Decomposition Not Available)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (₹)', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "static/plots/decomposition.png"
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
    
    return df, plot_path


def _calculate_strength(component, original):
    """
    Calculate the strength of a component relative to the original series.
    
    Parameters
    ----------
    component : pandas.Series
        Decomposed component (trend or seasonal)
    original : pandas.Series
        Original time series
        
    Returns
    -------
    float
        Strength value between 0 and 1
    """
    try:
        # Remove NaN values
        mask = ~(np.isnan(component) | np.isnan(original))
        component_clean = component[mask]
        original_clean = original[mask]
        
        if len(component_clean) == 0:
            return 0.0
        
        # Calculate variance ratio
        var_component = np.var(component_clean)
        var_original = np.var(original_clean)
        
        if var_original == 0:
            return 0.0
            
        strength = var_component / var_original
        return max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        
    except Exception:
        return 0.0
