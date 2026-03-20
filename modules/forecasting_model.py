"""
Forecasting Module
------------------
Forecast future prices using Exponential Smoothing with advanced seasonality detection.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


def _build_time_features(index):
    """Create deterministic calendar features known for future dates."""

    features = pd.DataFrame(index=index)
    day_of_week = index.dayofweek
    month = index.month
    day_of_year = index.dayofyear

    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    features["doy_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
    features["is_month_start"] = index.is_month_start.astype(int)
    features["is_month_end"] = index.is_month_end.astype(int)

    return features.astype(float)


def _build_market_profile(history_df):
    """Summarize recurring market-structure patterns from historical data."""

    profile_columns = ["Market_Count", "Price_STD", "Price_Range"]
    history = history_df.copy()
    history["day_of_week"] = history.index.dayofweek
    history["month"] = history.index.month

    profile = {}
    for column in profile_columns:
        if column not in history.columns:
            continue

        profile[column] = {
            "weekday": history.groupby("day_of_week")[column].mean().to_dict(),
            "month": history.groupby("month")[column].mean().to_dict(),
            "default": float(history[column].mean())
        }

    return profile


def _build_exogenous_features(index, market_profile):
    """Create model features available at forecast time."""

    features = _build_time_features(index)
    day_of_week = pd.Series(index.dayofweek, index=index)
    month = pd.Series(index.month, index=index)

    for column, values in market_profile.items():
        expected_column = f"expected_{column.lower()}"
        weekday_component = day_of_week.map(values["weekday"]).fillna(values["default"])
        month_component = month.map(values["month"]).fillna(values["default"])
        features[expected_column] = ((weekday_component + month_component) / 2).astype(float)

    return features.astype(float)


def _fit_forecasting_model(series, seasonal_period, exog=None, selected_config=None):
    """Fit a forecasting model and return both the fitted model and its config."""

    if selected_config is not None:
        model_type = selected_config["type"]

        if model_type == "sarima":
            model = SARIMAX(
                series,
                exog=exog,
                order=selected_config["order"],
                seasonal_order=selected_config["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            return model.fit(disp=False), selected_config

        if model_type == "exp_smoothing":
            seasonal = selected_config["seasonal"]
            model_kwargs = {
                "trend": selected_config["trend"],
                "seasonal": seasonal,
                "initialization_method": "estimated"
            }
            if seasonal is not None:
                model_kwargs["seasonal_periods"] = selected_config["seasonal_periods"]

            model = ExponentialSmoothing(series, **model_kwargs)
            return model.fit(optimized=True, use_boxcox=False), selected_config

        model = ExponentialSmoothing(
            series,
            trend=selected_config["trend"],
            seasonal=None,
            damped_trend=selected_config["damped_trend"]
        )
        return model.fit(optimized=True), selected_config

    try:
        auto_model = auto_arima(
            series,
            X=exog,
            seasonal=True,
            m=seasonal_period,
            max_order=5,
            max_seasonal_order=2,
            max_p=5,
            max_d=2,
            max_q=5,
            max_D=1,
            stepwise=True,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            information_criterion='aic'
        )

        config = {
            "type": "sarima",
            "order": auto_model.order,
            "seasonal_order": auto_model.seasonal_order,
            "label": (
                f"SARIMAX{auto_model.order}x{auto_model.seasonal_order}"
                if exog is not None else
                f"SARIMA{auto_model.order}x{auto_model.seasonal_order}"
            )
        }

        model = SARIMAX(
            series,
            exog=exog,
            order=config["order"],
            seasonal_order=config["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        return model_fit, config

    except Exception as sarima_error:
        print(f"SARIMA failed: {str(sarima_error)}. Trying Exponential Smoothing...")

    try:
        config = {
            "type": "exp_smoothing",
            "trend": "add",
            "seasonal": "add",
            "seasonal_periods": seasonal_period,
            "label": f"Exponential Smoothing (trend=add, seasonal=add, period={seasonal_period})"
        }
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_period,
            initialization_method='estimated'
        )
        model_fit = model.fit(optimized=True, use_boxcox=False)
        return model_fit, config

    except Exception as exp_error:
        print(f"Exponential Smoothing failed: {str(exp_error)}. Using damped trend model...")

    config = {
        "type": "damped_trend",
        "trend": "add",
        "damped_trend": True,
        "label": "Exponential Smoothing (trend=add, damped_trend=True)"
    }
    model = ExponentialSmoothing(series, trend='add', seasonal=None, damped_trend=True)
    model_fit = model.fit(optimized=True)
    return model_fit, config


def _forecast_steps(model_fit, steps, index, exog=None):
    """Generate forecasts for a fixed number of steps and align them to an index."""

    try:
        predicted = model_fit.get_forecast(steps=steps, exog=exog).predicted_mean
    except TypeError:
        predicted = model_fit.get_forecast(steps=steps).predicted_mean
    except Exception:
        try:
            predicted = model_fit.forecast(steps=steps, exog=exog)
        except TypeError:
            predicted = model_fit.forecast(steps=steps)

    return pd.Series(np.asarray(predicted), index=index)


def forecast_prices(df, forecast_days=365, test_size=0.2):
    """
    Forecast future prices using advanced Exponential Smoothing with seasonality detection.
    
    PURPOSE: Predicts agricultural commodity prices for the entire year 2026 using historical 2025 data.
    
    HOW IT WORKS:
    1. Takes historical price data from 2025
    2. Detects seasonality period (7, 14, 30 days)
    3. Splits into 80% training and 20% testing
    4. Uses Exponential Smoothing with trend and seasonality
    5. Falls back to SARIMA if needed
    6. Predicts 365 days into the future (entire year 2026)
    7. Provides realistic confidence intervals (95% CI)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Time series data with Modal_Price index (2025 historical prices)
    forecast_days : int
        Number of days to forecast ahead (default: 365 = full year 2026)
    test_size : float
        Proportion of data to use as test set (0.2 = 20%)
    
    Returns:
    --------
    tuple : (forecast_values, conf_lower, conf_upper, forecast_plot, metrics_dict)
    """

    print("\nStarting Advanced Forecasting Model...")

    series = df["Modal_Price"].copy()
    
    # -----------------------------------------------
    # Step 1: Detect Seasonality Period
    # -----------------------------------------------
    print("Detecting seasonality period...")
    
    seasonal_period = None
    
    # If we have enough data, try detecting seasonality at common periods
    if len(series) > 60:
        # Try weekly (7) and bi-weekly (14) patterns
        for period in [7, 14, 30]:
            if len(series) > period * 4:  # Need at least 4 cycles
                try:
                    # Calculate autocorrelation
                    acf_values = np.correlate(
                        series.values - series.mean(),
                        series.values - series.mean(),
                        mode='full'
                    )
                    acf_values = acf_values[len(acf_values)//2:]
                    acf_values = acf_values / acf_values[0]
                    
                    if period < len(acf_values) and abs(acf_values[period]) > 0.2:
                        seasonal_period = period
                        print(f"Weekly seasonality detected at period: {seasonal_period} days")
                        break
                except:
                    pass
    
    # Default to 7-day (weekly) seasonality if not detected
    if seasonal_period is None:
        seasonal_period = 7
        print(f"Using default seasonality period: {seasonal_period} days (weekly)")

    
    # -----------------------------------------------
    # Step 2: Train-Test Split
    # -----------------------------------------------
    train_size = int(len(series) * (1 - test_size))
    train_size = max(train_size, seasonal_period * 2)
    train_size = min(train_size, len(series) - 1)
    train_data = series[:train_size]
    test_data = series[train_size:]

    train_profile = _build_market_profile(df.iloc[:train_size])
    all_exog_train_profile = _build_exogenous_features(df.index, train_profile)
    train_exog = all_exog_train_profile.iloc[:train_size]
    test_exog = all_exog_train_profile.iloc[train_size:]
    
    print(f"Train-test split: Train={len(train_data)}, Test={len(test_data)}")
    
    # -----------------------------------------------
    # Step 3: Fit SARIMAX Model (Primary) or Exponential Smoothing
    # -----------------------------------------------
    print("Training forecasting model with engineered features...")
    
    model_fit, model_config = _fit_forecasting_model(train_data, seasonal_period, exog=train_exog)
    print(f"{model_config['label']} selected")
    
    # -----------------------------------------------
    # Step 4: Validate on Test Set
    # -----------------------------------------------
    print("Validating model on test set...")
    
    test_pred_values = _forecast_steps(model_fit, len(test_data), test_data.index, exog=test_exog)
    
    # Calculate accuracy metrics
    rmse = np.sqrt(mean_squared_error(test_data, test_pred_values))
    mae = mean_absolute_error(test_data, test_pred_values)
    non_zero_actuals = test_data.replace(0, np.nan)
    mape = np.nanmean(np.abs((test_data - test_pred_values) / non_zero_actuals)) * 100
    if np.isnan(mape):
        mape = 0.0
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # -----------------------------------------------
    # Step 5: Forecast Future Values with CI
    # -----------------------------------------------
    print("Generating forecast with confidence intervals...")
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=forecast_days+1, freq="D")[1:]
    full_profile = _build_market_profile(df)
    full_exog = _build_exogenous_features(df.index, full_profile)
    future_exog = _build_exogenous_features(future_dates, full_profile)
    final_model_fit, _ = _fit_forecasting_model(series, seasonal_period, exog=full_exog, selected_config=model_config)

    # -----------------------------------------------
    # Step 6: Create High-Quality Visualization with CI
    # -----------------------------------------------
    residual_std = np.std(test_data.values - test_pred_values.values)
    z_score = 1.96

    try:
        forecast_result = final_model_fit.get_forecast(steps=forecast_days, exog=future_exog)
        forecast = pd.Series(np.asarray(forecast_result.predicted_mean), index=future_dates)

        try:
            conf_int = forecast_result.conf_int(alpha=0.05)
            conf_lower = pd.Series(conf_int.iloc[:, 0].values, index=future_dates)
            conf_upper = pd.Series(conf_int.iloc[:, 1].values, index=future_dates)
        except Exception:
            conf_lower = forecast - (z_score * residual_std)
            conf_upper = forecast + (z_score * residual_std)

    except Exception:
        forecast = _forecast_steps(final_model_fit, forecast_days, future_dates, exog=future_exog)
        conf_lower = forecast - (z_score * residual_std)
        conf_upper = forecast + (z_score * residual_std)

    conf_lower = pd.Series(np.maximum(conf_lower, 0), index=future_dates)
    conf_upper = pd.Series(np.maximum(conf_upper, 0), index=future_dates)
    
    os.makedirs("static/plots", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical data
    ax.plot(df.index, df["Modal_Price"], label="Historical Price", 
            color="darkblue", linewidth=2.5, alpha=0.8)
    
    # Plot test predictions vs actual
    test_dates = df.index[train_size:]
    ax.plot(test_dates, test_pred_values, label="Model Testing Predictions", 
            color="green", linewidth=2, linestyle='--', alpha=0.7)
    
    # Plot forecast with confidence interval
    ax.plot(future_dates, forecast, label="2026 Forecast", 
            color="red", linewidth=3, alpha=0.9)
    ax.fill_between(
        future_dates,
        conf_lower,
        conf_upper,
        alpha=0.25,
        color="red",
        label="95% Confidence Interval"
    )
    
    # Formatting
    ax.set_title("Agricultural Commodity Price Forecast with Confidence Intervals", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=13, fontweight='bold')
    ax.set_ylabel("Modal Price (₹)", fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis for currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    plt.tight_layout()
    
    plot_path = "static/plots/forecast.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Forecast visualization created")
    
    # -----------------------------------------------
    # Step 7: Prepare Metrics Dictionary
    # -----------------------------------------------
    metrics = {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'mape': round(mape, 2),
        'model_order': model_config['label'],
        'feature_basis': 'Modal_Price + calendar signals + expected market activity and dispersion',
        'test_samples': len(test_data),
        'train_samples': len(train_data)
    }
    
    return forecast, conf_lower, conf_upper, plot_path, metrics