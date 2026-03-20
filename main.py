from flask import Flask, render_template, request
from pymongo import MongoClient

# Import project modules
from modules.data_extraction import extract_data
from modules.data_cleaning import clean_data
from modules.visualization import plot_price_trend
from modules.moving_average import calculate_moving_average
from modules.outlier_detection import detect_outliers
from modules.decomposition import decompose_time_series
from modules.forecasting_model import forecast_prices


app = Flask(__name__)

# ---------------------------------------
# MongoDB Connection
# ---------------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["agricultureDB"]
collection = db["prices2025"]


# ---------------------------------------
# Homepage
# ---------------------------------------
@app.route("/")
def index():

    # Get list of commodities
    commodities = collection.distinct("Commodity")
    commodities.sort()

    return render_template(
        "index.html",
        commodities=commodities
    )


# ---------------------------------------
# Forecast Route
# ---------------------------------------
@app.route("/forecast", methods=["POST"])
def forecast():

    commodity = request.form["commodity"]

    print(f"\nSelected Commodity: {commodity}")

    # ---------------------------------------
    # Step 1: Data Extraction
    # ---------------------------------------
    df = extract_data(commodity)

    # ---------------------------------------
    # Step 2: Data Cleaning
    # ---------------------------------------
    df = clean_data(df)

    # ---------------------------------------
    # Step 3: Visualization
    # ---------------------------------------
    trend_plot = plot_price_trend(df)

    # ---------------------------------------
    # Step 4: Moving Average
    # ---------------------------------------
    df, ma_plot = calculate_moving_average(df)

    # ---------------------------------------
    # Step 5: Outlier Detection
    # ---------------------------------------
    df, outlier_plot = detect_outliers(df)

    # ---------------------------------------
    # Step 6: Time Series Decomposition
    # ---------------------------------------
    df, decomposition_plot = decompose_time_series(df, period=30)

    # ---------------------------------------
    # Step 7: Forecasting (365 days = Full Year 2026)
    # -----------------------------------------------
    # Predict prices for the entire year 2026 (365 days ahead from last data point in 2025)
    forecast_values, conf_lower, conf_upper, forecast_plot, metrics = forecast_prices(df, forecast_days=365)

    # Convert forecast to list for template with currency formatting
    forecast_list = forecast_values.tolist()
    forecast_list = [f"₹{round(float(x), 2):.2f}" for x in forecast_list]
    
    # Convert confidence intervals to lists
    conf_lower_list = conf_lower.tolist()
    conf_upper_list = conf_upper.tolist()
    conf_lower_list = [f"₹{round(float(x), 2):.2f}" for x in conf_lower_list]
    conf_upper_list = [f"₹{round(float(x), 2):.2f}" for x in conf_upper_list]

    return render_template(
        "results.html",
        commodity=commodity,
        trend_plot=trend_plot,
        ma_plot=ma_plot,
        outlier_plot=outlier_plot,
        decomposition_plot=decomposition_plot,
        forecast_plot=forecast_plot,
        forecast=forecast_list,
        conf_lower=conf_lower_list,
        conf_upper=conf_upper_list,
        metrics=metrics
    )


# ---------------------------------------
# Run Flask App
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8000)