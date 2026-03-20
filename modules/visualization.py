"""
Visualization Module
--------------------
Generates time series visualization.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


def plot_price_trend(df):

    print("\nGenerating Time Series Visualization...")

    os.makedirs("static/plots", exist_ok=True)

    plt.figure(figsize=(12,6))

    plt.plot(df.index, df["Modal_Price"], color="blue")

    plt.title("Commodity Price Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (₹)")

    plt.grid(True)

    plot_path = "static/plots/price_trend.png"

    plt.savefig(plot_path)
    plt.close()

    print("Visualization created successfully")

    return plot_path