"""
Moving Average Module
---------------------
Computes moving averages for price smoothing.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


def calculate_moving_average(df):

    print("\nCalculating Moving Averages...")

    df["MA_7"] = df["Modal_Price"].rolling(window=7).mean()
    df["MA_14"] = df["Modal_Price"].rolling(window=14).mean()

    os.makedirs("static/plots", exist_ok=True)

    plt.figure(figsize=(12,6))

    plt.plot(df.index, df["Modal_Price"], label="Actual Price", color="blue")
    plt.plot(df.index, df["MA_7"], label="7 Day Avg", color="red")
    plt.plot(df.index, df["MA_14"], label="14 Day Avg", color="green")

    plt.title("Moving Average Analysis")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (₹)")

    plt.legend()
    plt.grid(True)

    plot_path = "static/plots/moving_average.png"

    plt.savefig(plot_path)
    plt.close()

    print("Moving average visualization created")

    return df, plot_path