"""
Outlier Detection Module
------------------------
Detects price anomalies using IQR.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


def detect_outliers(df):

    print("\nDetecting Outliers...")

    Q1 = df["Modal_Price"].quantile(0.25)
    Q3 = df["Modal_Price"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["Outlier"] = (df["Modal_Price"] < lower) | (df["Modal_Price"] > upper)

    os.makedirs("static/plots", exist_ok=True)

    plt.figure(figsize=(12,6))

    normal = df[df["Outlier"] == False]
    outliers = df[df["Outlier"] == True]

    plt.scatter(normal.index, normal["Modal_Price"], color="blue", label="Normal")
    plt.scatter(outliers.index, outliers["Modal_Price"], color="red", label="Outlier")

    plt.title("Outlier Detection")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (₹)")

    plt.legend()
    plt.grid(True)

    plot_path = "static/plots/outliers.png"

    plt.savefig(plot_path)
    plt.close()

    print("Outlier visualization created")

    # Actually handle the outliers for better forecasting by replacing them with interpolated values
    # We replace outlier values with NaN, then interpolate linearly
    import pandas as pd
    df.loc[df["Outlier"], "Modal_Price"] = pd.NA
    df["Modal_Price"] = df["Modal_Price"].interpolate(method='linear').bfill().ffill()

    return df, plot_path