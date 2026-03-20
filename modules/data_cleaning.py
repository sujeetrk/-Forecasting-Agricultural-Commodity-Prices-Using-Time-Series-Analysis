"""
Data Cleaning Module
--------------------
Cleans and prepares the commodity price dataset for time series analysis.
"""

import pandas as pd


def clean_data(df):

    print("\nStarting Data Cleaning...")

    # --------------------------------
    # Remove Missing Values
    # --------------------------------
    print("Checking missing values...")
    df = df.dropna()

    # --------------------------------
    # Convert Date Column
    # --------------------------------
    print("Converting Arrival_Date to datetime...")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

    # --------------------------------
    # Remove Duplicate Records
    # --------------------------------
    print("Removing duplicate records...")
    df = df.drop_duplicates()

    # --------------------------------
    # Remove Unrealistic Prices
    # --------------------------------
    print("Removing unrealistic price spikes...")
    df = df[df["Modal_Price"] < 100000]

    # --------------------------------
    # Aggregate Prices Per Day
    # (important because multiple markets exist)
    # --------------------------------
    print("Aggregating daily prices...")

    df = df.groupby("Arrival_Date").agg(
        Modal_Price=("Modal_Price", "mean"),
        Market_Count=("Market", "nunique"),
        Price_STD=("Modal_Price", "std"),
        Price_Min=("Modal_Price", "min"),
        Price_Max=("Modal_Price", "max")
    ).reset_index()

    df["Price_STD"] = df["Price_STD"].fillna(0)
    df["Price_Range"] = df["Price_Max"] - df["Price_Min"]

    # --------------------------------
    # Sort Data
    # --------------------------------
    df = df.sort_values(by="Arrival_Date")

    # --------------------------------
    # Set Date Index
    # --------------------------------
    df = df.set_index("Arrival_Date")

    # --------------------------------
    # Set Daily Frequency
    # --------------------------------
    df = df.asfreq("D")

    # Fill missing days
    numeric_columns = [
        "Modal_Price",
        "Market_Count",
        "Price_STD",
        "Price_Min",
        "Price_Max",
        "Price_Range"
    ]
    df[numeric_columns] = df[numeric_columns].interpolate(method="time").bfill().ffill()
    df["Market_Count"] = df["Market_Count"].clip(lower=1)

    print("Data Cleaning Completed")

    return df