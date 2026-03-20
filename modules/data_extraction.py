"""
Data Extraction Module
----------------------
This module retrieves commodity price data from MongoDB
based on the commodity selected by the user.
"""

from pymongo import MongoClient
import pandas as pd


def extract_data(commodity):
    """
    Retrieve commodity price data from MongoDB.

    Parameters
    ----------
    commodity : str
        Commodity selected by the user.

    Returns
    -------
    pandas.DataFrame
        Raw dataset retrieved from MongoDB.
    """

    print("\nConnecting to MongoDB...")

    # --------------------------------
    # MongoDB Connection
    # --------------------------------
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agricultureDB"]
    collection = db["prices2025"]

    print("Connected to MongoDB")

    # --------------------------------
    # Query MongoDB
    # --------------------------------
    print(f"Retrieving data for commodity: {commodity}")

    data = collection.find(
        {"Commodity": commodity},
        {
            "Arrival_Date": 1,
            "Modal_Price": 1,
            "Market": 1,
            "_id": 0
        }
    )

    # --------------------------------
    # Convert to pandas DataFrame
    # --------------------------------
    df = pd.DataFrame(list(data))

    print(f"Records retrieved: {len(df)}")

    return df