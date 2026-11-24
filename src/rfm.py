"""
rfm.py
------
Loads raw transactions from SQLite, cleans them, and engineers
Recency / Frequency / Monetary (RFM) features per customer.
"""

import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "customers.db")


def load_raw_transactions(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT t.TransactionID, t.CustomerID, t.TransactionDate,
               t.ProductCategory, t.Quantity, t.Amount,
               c.Region, c.SignupDate
        FROM transactions t
        LEFT JOIN customers c ON t.CustomerID = c.CustomerID
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def clean_transactions(df):
    """Handle duplicates, nulls, invalid amounts, and inconsistent text."""
    df = df.copy()
    before = len(df)

    # Standardize category text
    df["ProductCategory"] = df["ProductCategory"].str.strip().str.title()

    # Drop exact duplicate transaction rows
    df = df.drop_duplicates(subset=["TransactionID", "CustomerID", "TransactionDate", "Amount"])

    # Drop rows with missing essential fields
    df = df.dropna(subset=["CustomerID", "TransactionDate", "Amount"])

    # Remove negative / zero amounts (refunds or bad data) — keep as separate
    # metric if needed, but excluded from spend-based RFM for this project
    df = df[df["Amount"] > 0]

    # Parse dates, drop unparseable rows
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df = df.dropna(subset=["TransactionDate"])

    after = len(df)
    print(f"Cleaned transactions: {before} -> {after} rows ({before - after} removed)")
    return df.reset_index(drop=True)


def compute_rfm(df, snapshot_date=None):
    """Compute Recency, Frequency, Monetary per CustomerID."""
    if snapshot_date is None:
        snapshot_date = df["TransactionDate"].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm = df.groupby("CustomerID").agg(
        Recency=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionID", "count"),
        Monetary=("Amount", "sum"),
        AvgOrderValue=("Amount", "mean"),
        Region=("Region", "first"),
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AvgOrderValue"] = rfm["AvgOrderValue"].round(2)
    return rfm


def build_rfm_table(db_path=DB_PATH, snapshot_date=None):
    raw = load_raw_transactions(db_path)
    clean = clean_transactions(raw)
    rfm = compute_rfm(clean, snapshot_date=snapshot_date)
    return rfm, clean


if __name__ == "__main__":
    rfm, clean = build_rfm_table()
    print(rfm.describe())
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "rfm_table.csv")
    rfm.to_csv(out_path, index=False)
    print(f"Saved RFM table to {out_path}")
