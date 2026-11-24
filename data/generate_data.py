"""
generate_data.py
-----------------
Generates a realistic synthetic e-commerce transactions dataset and saves it
as data/transactions.csv. Replace this with your real transactional export
(must contain CustomerID, TransactionDate, Amount at minimum) to use real data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 600
SNAPSHOT_DATE = datetime(2026, 9, 1)
START_DATE = SNAPSHOT_DATE - timedelta(days=730)  # 2 years of history

PRODUCT_CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Beauty",
    "Sports", "Books", "Groceries", "Toys"
]

# Define 4 latent behavioral archetypes so the synthetic data has genuine,
# separable clusters for K-Means to discover (mirrors real-world patterns).
ARCHETYPES = {
    "vip":        {"weight": 0.12, "freq_range": (18, 40), "amt_range": (80, 400), "recency_bias": 0.05},
    "loyal":      {"weight": 0.28, "freq_range": (8, 18),  "amt_range": (40, 150), "recency_bias": 0.20},
    "potential":  {"weight": 0.30, "freq_range": (3, 8),   "amt_range": (20, 90),  "recency_bias": 0.45},
    "at_risk":    {"weight": 0.30, "freq_range": (1, 4),   "amt_range": (10, 60),  "recency_bias": 0.85},
}


def assign_archetypes(n):
    names = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[n_]["weight"] for n_ in names]
    return np.random.choice(names, size=n, p=weights)


def generate_customers(n):
    ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n + 1)]
    signup_days_ago = np.random.randint(30, 730, size=n)
    signup_dates = [SNAPSHOT_DATE - timedelta(days=int(d)) for d in signup_days_ago]
    regions = np.random.choice(
        ["North", "South", "East", "West", "Central"], size=n
    )
    archetypes = assign_archetypes(n)
    return pd.DataFrame({
        "CustomerID": ids,
        "SignupDate": signup_dates,
        "Region": regions,
        "_Archetype": archetypes,  # kept only for validation, dropped before saving
    })


def generate_transactions(customers_df):
    rows = []
    txn_id = 1
    for _, cust in customers_df.iterrows():
        arch = ARCHETYPES[cust["_Archetype"]]
        n_txns = np.random.randint(arch["freq_range"][0], arch["freq_range"][1] + 1)

        # recency_bias closer to 1 => last purchase further in the past
        max_recency_days = int(arch["recency_bias"] * 365) + 5
        last_purchase_recency = np.random.randint(1, max_recency_days)

        # Spread earlier purchases across the customer's history
        cust_start = max(cust["SignupDate"], START_DATE)
        available_days = max((SNAPSHOT_DATE - cust_start).days - last_purchase_recency, 1)

        purchase_offsets = sorted(
            np.random.choice(available_days, size=min(n_txns - 1, available_days), replace=False)
        ) if n_txns > 1 else []

        purchase_days_ago = [last_purchase_recency] + [
            last_purchase_recency + (available_days - int(o)) for o in purchase_offsets
        ]

        for days_ago in purchase_days_ago:
            txn_date = SNAPSHOT_DATE - timedelta(days=int(days_ago))
            amount = round(np.random.uniform(*arch["amt_range"]) *
                            np.random.choice([1, 1, 1, 1.5, 2, 0.7]), 2)
            quantity = np.random.randint(1, 6)
            category = np.random.choice(PRODUCT_CATEGORIES)
            rows.append({
                "TransactionID": f"TXN{str(txn_id).zfill(6)}",
                "CustomerID": cust["CustomerID"],
                "TransactionDate": txn_date.strftime("%Y-%m-%d"),
                "ProductCategory": category,
                "Quantity": quantity,
                "Amount": max(amount, 5.0),
            })
            txn_id += 1
    return pd.DataFrame(rows)


def inject_messiness(df):
    """Introduce realistic data-quality issues for the cleaning step to fix."""
    df = df.copy()
    n = len(df)

    # Duplicate a few rows
    dup_idx = np.random.choice(df.index, size=int(0.01 * n), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Null out a few amounts / categories
    null_idx = np.random.choice(df.index, size=int(0.005 * n), replace=False)
    df.loc[null_idx, "Amount"] = np.nan

    # A few negative amounts (returns/refunds) to be cleaned or handled
    neg_idx = np.random.choice(df.index, size=int(0.01 * n), replace=False)
    df.loc[neg_idx, "Amount"] = -df.loc[neg_idx, "Amount"].abs()

    # Inconsistent category casing
    cat_idx = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[cat_idx, "ProductCategory"] = df.loc[cat_idx, "ProductCategory"].str.upper()

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    customers = generate_customers(N_CUSTOMERS)
    transactions = generate_transactions(customers)
    transactions = inject_messiness(transactions)

    customers_out = customers.drop(columns=["_Archetype"])
    customers_out["SignupDate"] = customers_out["SignupDate"].dt.strftime("%Y-%m-%d")

    customers_out.to_csv("data/customers.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)

    print(f"Generated {len(customers_out)} customers and {len(transactions)} transactions.")
    print("Saved to data/customers.csv and data/transactions.csv")


if __name__ == "__main__":
    main()
