"""
create_db.py
------------
Loads the raw CSVs into a local SQLite database (database/customers.db)
and creates the `customers` and `transactions` tables. This gives the
project a real SQL layer for extraction/analysis instead of just reading CSVs.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "customers.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SCHEMA_SQL = """
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    CustomerID TEXT PRIMARY KEY,
    SignupDate TEXT,
    Region TEXT
);

-- Note: TransactionID is NOT enforced as a primary key here on purpose.
-- Real-world raw extracts often contain duplicate rows; the RFM pipeline
-- (src/rfm.py) is responsible for de-duplicating before analysis.
CREATE TABLE transactions (
    TransactionID TEXT,
    CustomerID TEXT,
    TransactionDate TEXT,
    ProductCategory TEXT,
    Quantity INTEGER,
    Amount REAL,
    FOREIGN KEY (CustomerID) REFERENCES customers (CustomerID)
);
"""


def build_database():
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    customers.to_sql("customers", conn, if_exists="append", index=False)
    transactions.to_sql("transactions", conn, if_exists="append", index=False)
    conn.commit()

    # Sanity check query
    check = pd.read_sql(
        """
        SELECT COUNT(DISTINCT CustomerID) AS n_customers, COUNT(*) AS n_transactions
        FROM transactions
        """,
        conn,
    )
    print(check)
    conn.close()
    print(f"Database built at {DB_PATH}")


if __name__ == "__main__":
    build_database()
