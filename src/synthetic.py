"""Generate a realistic local fallback when the public workbook is inconvenient."""
import numpy as np
import pandas as pd


def make_synthetic_retail(n_customers: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_ids = np.arange(10000, 10000 + n_customers)
    rows = []
    start = pd.Timestamp("2011-01-01")
    for customer_id in customer_ids:
        invoices = int(rng.poisson(5) + 1)
        segment = rng.choice(["loyal", "at_risk", "low_spend"], p=[0.35, 0.25, 0.40])
        for invoice_idx in range(invoices):
            if segment == "at_risk":
                day = int(rng.integers(0, 80))
            elif segment == "loyal":
                day = int(rng.integers(180, 365))
            else:
                day = int(rng.integers(250, 365))
            invoice_no = f"{customer_id}-{invoice_idx:03d}"
            for _ in range(int(rng.integers(1, 5))):
                rows.append({
                    "InvoiceNo": invoice_no,
                    "InvoiceDate": start + pd.Timedelta(days=day),
                    "Quantity": int(rng.integers(1, 8)),
                    "UnitPrice": round(float(rng.lognormal(2.0 if segment == "loyal" else 1.3, 0.55)), 2),
                    "CustomerID": customer_id,
                })
    return pd.DataFrame(rows)
