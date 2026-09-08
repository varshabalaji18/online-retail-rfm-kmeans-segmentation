"""RFM feature engineering with an explicit as-of snapshot to prevent leakage."""
import pandas as pd
from .config import RFMConfig


def build_rfm(transactions: pd.DataFrame, config: RFMConfig | None = None,
              as_of: str | None = None) -> pd.DataFrame:
    config = config or RFMConfig()
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Transactions missing columns: {sorted(missing)}")
    data = transactions.copy()
    if "SalesAmount" not in data:
        data["SalesAmount"] = data[config.quantity_col] * data[config.price_col]
    if data.empty:
        raise ValueError("No eligible transactions for RFM construction")
    snapshot_date = (pd.Timestamp(as_of) if as_of else
                     data[config.invoice_date_col].max().normalize() + pd.Timedelta(days=1))
    data = data[data[config.invoice_date_col] < snapshot_date].copy()
    if data.empty:
        raise ValueError("No transactions before the exclusive as-of cutoff")
    rfm = data.groupby(config.customer_id_col).agg(
        Recency=(config.invoice_date_col, lambda x: (snapshot_date.normalize() - x.max().normalize()).days),
        Frequency=(config.invoice_col, "nunique"),
        Monetary=("SalesAmount", "sum"),
        LastPurchase=(config.invoice_date_col, "max"),
    ).reset_index()
    rfm = rfm[rfm["Frequency"] >= config.min_frequency].copy()
    rfm.attrs["snapshot_date"] = snapshot_date.isoformat()
    return rfm
