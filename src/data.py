"""Robust ingestion and transaction-level quality controls."""
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {"InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"}


def load_online_retail(path: str | Path) -> pd.DataFrame:
    """Load the UCI Online Retail workbook or a CSV export.

    The UCI workbook is commonly named ``Online Retail.xlsx`` and contains
    one row per invoice line. This loader intentionally keeps raw columns
    available for auditability and normalizes dates/IDs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() in {".csv", ".parquet"}:
        frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    else:
        raise ValueError("Supported dataset formats: .xlsx, .xls, .csv, .parquet")

    frame.columns = [str(c).strip() for c in frame.columns]
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["CustomerID"] = pd.to_numeric(frame["CustomerID"], errors="coerce").astype("Int64")
    frame["InvoiceNo"] = frame["InvoiceNo"].astype("string").str.strip()
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    return frame


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable IDs, returns/cancellations, and non-positive sales."""
    data = frame.copy()
    before = len(data)
    audit = {"raw_rows": before}
    data = data.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"])
    data = data[data["InvoiceNo"].astype("string").str.strip().ne("")]
    audit["missing_required_or_blank_invoice"] = before - len(data)
    previous = len(data)
    data = data[~data["InvoiceNo"].str.upper().str.startswith("C")]
    audit["cancelled_rows"] = previous - len(data)
    previous = len(data)
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data = data[np.isfinite(data["Quantity"]) & np.isfinite(data["UnitPrice"])]
    audit["invalid_or_nonpositive_sales"] = previous - len(data)
    previous = len(data)
    # Remove only exact full-row duplicates when product identity is available.
    # Without StockCode, identical sales may be different legitimate products.
    if "StockCode" in data:
        data = data.drop_duplicates()
    audit["exact_duplicate_rows"] = previous - len(data)
    data["SalesAmount"] = data["Quantity"] * data["UnitPrice"]
    if not np.isfinite(data["SalesAmount"]).all():
        raise ValueError("Non-finite sales amounts")
    audit["retained_rows"] = len(data)
    data.attrs["quality_report"] = audit
    data.attrs["rows_removed"] = before - len(data)
    return data
