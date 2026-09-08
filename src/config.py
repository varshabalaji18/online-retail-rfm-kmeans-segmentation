from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RFMConfig:
    customer_id_col: str = "CustomerID"
    invoice_date_col: str = "InvoiceDate"
    invoice_col: str = "InvoiceNo"
    quantity_col: str = "Quantity"
    price_col: str = "UnitPrice"
    n_clusters: int = 4
    random_state: int = 42
    min_frequency: int = 1
    artifact_dir: Path = Path("artifacts")
