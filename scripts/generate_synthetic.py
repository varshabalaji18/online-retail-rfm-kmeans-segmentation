from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synthetic import make_synthetic_retail

output = Path("data/synthetic_online_retail.csv")
output.parent.mkdir(exist_ok=True)
make_synthetic_retail().to_csv(output, index=False)
print(f"Saved {output}")
