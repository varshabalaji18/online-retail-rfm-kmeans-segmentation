"""CLI: python scripts/train_rfm.py --input data/Online Retail.xlsx"""
import argparse
from pathlib import Path
import sys
import json
import hashlib
import platform
from importlib.metadata import version
import joblib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clean_transactions, load_online_retail
from src.rfm import build_rfm
from src.model import save_artifacts, train_model
from src.model import RFM_COLUMNS
from src.validation import compare_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer RFM/K-Means segmentation")
    parser.add_argument("--input", required=True, help="Path to Online Retail.xlsx/csv/parquet")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--clusters", type=int, default=None, help="Override automatic best-silhouette k")
    parser.add_argument("--as-of", help="Exclusive historical cutoff, e.g. 2011-12-01")
    args = parser.parse_args()

    raw = load_online_retail(args.input)
    clean = clean_transactions(raw)
    rfm = build_rfm(clean, as_of=args.as_of)
    validation = compare_clusters(rfm)
    selected_k = args.clusters or int(validation.sort_values('silhouette', ascending=False).iloc[0]['k'])
    pipeline, scored, profiles = train_model(rfm, n_clusters=selected_k)
    save_artifacts(pipeline, scored, profiles, args.output_dir, rfm.attrs.get("snapshot_date"))
    output = Path(args.output_dir)
    validation.to_csv(output / 'model_comparison.csv', index=False)
    rfm.to_csv(output / 'rfm_features.csv', index=False)
    report = dict(clean.attrs['quality_report'])
    report['rows_at_or_after_cutoff'] = int((clean.InvoiceDate >= rfm.attrs['snapshot_date']).sum())
    (output / 'data_quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    metadata = json.loads((output / 'metadata.json').read_text())
    metadata.update({'input_file': Path(args.input).name,
        'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'selection_rule': 'explicit override' if args.clusters else 'maximum sampled silhouette over k=2..6',
        'currency': 'GBP for UCI; synthetic units for generated data',
        'monetary_definition': 'gross positive purchase value, excluding returns; not profit or net revenue',
        'python_version': platform.python_version(),
        'package_versions': {p: version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'joblib']}})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    restored = joblib.load(output / 'rfm_kmeans_pipeline.joblib')
    np.testing.assert_array_equal(restored.predict(rfm[RFM_COLUMNS]), scored.Cluster)
    print('Artifact reload verification passed: all customer predictions match')
    print(validation.to_string(index=False))
    print(f"Loaded {len(raw):,} rows; retained {len(clean):,} transaction lines")
    print(f"Built {len(rfm):,} customer records; snapshot={rfm.attrs.get('snapshot_date')}")
    print(f"Saved artifacts to {Path(args.output_dir).resolve()}")
    print(profiles)


if __name__ == "__main__":
    main()
