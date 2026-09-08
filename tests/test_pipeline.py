import numpy as np
import pandas as pd
import pytest
import joblib
from src.data import clean_transactions, load_online_retail
from src.rfm import build_rfm
from src.model import train_model, save_artifacts, RFM_COLUMNS
from src.synthetic import make_synthetic_retail


def test_rfm_cutoff_invoice_count_and_gross_spend():
    raw = pd.DataFrame({'CustomerID': [1]*5, 'InvoiceNo': ['A','A','B','C1','F'],
        'InvoiceDate': pd.to_datetime(['2011-01-01','2011-01-01','2011-01-03','2011-01-04','2011-02-01']),
        'Quantity': [2,1,3,-1,100], 'UnitPrice': [5,7,2,5,100]})
    rfm = build_rfm(clean_transactions(raw), as_of='2011-02-01')
    assert rfm.iloc[0][RFM_COLUMNS].tolist() == [29, 2, 23]
    with pytest.raises(ValueError, match='before'):
        build_rfm(clean_transactions(raw), as_of='2010-01-01')


def test_quality_and_schema(tmp_path):
    raw = make_synthetic_retail(8)
    raw['StockCode'] = range(len(raw))
    duplicate = raw.iloc[[0]]
    bad = raw.iloc[[1]].assign(CustomerID=pd.NA)
    clean = clean_transactions(pd.concat([raw, duplicate, bad], ignore_index=True))
    assert len(clean) == len(raw)
    assert clean.attrs['quality_report']['exact_duplicate_rows'] == 1
    file = tmp_path / 'bad.csv'
    pd.DataFrame({'x':[1]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match='Missing required'):
        load_online_retail(file)


def test_artifact_round_trip_and_invalid_features(tmp_path):
    rfm = build_rfm(clean_transactions(make_synthetic_retail(50)))
    model, scored, profiles = train_model(rfm, 3)
    save_artifacts(model, scored, profiles, tmp_path, rfm.attrs['snapshot_date'])
    restored = joblib.load(tmp_path / 'rfm_kmeans_pipeline.joblib')
    np.testing.assert_array_equal(restored.predict(rfm[RFM_COLUMNS]), scored.Cluster)
    with pytest.raises(ValueError, match='finite'):
        train_model(rfm.assign(Monetary=np.nan), 3)
    with pytest.raises(ValueError, match='distinct'):
        train_model(rfm, 100)
