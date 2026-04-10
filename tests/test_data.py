from datetime import datetime
import pandas as pd

from qrt.data import get_data, get_history, get_fundamental_data, get_single_timeseries, get_timeseries, eligible_to_trade, PRICE_DIR
from qrt.constants import RUA, LSEG_ACTIVE, FUNDAMENTAL_METRICS_QUARTERLY

def test_get_data():
    output_cols = ['Instrument', 'Primary Issue RIC', 'ISIN', 'Company Common Name']
    result = get_data(['AAPL.OQ'])
    assert all(c in result.columns for c in output_cols)

def test_get_history():
    output_cols = ['Price Close', 'Volume']
    result = get_history(['AAPL.OQ'])
    assert all(c in result.columns for c in output_cols)

def test_get_fundamental_data():
    assert len(get_fundamental_data('AAPL.OQ').columns)==len(FUNDAMENTAL_METRICS_QUARTERLY)

def test_get_single_timeseries():
    TWO_DAYS_AGO = (datetime.now() - pd.DateOffset(days=2)).date()
    assert get_single_timeseries(RUA.benchmark).index[-1].date() >= TWO_DAYS_AGO

def test_get_timeseries():
    spx = RUA.benchmark
    df = pd.read_parquet(PRICE_DIR / LSEG_ACTIVE / f"RIC={spx}/part.parquet").assign(RIC=spx)
    assert get_timeseries(df).columns == spx

def test_eligible_to_trade():
    spx = RUA.benchmark
    df = pd.read_parquet(PRICE_DIR / LSEG_ACTIVE / f"RIC={spx}/part.parquet").assign(RIC=spx)
    price  = get_timeseries(df, value='Close')
    volume = get_timeseries(df, value='Volume')
    assert eligible_to_trade(price, volume).columns == spx
