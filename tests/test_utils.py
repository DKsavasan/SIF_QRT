import pandas as pd
from qrt.constants import RUA
from qrt.utils import generate_results_file_path, summary_statistics


def test_generate_results_file_path():
    path = generate_results_file_path('my_strat', '2020-01-01', '2025-01-01', RUA, 10, lookback=60)
    assert path.suffix == '.csv'
    assert 'my_strat' in path.name
    assert 'RUA' in path.name
    assert 'lookback=60' in path.name

def test_summary_statistics():
    rets = pd.Series(0.01, index=pd.date_range('2024-01-01', periods=100))
    summary = summary_statistics(rets, RUA, 10)
    assert 'Sharpe' in summary.index
    assert summary['Rebalance Freq'] == 10
