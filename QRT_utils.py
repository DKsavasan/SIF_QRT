import os
import numpy as np
import pandas as pd
from clients.qsec_client.sample_code import *

GROUP_ID = 'ICL05'
USER = 'q8576'
REGION = 'EMEA'
KEY_PATH = os.path.expanduser('~/.ssh/icl04_id_rsa')
TARGETS_PATH = 'target_files/'



def update_portfolio(targets: pd.DataFrame):
    """Update the actual team portfolio at the current market prices.
    targets = pd.DataFrame({'internal_code': 'AAPL.OQ', 'currency': 'USD', 'target_notional': 100})
    """
    target_path = prepare_targets_file(targets, GROUP_ID, REGION, TARGETS_PATH)
    print(pd.read_csv(target_path))
    if not validate_targets_file(target_path):
        upload_targets_file(
            targets_csv_path=target_path, 
            region=REGION, 
            sftp_username=USER, 
            private_key_path=KEY_PATH,
            sftp_host='sftp.qrt.cloud'
        )

def beta(ric: str) -> float:
    """The QRT calculation for the market beta of a stock"""
    lseg_rics = pd.read_csv("local_data/data/lseg_russell_stoxx_2026_02_17.csv")
    if ric in ['.SPX', '.STOXX50E']:
        return 1.0
    
    market = '.SPX' if (lseg_rics[lseg_rics['RIC']==ric]['MktIndex'] == 'RUSSELL3000').iloc[0] else '.STOXX50E'

    stock_return = pd.read_parquet(f"local_data/data/lseg/RIC={ric}").set_index("Date")['Close'].dropna().pct_change().tail(250).dropna()
    benchmark_return = pd.read_parquet(f"local_data/data/lseg/RIC={market}").set_index("Date")['Close'].dropna().pct_change().tail(250).dropna()

    # beta = cov(stock, mkt) / var(mkt)
    cov = (stock_return).cov(benchmark_return)
    var = (benchmark_return).var()
    beta_value = cov / var
    
    # QRT beta calculation
    return 0.2 + 0.8 * float(beta_value)

def portfolio_beta(positions: pd.Series) -> float:
    weights = positions / positions.abs().sum()
    return sum(weights[ric] * beta(ric) for ric in positions.index)

def forced_hedge(positions: pd.Series) -> float:
    hedge = -portfolio_beta(positions) * positions.abs().sum()
    if abs(hedge) < 0.01:
        return 0.0
    return hedge.round(2)

def risk(positions: pd.Series) -> int:
    """The QRT calculation for the portfolios risk, pd.Series({'AAPL': -2500, 'V': 4000})"""

    returns = []
    # Last 60 tradin days of position returns
    for stock in positions.index:
        df = pd.read_parquet(
            f"local_data/data/lseg/RIC={stock}"
        ).set_index("Date")[['Close']].dropna().pct_change().tail(60).dropna()
        returns.append(df.rename(columns={'Close': stock}))

    returns_matrix = pd.concat(returns, axis=1).dropna()
    returns_matrix = returns_matrix[positions.index]

    daily_pnl = returns_matrix @ positions

    risk = daily_pnl.std(ddof=0) * np.sqrt(252)

    return int(risk.round())
