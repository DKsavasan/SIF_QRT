import os
import logging
import numpy as np
import pandas as pd
from glob import glob
from typing import Literal
import matplotlib.pyplot as plt
from local_data.LSEG_data import *
from clients.qsec_client.sample_code import *

logger = logging.getLogger(__name__)

GROUP_ID = 'ICL05'
USER = 'q8576'
KEY_PATH = os.path.expanduser('~/.ssh/icl05_id_rsa')
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, 'local_data', 'data')
PRICE_DIR = os.path.join(DATA_DIR, 'lseg')
TARGETS_DIR = os.path.join(_SCRIPT_DIR, 'target_files')

logging.basicConfig(filename=os.path.join(_SCRIPT_DIR, 'logs/send_portfolio.log'), level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def send_new_portfolio(targets: pd.DataFrame, region: Literal['AMER', 'EMEA']):
    """Update the actual team portfolio at the current market prices.
    targets = pd.DataFrame({'internal_code': 'AAPL.OQ', 'currency': 'USD', 'target_notional': 100})
    """
    try:
        target_path = prepare_targets_file(targets, GROUP_ID, region, TARGETS_DIR)
        logger.info(pd.read_csv(target_path))
        formating_issues = validate_targets_file(target_path)
        if formating_issues == []:
            upload_targets_file(
                targets_csv_path=target_path, 
                region=region, 
                sftp_username=USER, 
                private_key_path=KEY_PATH,
                sftp_host='sftp.qrt.cloud'
            )
            logging.info(f"Portfolio successfully uploaded: {target_path}")
        else:
            logging.error(f"Validation failed, portfolio not uploaded: {target_path}. Issues: {formating_issues}")
            raise
    except Exception as e:
        logging.error(f"Failed to send portfolio: {e}")
        raise

def beta(ric: str, market: Literal['.RUA', '.STOXX50E']) -> float | None:
    """The QRT calculation for the market beta of a stock"""
    if ric in ['.RUA', '.STOXX50E']:
        return 1.0

    stock_return = pd.read_parquet(os.path.join(PRICE_DIR, f"RIC={ric}")).set_index("Date")['Close'].dropna().pct_change().tail(250).dropna()
    benchmark_return = pd.read_parquet(os.path.join(PRICE_DIR, f"RIC={market}")).set_index("Date")['Close'].dropna().pct_change().tail(250).dropna()

    if stock_return.index[-1] < benchmark_return.index[-3]:
        # print(f"Skipping {isin}: last trade {stock_return.index[-1]} is before {benchmark_return.index[-2]}")
        return None

    # beta = cov(stock, mkt) / var(mkt)
    cov = (stock_return).cov(benchmark_return)
    var = (benchmark_return).var()
    beta_value = cov / var

    # QRT beta calculation
    return 0.2 + 0.8 * float(beta_value)

def portfolio_beta(positions: pd.Series, market: Literal['.RUA', '.STOXX50E']) -> float:
    weights = positions / positions.abs().sum()
    total = 0.0
    for ric in positions.index:
        b = beta(ric, market)
        if b is None:
            print(f"Skipping {ric}: no beta")
            continue
        total += weights[ric] * b
    return total

def forced_hedge(positions: pd.Series, market: Literal['.RUA', '.STOXX50E']) -> float:
    """Nominal currency to hedge against beta exposure"""
    hedge = -portfolio_beta(positions, market) * positions.abs().sum()
    if abs(hedge) < 0.01:
        return 0.0
    return hedge.round(2)

def risk(positions: pd.Series, date: str = None) -> int:
    """Annualised volatility of daily PnL in currency units using previous 60 trading days of returns
    QRT calculation for the portfolio risk
    Parameters:
        positions: RIC-to-value (currency) Series e.g. pd.Series({'AAPL': -2500, 'V': 4000}).
        date: Close date to measure risk exposure for.
    Returns:
        int: Nominal risk exposure."""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")

    date = pd.Timestamp(date)

    returns = []
    # Last 60 tradin days of position returns
    for ric in positions.index:
        df = pd.read_parquet(
            os.path.join(PRICE_DIR, f"RIC={ric}")
        ).set_index("Date")[['Close']]

        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        df = df[df.index <= date]
        df = df.dropna()

        # strictly last 60 calendar days from date
        cutoff = date - pd.Timedelta(days=60)
        prices = df[df.index >= cutoff]
        rets = prices.pct_change().dropna()

        returns.append(rets.rename(columns={'Close': ric}))

    returns_matrix = pd.concat(returns, axis=1).dropna()
    returns_matrix = returns_matrix[positions.index]

    daily_pnl = returns_matrix @ positions

    risk = daily_pnl.std(ddof=1) * np.sqrt(252)

    return int(risk.round())

def load_returns_from(rics: pd.Index | list, start: str = '2026-01-01') -> pd.DataFrame:
    """Get daily returns DataFrame from local data, one column per RIC
    Parameters:
        rics: List of RIC's to fetch price data for.
        start: String date to get returns from.
    Returns:
        pd.DataFrame: Returns with date as index and RIC in columns.
    """
    returns_list = []
    for ric in rics:
        df = pd.read_parquet(os.path.join(PRICE_DIR, f"RIC={ric}")).set_index("Date")[['Close']].dropna()
        df = df[~df.index.duplicated(keep='first')]
        df.index = pd.to_datetime(df.index)
        df = df[df.index >= start]
        ret = df['Close'].pct_change().fillna(0)
        returns_list.append(ret.rename(ric))

    returns_df = pd.concat(returns_list, axis=1)

    cols_with_nulls = returns_df.columns[returns_df.isnull().any()]

    if len(cols_with_nulls) > 0:
        print(f"Skipped stocks with null values: {list(cols_with_nulls)}")

    returns_df_clean = returns_df.drop(columns=cols_with_nulls)

    return returns_df_clean

def plot_portfolio_returns(positions: pd.Series, market: Literal['.RUA', '.STOXX50E'], start_date: str = '2026-01-01', benchmark: pd.Series = None, figsize=(10, 5)):
    """Plot cumulative portfolio returns (%) since start_date.

    Parameters:
        positions: Portfolio nominal amounts, e.g. pd.Series({'AAPL.OQ': 500_000, 'V.N': -400_000})
        start_date: Date to start calculating returns from assuming bought at close,
                    first plotted point has zero cumulative return.
        benchmark: ISIN to notional series for the benchmark portfolio.
    """
    if benchmark is None:
        benchmark = pd.Series({market: 1})

    def cum_returns(returns_df: pd.DataFrame, pos: pd.Series) -> pd.Series:
        returns_df = returns_df[pos.index]
        total_notional = pos.abs().sum()
        weights = pos / total_notional
        daily_ret = (returns_df * weights).sum(axis=1)
        return (1 + daily_ret).cumprod() - 1

    # Portfolio
    port_returns_df = load_returns_from(positions.index, start_date)
    port_cum = cum_returns(port_returns_df, positions[port_returns_df.columns])

    # Benchmark
    bench_returns_df = load_returns_from(benchmark.index, start_date)
    bench_cum = cum_returns(bench_returns_df, benchmark)

    # Plot
    if True:
        plt.figure(figsize=figsize)
        plt.plot(port_cum.index, port_cum.values * 100, label='Portfolio')
        plt.plot(bench_cum.index, bench_cum.values * 100, label=f'Benchmark ({", ".join(benchmark.index)})', linestyle='--')
        plt.title(f'Portfolio vs Benchmark Cumulative Return since {start_date}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def most_recent_positions(folder_path: str = TARGETS_DIR, pattern: str = "*.csv") -> pd.Series:
    """
    Reads the most recent file positions in `folder_path` matching the `pattern` and returns as a DataFrame.
    
    Parameters:
        folder_path (str): Path to the folder containing files.
        pattern (str): Glob pattern to match files, default '*.csv'.
    
    Returns:
        pd.Series: Series of the most recent file positions.
    """
    # Build full search pattern
    search_pattern = os.path.join(folder_path, pattern)
    
    # Get all matching files
    files = glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No files found in {folder_path} matching {pattern}")
    
    # Get the most recently modified file
    most_recent_file = max(files, key=os.path.getmtime)
    
    # Read into DataFrame
    df = pd.read_csv(most_recent_file)

    df = pd.Series(data=df["target_notional"].values, index=df["ric"])

    return df
