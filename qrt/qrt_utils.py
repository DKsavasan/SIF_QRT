import logging
from glob import glob
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import qrt
from qrt.constants import StockIndex, DataType, GROUP_ID, USER, DATA_CONFIG
from qsec_client.sample_code import prepare_targets_file, validate_targets_file, upload_targets_file

logger = logging.getLogger(__name__)


KEY_PATH = Path.home() / ".ssh" / "icl05_id_rsa"
_SCRIPT_DIR = Path(qrt.__file__).resolve().parent.parent
DATA_DIR = _SCRIPT_DIR / "data"
PRICE_DIR = DATA_DIR / "price_volume"
TARGETS_DIR = _SCRIPT_DIR / "target_files"
LOG_FILE = _SCRIPT_DIR / "logs" / "send_portfolio.log"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def send_new_portfolio(positions: pd.Series, stock_index: StockIndex, validate_only: bool = True):
    """Validate and upload portfolio targets to QRT SFTP for execution at current market prices.

    Parameters:
        positions: RIC-to-value (notational) Series
        stock_index: Stock index: RUA or STOXX.
        validate_only: If True, validate without uploading.
    """
    targets = (
        positions
        .rename_axis('internal_code')
        .reset_index(name='target_notional')
        .assign(currency=stock_index.currency)
        .sort_values('target_notional', ascending=False, ignore_index=True)
        .assign(internal_code=lambda df: df['internal_code'].astype(str))
    )

    target_path = prepare_targets_file(targets, GROUP_ID, stock_index.region, TARGETS_DIR / stock_index.region)
    logger.info(pd.read_csv(target_path))
    issues = validate_targets_file(target_path)

    if issues:
        raise ValueError(f"Validation failed: {issues}, target path: {target_path}")
    
    if validate_only:
        return

    try:
        upload_targets_file(
            targets_csv_path=target_path, 
            region=stock_index.region, 
            sftp_username=USER, 
            private_key_path=KEY_PATH,
            sftp_host='sftp.qrt.cloud'
        )
        logger.info(f"Portfolio successfully uploaded: {target_path}")
    except Exception as e:
        logger.error(f"SFTP upload failed: {e}")
        raise

def beta(inst: str, stock_index: StockIndex, data_type: DataType = 'active') -> float:
    """Market beta of a single instrument using trailing 250-day returns.

    Computes cov(stock, market) / var(market), then applies the QRT shrinkage formula: 0.2 + 0.8 * β.

    Parameters:
        inst: Instrument RIC or ISIN.
        stock_index: Stock index: RUA or STOXX.
        data_type: Active or historical data.

    Returns:
        Shrunken beta, 1.0 if inst == market, or None if insufficient data.
    """
    if inst == stock_index.benchmark:
        return 1.0
    
    config = DATA_CONFIG[data_type]

    try:
        stock_return = pd.read_parquet(PRICE_DIR / config['sub_dir'] / f"{config['inst_name']}={inst}").set_index("Date")['Close'].dropna().pct_change().tail(250).dropna()
    except FileNotFoundError:
        logger.info(f"Data not found for {config['inst_name']}={inst}")
        return None

    benchmark_return = pd.read_parquet(PRICE_DIR / config['sub_dir'] / f"RIC={stock_index.benchmark}").set_index("Date")['Close'].dropna().pct_change().tail(250).dropna()

    if len(benchmark_return) < 3 or stock_return.index[-1] < benchmark_return.index[-3]:
        logger.info(f"Skipping {inst}: last trade {stock_return.index[-1]} is before {benchmark_return.index[-2]}")
        return None

    # beta = cov(stock, mkt) / var(mkt)
    cov = (stock_return).cov(benchmark_return)
    var = (benchmark_return).var()
    beta_value = cov / var

    # QRT beta calculation
    return 0.2 + 0.8 * float(beta_value)

def portfolio_beta(positions: pd.Series, stock_index: StockIndex, data_type: DataType = 'active') -> float:
    """Value-weighted portfolio beta against a market benchmark.

    Parameters:
        positions: RIC-to-value (currency) Series.
        stock_index: Stock index: RUA or STOXX.
        data_type: Active or historical data.

    Returns:
        Weighted average of per-instrument shrunk betas. Instruments with no computable beta are excluded.
    """
    gross = positions.abs().sum()
    if gross == 0:
        return 0.0

    total = 0.0

    for inst, pos in positions.items():
        b = beta(inst=inst, stock_index=stock_index, data_type=data_type)
        if b is None:
            continue
        total += (pos / gross) * b

    return float(total)

def forced_hedge(positions: pd.Series, stock_index: StockIndex, data_type: DataType = 'active') -> float:
    """Nominal currency to hedge against beta exposure"""
    hedge = -portfolio_beta(positions, stock_index, data_type) * positions.abs().sum()
    if abs(hedge) < 0.01:
        return 0.0
    return hedge.round(2)

def risk(positions: pd.Series, date: str = None, data_type: DataType = 'active') -> float:
    """Annualised volatility or notational risk of daily PnL in currency units using previous 60 
    trading days of returns QRT calculation for the portfolio risk

    Parameters:
        positions: or weights of RIC-to-value (currency) Series e.g. pd.Series({'AAPL': -2500, 'V': 4000}).
        date: Close date to measure risk exposure for.
        data_type: Active or historical data.
    Returns:
        float: Nominal risk exposure or volatility if weights.
    """
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")

    date = pd.Timestamp(date)
    config = DATA_CONFIG[data_type]

    returns = []
    # Last 60 trading days of position returns
    for ric in positions.index:
        df = pd.read_parquet(
            PRICE_DIR / config['sub_dir'] / f"{config['inst_name']}={ric}"
        ).set_index("Date")[['Close']]
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')].dropna()

        # strictly last 60 days of returns from date
        rets = df.loc[:date].pct_change().dropna().tail(60)
        returns.append(rets.rename(columns={'Close': ric}))

    # Fill na with zero for different holidays
    returns_matrix = pd.concat(returns, axis=1)[positions.index].fillna(0)

    daily_pnl = returns_matrix @ positions

    return float(daily_pnl.std(ddof=1) * np.sqrt(252))

def load_returns_from(insts: pd.Index | list, start: str = '2026-01-01', data_type: DataType = 'active') -> pd.DataFrame:
    """Get daily returns DataFrame from local data, one column per Instrument
    Parameters:
        insts: List of Instrument's to fetch price data for.
        start: String date to get returns from.
        data_type: Active or historical data.
    Returns:
        pd.DataFrame: Returns with date as index and Instruments in columns.
    """
    config = DATA_CONFIG[data_type]
    inst_name = config['inst_name']
    returns_list = []
    for inst in insts:
        try:
            df = pd.read_parquet(PRICE_DIR / config['sub_dir'] / f"{inst_name}={inst}").set_index("Date")[['Close']].dropna()
        except FileNotFoundError:
            logger.info(f"load_returns_from: Data not found for {inst_name}={inst}")
            continue
        df = df[~df.index.duplicated(keep='first')]
        df.index = pd.to_datetime(df.index)
        ret = df['Close'].pct_change()
        ret = ret[ret.index >= start]
        returns_list.append(ret.rename(inst))

    returns_df = pd.concat(returns_list, axis=1)

    cols_with_nulls = returns_df.columns[returns_df.isnull().any()]

    for col in cols_with_nulls:
        nulls = returns_df[col].isnull()
        consecutive = nulls & nulls.shift(-1)  # 2+ nulls in a row
        if consecutive.any():
            logger.info(f"load_returns_from: Dropped {col}: consecutive missing dates")
            returns_df = returns_df.drop(columns=col)
        else:
            returns_df[col] = returns_df[col].fillna(0)
            logger.info(f"load_returns_from: Forward-filled {nulls.sum()} missing prices for {col}")

    return returns_df

def plot_portfolio_returns(positions: pd.Series, stock_index: StockIndex | pd.Series, start_date: str = '2026-01-01', data_type: DataType = 'active', figsize=(10, 5)):
    """Plot cumulative portfolio returns (%) since start_date.

    Parameters:
        positions: Signed weights or notional amounts indexed by Instrument, e.g. pd.Series({'AAPL.OQ': 500_000, 'V.N': -400_000}).
        benchmark: StockIndex identifier or series of notional positions indexed by instrument for the benchmark portfolio.
        start_date: Date to start calculating returns from, assuming bought at close. First plotted point has zero cumulative return.
        data_type: Active or historical data.
        figsize: Matplotlib figure size tuple.
    """
    if isinstance(stock_index, pd.Series):
        bench_positions = stock_index
    else:
        bench_positions = pd.Series({stock_index.benchmark: 1})

    def cum_returns(returns_df: pd.DataFrame, pos: pd.Series) -> pd.Series:
        returns_df = returns_df[pos.index]
        total_notional = pos.abs().sum()
        if total_notional == 0:
            return pd.Series(0.0, index=returns_df.index)
        weights = pos / total_notional
        daily_ret = (returns_df * weights).sum(axis=1)
        return (1 + daily_ret).cumprod() - 1
    
    # Portfolio
    port_returns_df = load_returns_from(insts=positions.index, start=start_date, data_type=data_type)
    port_cum = cum_returns(port_returns_df, positions[port_returns_df.columns])

    # Benchmark
    bench_returns_df = load_returns_from(insts=bench_positions.index, start=start_date, data_type=data_type)
    bench_cum = cum_returns(bench_returns_df, bench_positions)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(port_cum.index, port_cum.values * 100, label='Portfolio')
    plt.plot(bench_cum.index, bench_cum.values * 100, label=f'Benchmark ({", ".join(bench_positions.index)})', linestyle='--')
    plt.title(f'Portfolio vs Benchmark Cumulative Return since {start_date}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def most_recent_positions(stock_index: StockIndex, pattern: str = "*.csv") -> pd.Series:
    """
    Reads the most recent file positions for the region matching the file pattern and returns as a DataFrame.
    
    Parameters:
        stock_index: Stock index: RUA or STOXX.
        pattern (str): Glob pattern to match files, default '*.csv'.
    
    Returns:
        pd.Series: Series of the most recent file positions.
    """
    # Build full search pattern
    search_pattern = str(TARGETS_DIR / stock_index.region / pattern)
    
    # Get all matching files
    files = glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No files found in {TARGETS_DIR}/{stock_index.region} matching {pattern}")
    
    # Get the most recently added file
    most_recent_file = max(files, key=lambda f: Path(f).stat().st_mtime)
    
    # Read into DataFrame
    df = pd.read_csv(most_recent_file)

    df = pd.Series(data=df["target_notional"].values, index=df["ric"])

    return df

@lru_cache(maxsize=128)
def eur_usd(date: str | None = None) -> float:
    """
    Get the current EURUSD exchange rate from Yahoo Finance.
    
    Parameters:
        date (str): String date for the fx rate
    
    Returns:
        float: Exchange rate value
    """
    today = str(datetime.now().date()) if date is None else date
    fx_rate = yf.download(
        "EURUSD=X", end=today, auto_adjust=True, progress=False
    )['Close']['EURUSD=X'].iloc[-1]
    return float(fx_rate)

if __name__ == '__main__':
    from qrt.constants import RUA

    pos = most_recent_positions(RUA)
    print(pos.head())
    print(len(pos.index))
    print(portfolio_beta(pos, stock_index=RUA))
