import logging
from pathlib import Path
from typing import Callable
from datetime import datetime
from joblib import Parallel, delayed

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from qrt.data import get_single_timeseries, get_data
from qrt.qrt_utils import to_usd, risk, beta
from qrt.constants import (
    BACKTEST_RESULTS,
    TRADING_DAYS,
    EXECUTION_COST_BPS,
    FINANCING_COST_ANNUAL,
    StockIndex,
    STOCK_INDICES,
)
import qrt

logger = logging.getLogger(__name__)
_SCRIPT_DIR = Path(qrt.__file__).resolve().parent.parent

# ----- Evaluate backtest -----

def generate_results_file_path(strat_name: str, start: str, end: str, stock_index: StockIndex, reb_freq: int, **strategy_kwargs) -> Path:
    """Construct CSV file name from strategy input parameters"""
    backtest_results_dir = _SCRIPT_DIR / BACKTEST_RESULTS
    backtest_results_dir.mkdir(parents=True, exist_ok=True)
    kw_str = "__".join(f"{k}={repr(v)}" for k, v in sorted(strategy_kwargs.items()) if not isinstance(v, dict))
    suffix = f"__{kw_str}" if kw_str else ""
    fname = f"{strat_name}__index={stock_index.name}__start={start}__end={end}__reb={reb_freq}{suffix}.csv"
    return backtest_results_dir / fname

def best_backtests(strategy: str = None, start: str = None, metric: str = 'Sharpe', top_n: int = 5):
    rows = []

    for fp in (_SCRIPT_DIR / BACKTEST_RESULTS).glob("*.csv"):
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            daily_returns = df.iloc[:, 0]

            parts = fp.stem.split("__")

            # first part = strategy name
            strat_name = parts[0]

            # parse key=value parts
            meta = {k: v for k, v in (p.split("=", 1) for p in parts[1:] if "=" in p)}

            # filtering (do early → faster)
            if strategy and strat_name != strategy:
                continue
            if start and meta.get('start') != start:
                continue

            stock_index = STOCK_INDICES[meta['index']]
            reb_freq = int(meta['reb'])

            stats = summary_statistics(daily_returns, stock_index, reb_freq)

            # add metadata columns
            stats['Strategy'] = strat_name
            stats['Start'] = meta.get('start')
            stats['End'] = meta.get('end')
            stats['File'] = fp.name

            rows.append(stats)

        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    if not rows:
        print("No valid backtests found.")
        return

    stats_df = pd.DataFrame(rows)

    # convert metric for sorting
    stats_df['_metric'] = stats_df[metric].str.rstrip('%').astype(float)

    stats_df = (
        stats_df
        .sort_values('_metric', ascending=False)
        .drop(columns='_metric')
    )

    return stats_df.head(top_n)

def analyze_backtest(filename: str, start: str = None):
    """Load backtest from results folder, compute stats, and plot."""

    fp = _SCRIPT_DIR / BACKTEST_RESULTS / filename
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    daily_returns = df.iloc[:, 0]

    # parse metadata
    parts = filename.split("__")
    strat_name = parts[0]
    meta = {k: v for k, v in (p.split("=", 1) for p in parts[1:] if "=" in p)}

    stock_index = STOCK_INDICES[meta['index']]
    reb_freq = int(meta['reb'])

    # stats
    stats = summary_statistics(daily_returns, stock_index, reb_freq)

    # add EVERYTHING from filename
    stats['Strategy'] = strat_name
    for k, v in meta.items():
        stats[k] = v

    stats['File'] = filename

    # plot
    cum_ret = (1 + daily_returns).cumprod()
    if start:
        cum_ret = cum_ret.loc[start:]
    
    cum_ret = (cum_ret / cum_ret.iloc[0] - 1) * 100

    plt.figure(figsize=(10, 3))
    plt.plot(cum_ret)
    plt.title(f"{strat_name} | {meta.get('start')} → {meta.get('end')}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return %")
    plt.grid(True)
    plt.show()

    return stats

def summary_statistics(daily_returns: pd.Series, stock_index: StockIndex, rebalance_freq: int):
    cum_ret = (1 + daily_returns).cumprod() - 1
    ann_ret = daily_returns.mean() * TRADING_DAYS
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (cum_ret - cum_ret.cummax()).min()

    return pd.Series({
        'Index': stock_index.name,
        'Period': f"{daily_returns.index[0].date()} → {daily_returns.index[-1].date()}",
        'Cumulative Return': f'{cum_ret.iloc[-1]:.2%}',
        'Ann. Return': f'{ann_ret:.2%}',
        'Ann. Vol': f'{ann_vol:.2%}',
        'Sharpe': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_dd:.2%}',
        'Rebalance Freq': rebalance_freq,
    })
    

# ----- Run backtest -----

def _compute_weights(strategy_fn, return_data, volume_eligible, stock_index, reb_date, strategy_kwargs):
    try:
        weights, stats = strategy_fn(
            return_data=return_data,
            volume_eligible=volume_eligible,
            portfolio_start=str(reb_date.date()),
            stock_index=stock_index,
            **strategy_kwargs
        )
        logger.info(str(reb_date.date()))
        if weights.empty:
            logger.info(f"[{reb_date.date()}] Empty weights")
            return None
        return weights
    except (ValueError, RuntimeError) as e:
        logger.info(f"[{reb_date.date()}] Skipping: {e}")
        return None

def backtest(
    strategy_fn: Callable,
    return_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    stock_index: StockIndex,
    start_date: str = None,
    end_date: str = None,
    rebalance_freq: int = 10,
    parallel: bool = False,
    save_results: bool = True,
    plot: bool = True,
    resample: str = 'W',
    **strategy_kwargs
):
    """Run a backtest for any strategy that returns (weights, stats). Params required for strategy_fn:
    price_data, volume_eligible, portfolio_start. Also ensure that price_data and volume_eligible are sliced
    to select only necessary data.

    Parameters
    ----------
    strategy_fn      : Callable(price_data, volume_eligible, portfolio_start, **kwargs) → (weights, stats)
    return_data      : Returns DataFrame with market index as first column.
    volume_eligible  : Boolean DataFrame of dates x RICs where dollar ADV ≥ threshold.
    stock_index      : Stock index: RUA or STOXX.
    start_date       : Backtest start date. Defaults to first date in price_data.
    end_date         : Backtest end date. Defaults to last date in price_data.
    rebalance_freq   : Trading days between rebalances.
    save_results     : Save daily returns to Parquet with strategy params in filename.
    plot             : Show cumulative return chart.
    **strategy_kwargs: Passed through to strategy_fn.

    Returns
    -------
    daily_returns : pd.Series of daily portfolio returns.
    summary       : pd.Series of backtest performance metrics.
    """
    all_dates = return_data.index
    start_idx = 0 if start_date is None else all_dates.searchsorted(pd.Timestamp(start_date))
    end_idx = len(all_dates) if end_date is None else all_dates.searchsorted(pd.Timestamp(end_date))

    rebal_indices = list(range(start_idx, end_idx, rebalance_freq))

    daily_returns = pd.Series(0.0, index=all_dates[start_idx:end_idx], dtype=float)
    prev_weights  = pd.Series(dtype=float)

    backtest_results_file = generate_results_file_path(
        strat_name=strategy_fn.__name__,
        start=str(daily_returns.index[0].date()),
        end=str(daily_returns.index[-1].date()),
        stock_index=stock_index,
        reb_freq=rebalance_freq,
        **strategy_kwargs
    )

    if not backtest_results_file.exists():
        reb_dates = [all_dates[idx] for idx in rebal_indices]
        print(f"Running {len(reb_dates)} reb_dates")

        if parallel:
            all_weights = Parallel(n_jobs=-1)(
                delayed(_compute_weights)(strategy_fn, return_data, volume_eligible, stock_index, d, strategy_kwargs)
                for d in reb_dates
            )
        else:
            all_weights = [
                _compute_weights(strategy_fn, return_data, volume_eligible, stock_index, d, strategy_kwargs)
                for d in reb_dates
            ]
        print(f"Completed {len(all_weights)} all_weights")

        for i, weights in enumerate(all_weights):
            if weights is None:
                continue
            reb_idx = rebal_indices[i]
            next_reb_idx = rebal_indices[i + 1] if i + 1 < len(rebal_indices) else end_idx

            # Turnover
            all_tickers = weights.index.union(prev_weights.index)
            old = prev_weights.reindex(all_tickers, fill_value=0)
            new = weights.reindex(all_tickers, fill_value=0)
            # Change in weights from last rebalance to the next, includes all tickers
            turnover = (new - old).abs().sum()

            # Daily returns for holding period: this rebalance date to the next one
            hold_returns = return_data.iloc[reb_idx + 1:next_reb_idx]

            if hold_returns.empty:
                continue

            # Weighted portfolio return
            missing = weights.index.difference(hold_returns.columns)
            if not missing.empty:
                raise ValueError(f"Weights reference unknown tickers: {list(missing)}")
            
            daily_portfolio_rets = (hold_returns[weights.index] * weights).sum(axis=1)

            # Reduce portfolio returns by short position financing costs and execution costs
            short_weight = weights[weights < 0].abs().sum()
            daily_portfolio_rets = daily_portfolio_rets - short_weight * FINANCING_COST_ANNUAL / TRADING_DAYS
            daily_portfolio_rets.iloc[0] -= turnover * EXECUTION_COST_BPS

            # Update return series with the portfolio returns in this rebalance period
            daily_returns.loc[daily_portfolio_rets.index] = daily_portfolio_rets

            # Drifted weights for the next turnover calculation with prev_weights,
            # since weights change (drift) over the holding period
            hold_rets = return_data.iloc[reb_idx + 1:next_reb_idx][weights.index]
            drift_HPR = (1 + hold_rets).prod()

            drifted = weights * drift_HPR
            gross = drifted.abs().sum()
            prev_weights = drifted / gross * weights.abs().sum() if gross > 0 else drifted
    else:
        daily_returns = pd.read_csv(backtest_results_file, index_col=0, parse_dates=True).squeeze()

    # Summary
    summary = summary_statistics(daily_returns=daily_returns, stock_index=stock_index, rebalance_freq=rebalance_freq)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 3))

        # weekly returns
        plot_returns = (1 + daily_returns).resample(resample).prod() - 1
        vals = plot_returns.values * 100
        x = range(len(vals))
        ax.fill_between(x, vals, 0, where=np.array(vals) >= 0, color='green', alpha=0.5, interpolate=True)
        ax.fill_between(x, vals, 0, where=np.array(vals) < 0, color='red', alpha=0.5, interpolate=True)
        ax.axhline(0, color='black', lw=0.5)
        ax.axhline(np.mean(vals), color='blue', lw=1, ls='--', label=f'Mean: {np.mean(vals):.3f}%')
        ticks = np.linspace(0, len(vals) - 1, 5, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([plot_returns.index[i].strftime('%Y-%m-%d') for i in ticks], rotation=45)

        # cumulative on secondary y-axis
        ax2 = ax.twinx()
        # cum_ret = (1 + daily_returns).cumprod() - 1
        cum_ret = (1 + daily_returns).cumprod()
        cum_ret = cum_ret / cum_ret.iloc[0] - 1

        bench_ret = get_single_timeseries(stock_index.benchmark).pct_change(fill_method=None).iloc[1:].reindex(daily_returns.index, fill_value=0)
        # cum_bench = (1 + bench_ret).cumprod() - 1
        cum_bench = (1 + bench_ret).cumprod()
        cum_bench = cum_bench / cum_bench.iloc[0] - 1

        # map daily index to weekly x-axis scale
        x_cum = np.linspace(0, len(vals) - 1, len(cum_ret))
        ax2.plot(x_cum, cum_ret.values * 100, color='black', lw=1.5, label='Strategy')
        ax2.plot(x_cum, cum_bench.values * 100, color='grey', lw=1, ls='--', label=f'Benchmark ({stock_index.benchmark})')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.legend(loc='upper left')

        ax.set_ylabel(f'Period={resample} Return (%)')
        ax.legend(loc='lower left')
        ax.set_title(f'{strategy_fn.__name__.replace("_", " ")} ({stock_index.name})')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        


    if save_results and not backtest_results_file.exists():
        daily_returns.to_csv(backtest_results_file)
        logger.info(f"Saved: {backtest_results_file}")

    return daily_returns, summary

def scale_portfolio_risk(
    weights_rua: pd.Series | None = None,
    weights_stoxx: pd.Series | None = None,
    target_risk_usd: float = 500_000,
    display_stats: bool = True,
) -> pd.DataFrame:
    """Scale US + EU weights so combined USD risk matches target.

    Parameters
    ----------
    weights_rua     : Unscaled US portfolio weights.
    weights_stoxx   : Unscaled EU portfolio weights (in EUR).
    target_risk_usd : Target combined risk in USD.
    display         : Print summary stats after scaling.

    Returns
    -------
    df : pd.DataFrame of scaled nominal portfolio positions, cols: internal_code, target_notional, currency, region, fx_rate, usd_notional.
    """
    if weights_rua is None and weights_stoxx is None:
        raise ValueError("At least one of weights_rua or weights_stoxx must be provided")
    if weights_stoxx is None:
        weights_stoxx = pd.Series(dtype=float)
    if weights_rua is None:
        weights_rua = pd.Series(dtype=float)

    currencies = get_data(
        weights_rua.index.to_list()+weights_stoxx.index.to_list(), 
        fields=["CF_CURR"]
    ).drop_duplicates().set_index('Instrument')
    
    unique_currs = currencies['CF_CURR'].unique()
    fx_map = {c: to_usd(c) for c in unique_currs if c != 'USD'}
    fx_map['USD'] = 1.0
    fx_rates = currencies['CF_CURR'].map(fx_map)

    # Explicit region mapping keyed by RIC (avoids alphabetical reordering by DataFrame ctor)
    region_map = pd.Series(
        ['AMER'] * len(weights_rua) + ['EMEA'] * len(weights_stoxx),
        index=list(weights_rua.index) + list(weights_stoxx.index),
        name='region',
    )

    target_notional = pd.concat([weights_rua, weights_stoxx])
    # Deduplicate (keep first) in case any RIC somehow appears in both
    target_notional = target_notional[~target_notional.index.duplicated(keep='first')]
    region_map = region_map[~region_map.index.duplicated(keep='first')]

    df = pd.DataFrame({'target_notional': target_notional})
    df['currency'] = currencies['CF_CURR'].reindex(df.index)
    df['region'] = region_map.reindex(df.index)
    df['fx_rate'] = fx_rates.reindex(df.index)
    df['usd_notional'] = df['target_notional'] * df['fx_rate']

    scale = target_risk_usd / risk(df['usd_notional'])
    df['target_notional'] = (df['target_notional'] * scale).round(2)
    df['usd_notional'] = (df['target_notional'] * df['fx_rate']).round(2)

    if display_stats:
        info = pd.Series({
            'Market Value': f"{df['usd_notional'].abs().sum():,.0f}",
            'Combined Risk (USD)': f"{risk(df['usd_notional']):,.0f}",
        })
        if len(weights_rua) > 0:
            info['AMER Risk'] = f"{risk(df.loc[df['region']=='AMER', 'usd_notional']):,.0f}"
        if len(weights_stoxx) > 0:
            info['EMEA Risk'] = f"{risk(df.loc[df['region']=='EMEA', 'usd_notional']):,.0f}"

        display(info)

    df.index.name = 'internal_code'
    df = df.reset_index()

    return df

def combine_weights(
    weights_list: list[pd.Series],
    allocations: list[float],
    stock_index: StockIndex = None,
    target_beta: float = None
) -> pd.Series:
    """Weighted combination of strategy weights, normalized to unit gross.
    
    If `target_beta` is provided, projects the blended weights onto the
    closest weight vector (L2) that hits the target shrunken portfolio beta,
    preserving long/short signs. Requires `stock_index` when used.
    """
    combined = pd.Series(0.0, index=set().union(*[w.index for w in weights_list]))
    for w, alpha in zip(weights_list, allocations):
        combined = combined.add(w * alpha, fill_value=0.0)
    combined = combined[combined.abs() > 1e-8]

    gross = combined.abs().sum()
    if gross > 0:
        combined = combined / gross

    if target_beta is None:
        return combined

    if stock_index is None:
        raise ValueError("stock_index required when target_beta is set.")

    # project to target beta
    betas = pd.Series({r: beta(r, stock_index) for r in combined.index}).dropna()
    combined = combined.loc[betas.index]
    raw_vec = combined.values
    sign = np.sign(raw_vec)
    w_cap = float(np.abs(raw_vec).max()) * 2.0

    w = cp.Variable(len(combined))
    constraints = [
        cp.sum(w) == 0,
        betas.values @ w == target_beta,
        cp.multiply(sign, w) >= 0,
        cp.abs(w) <= w_cap,
        cp.sum(cp.abs(w)) <= 1.0,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - raw_vec)), constraints)
    problem.solve(solver=cp.SCS)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Beta targeting failed: {problem.status}")

    out = pd.Series(w.value, index=combined.index).round(8)
    out = out / out.abs().sum()
    return out[out.abs() > 1e-8]
