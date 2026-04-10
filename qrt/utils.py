import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from qrt.data import get_single_timeseries
from qrt.qrt_utils import eur_usd, risk
from qrt.constants import (
    BACKTEST_RESULTS,
    TRADING_DAYS,
    EXECUTION_COST_BPS,
    FINANCING_COST_ANNUAL,
    StockIndex,
)
import qrt

logger = logging.getLogger(__name__)
_SCRIPT_DIR = Path(qrt.__file__).resolve().parent.parent

def generate_results_file_path(strat_name: str, start: str, end: str, stock_index: StockIndex, reb_freq: int, **strategy_kwargs) -> Path:
    """Construct CSV file name from strategy input parameters"""
    backtest_results_dir = _SCRIPT_DIR / BACKTEST_RESULTS
    backtest_results_dir.mkdir(parents=True, exist_ok=True)
    kw_str = "__".join(f"{k}={repr(v)}" for k, v in sorted(strategy_kwargs.items()))
    suffix = f"__{kw_str}" if kw_str else ""
    fname = f"{strat_name}__index={stock_index.name}__start={start}__end={end}__reb={reb_freq}{suffix}.csv"
    return backtest_results_dir / fname

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
    
def backtest(
    strategy_fn: Callable,
    price_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    stock_index: StockIndex,
    start_date: str = None,
    end_date: str = None,
    rebalance_freq: int = 10,
    save_results: bool = True,
    plot: bool = True,
    **strategy_kwargs
):
    """Run a backtest for any strategy that returns (weights, stats). Params required for strategy_fn:
    price_data, volume_eligible, portfolio_start. Also ensure that price_data and volume_eligible are sliced
    to select only necessary data.

    Parameters
    ----------
    strategy_fn      : Callable(price_data, volume_eligible, portfolio_start, **kwargs) → (weights, stats)
    price_data       : Price DataFrame with market index as first column.
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
    all_dates = price_data.index
    start_idx = 0 if start_date is None else all_dates.searchsorted(pd.Timestamp(start_date))
    end_idx = len(all_dates) if end_date is None else all_dates.searchsorted(pd.Timestamp(end_date))

    rebal_indices = list(range(start_idx, end_idx, rebalance_freq))

    daily_returns = pd.Series(index=all_dates[start_idx:end_idx], dtype=float)
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
        for i, reb_idx in enumerate(rebal_indices):
            reb_date = all_dates[reb_idx]
            next_reb_idx = rebal_indices[i + 1] if i + 1 < len(rebal_indices) else end_idx

            try:
                weights, stats = strategy_fn(
                    price_data=price_data,
                    volume_eligible=volume_eligible,
                    portfolio_start=str(reb_date.date()),
                    stock_index=stock_index,
                    **strategy_kwargs
                )
            except (ValueError, RuntimeError) as e:
                logger.info(f"[{reb_date.date()}] Skipping: {e}")
                continue

            if weights.empty:
                logger.info("Weights are empty")
                continue

            # Turnover
            all_tickers = weights.index.union(prev_weights.index)
            old = prev_weights.reindex(all_tickers, fill_value=0)
            new = weights.reindex(all_tickers, fill_value=0)
            # Change in weights from last rebalance to the next, includes all tickers
            turnover = (new - old).abs().sum()

            # Daily returns for holding period: this rebalance date to the next one
            hold_returns = price_data.iloc[reb_idx:next_reb_idx].pct_change(fill_method=None).iloc[1:]
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
            hold_period_prices = price_data.iloc[reb_idx:next_reb_idx][weights.index]
            drift_HPR = hold_period_prices.iloc[-1] / hold_period_prices.iloc[0]
            drifted = weights * drift_HPR
            gross = drifted.abs().sum()
            prev_weights = drifted / gross * weights.abs().sum() if gross > 0 else drifted
    else:
        daily_returns = pd.read_csv(backtest_results_file, index_col=0, parse_dates=True).squeeze()

    # Summary
    summary = summary_statistics(daily_returns=daily_returns, stock_index=stock_index, rebalance_freq=rebalance_freq)

    if plot:
        bench_ret = get_single_timeseries(stock_index.benchmark).pct_change(fill_method=None).iloc[1:].reindex(daily_returns.index, fill_value=0)
        cum_ret = (1 + daily_returns).cumprod() - 1
        cum_bench = (1 + bench_ret).cumprod() - 1
        (cum_ret * 100).plot(title='Cumulative Return (%)', figsize=(10, 3), label=f'Strategy ({stock_index.name})')
        (cum_bench * 100).plot(label=stock_index.benchmark)
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if save_results and not backtest_results_file.exists():
        daily_returns.to_csv(backtest_results_file)
        logger.info(f"Saved: {backtest_results_file}")

    return daily_returns, summary

def scale_portfolio_risk(
    weights_rua: pd.Series,
    weights_stoxx: pd.Series,
    target_risk_usd: float = 500_000,
    display_stats: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Scale US + EU weights so combined USD risk matches target.

    Parameters
    ----------
    weights_rua     : Unscaled US portfolio weights.
    weights_stoxx   : Unscaled EU portfolio weights (in EUR).
    target_risk_usd : Target combined risk in USD.
    display         : Print summary stats after scaling.

    Returns
    -------
    (scaled_rua, scaled_stoxx) : Two pd.Series of scaled nominal portfolio positions.
    """
    eurusd = eur_usd()

    combined = pd.concat([weights_rua, weights_stoxx * eurusd])
    scale = target_risk_usd / risk(combined)

    scaled_rua = (weights_rua * scale).round(2)
    scaled_stoxx = (weights_stoxx * scale).round(2)

    if display_stats:
        combined_scaled = pd.concat([scaled_rua, scaled_stoxx * eurusd])
        info = pd.Series({
            'Market Value': f"{combined_scaled.abs().sum():,.0f}",
            'Combined Risk (USD)': f"{risk(combined_scaled):,.0f}",
            'SPX Risk': f"{risk(scaled_rua):,.0f}",
            'STOXX Risk': f"{risk(scaled_stoxx):,.0f}",
        })
        display(info)

    return scaled_rua, scaled_stoxx
