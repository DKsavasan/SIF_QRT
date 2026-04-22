from datetime import datetime

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt

from qrt.constants import StockIndex, TRADING_DAYS
from qrt.qrt_utils import portfolio_beta, beta
from qrt.strat_helpers import optimal_weights, screen_event_impact, build_quality_score

# ----- Momentum Strategy -----
def momentum(
    return_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    stock_index: StockIndex,
    portfolio_start: str,
    lookback: int = 252,
    skip_last: int = 21,
    quantile_top: float = 0.9,
    quantile_bottom: float = 0.1,
    vol_quantile: float = 0.90,
    screen_days: int = 20,
    screen_z_threshold: float = 1.0,
    apply_screener: bool = False,
    display_screener: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """Long winners, short losers. Inverse-vol weighted, no optimizer."""

    all_dates = return_data.index
    start_idx = all_dates.searchsorted(pd.Timestamp(portfolio_start))
    entry_date = all_dates[min(start_idx, len(all_dates) - 1)]

    if start_idx < lookback:
        raise ValueError(f"Need {lookback} days before {entry_date}, have {start_idx}.")

    train_returns = return_data.iloc[start_idx - lookback : start_idx].dropna(axis=1, how='any')

    # 12-1 momentum signal
    cum_full = (1 + train_returns).prod() - 1
    cum_skip = (1 + train_returns.iloc[-skip_last:]).prod() - 1
    momentum = cum_full - cum_skip

    # filters
    eligible = volume_eligible.columns[volume_eligible.loc[entry_date]]
    momentum = momentum[momentum.index.intersection(eligible)].dropna()

    ann_vol = train_returns[momentum.index].std() * np.sqrt(252)
    low_vol = ann_vol[ann_vol < ann_vol.quantile(vol_quantile)].index
    momentum = momentum[momentum.index.intersection(low_vol)]

    # long top, short bottom
    longs = momentum[momentum >= momentum.quantile(quantile_top)].index
    shorts = momentum[momentum <= momentum.quantile(quantile_bottom)].index

    if len(longs) == 0 or len(shorts) == 0:
        raise ValueError("Empty long or short bucket.")

    weights = pd.Series(0.0, index=longs.union(shorts))

    inv_vol = 1.0 / ann_vol[longs]
    weights[longs] = inv_vol / inv_vol.sum()

    inv_vol = 1.0 / ann_vol[shorts]
    weights[shorts] = -inv_vol / inv_vol.sum()

    if apply_screener:
        flagged, weights = screen_event_impact(
            weights=weights, 
            return_data=train_returns, 
            stock_index=stock_index, 
            reb_date=portfolio_start, 
            days=screen_days, 
            z_threshold=screen_z_threshold, 
            plot=display_screener
        )
        if display_screener:
            print(flagged)

    stats = {
        'portfolio_start': str(portfolio_start),
        'positions': len(weights),
        'longs': len(longs),
        'shorts': len(shorts),
        'gross_leverage': f'{weights.abs().sum():.2f}x',
    }

    return weights, stats

# ----- Titman Momentum Strategy -----
def titman_min_var_momentum(
    return_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    stock_index: StockIndex,
    portfolio_start: str,
    sectors: pd.DataFrame,
    fundamentals: dict[str, pd.DataFrame],
    apply_screener: bool = False,
    display_screener: bool = False,
    lookback: int = TRADING_DAYS,
    skip_last: int = 21,
    vol_quantile: float = 0.90,
    mom_quantile_top: float = 0.85,
    mom_quantile_bottom: float = 0.15,
    target_vol: float = None,
    use_ewma_vol: bool = False,
    sector_neutral_tol: float = 0.05,
    long_weight_max: float = 0.07,
    long_weight_min: float = 0.005,
    short_weight_max: float = 0.04,
    short_weight_min: float = 0.005,
    use_residual_momentum: bool = True,
    sector_neutral: bool = True,
    quality_veto_quantile: float = 0.70,
    screen_days: int = 20,
    screen_z_threshold_long: float = 1.0,
    screen_z_threshold_short: float = 1.5,
    target_beta: float = 0.0,
) -> tuple[pd.Series, dict[str, Any]]:
    """Beta-neutral long/short momentum portfolio optimised for minimum variance.
    Filters universe by volatility, momentum percentile, and dollar volume eligibility,
    then solves for weights that maximise expected return subject to vol, beta, and leverage constraints.

    Parameters
    ----------
    return_data         : Returns DataFrame with market index as first column, stocks as remaining columns.
    volume_eligible     : Boolean DataFrame of dates x RICs where dollar ADV ≥ threshold.
    portfolio_start     : Entry date. Training window is [start - lookback, start).
    stock_index         : Stock index: RUA or STOXX.
    screen_stock_picks  : If True, apply short-term reversal screen to output weights.
    lookback            : Training window length in trading days.
    skip_last           : Days to skip at end of lookback for momentum calculation.
    vol_quantile        : Keep stocks below this volatility percentile.
    mom_quantile_top    : Long stocks above this momentum percentile.
    mom_quantile_bottom : Short stocks below this momentum percentile.
    target_vol          : Max annualised portfolio volatility constraint.
    weight_max          : Max absolute weight per stock.
    weight_min          : Min absolute weight to keep a position.
    screen_days         : Look-back window for reversal screen.
    screen_z_threshold  : Z-score threshold for reversal screen.

    Returns
    -------
    weights : pd.Series of portfolio weights indexed by RIC.
    stats   : Dict of in-sample performance metrics.
    """

    # --- Locate entry date in index ---
    all_dates = return_data.index
    start_idx = all_dates.searchsorted(pd.Timestamp(portfolio_start))
    
    if start_idx >= len(all_dates): # portfolio_start after most recent data
        entry_date = pd.Timestamp(portfolio_start)
    else:
        entry_date = all_dates[start_idx] # actual trading day from data
    
    # Recent data required to create the portfolio at the current moment of time 'entry_date' and removing cols with any nulls
    train_returns = return_data.iloc[start_idx - lookback : start_idx].dropna(axis=1, how='any')

    # --- Build quality score from most recent available fundamentals ---
    quality_score = build_quality_score(fundamentals=fundamentals, entry_date=entry_date)

    # --- Sector map aligned to universe ---
    sector_map = sectors.set_index('Instrument')['GICS Sector Name']


    # --- Volatility filter ---
    ann_vol    = train_returns.std() * np.sqrt(TRADING_DAYS)
    vol_filter = ann_vol[ann_vol < ann_vol.quantile(vol_quantile)].index

    # --- Momentum calculation (residual or raw) ---
    if use_residual_momentum:
        mkt_train = train_returns.iloc[:, 0]
        stocks_train = train_returns.iloc[:, 1:]
        mkt_var = (mkt_train @ mkt_train)
        betas_train = (stocks_train.T @ mkt_train) / mkt_var
        # residuals = r_stock - beta * r_mkt, then compound
        mkt_arr = np.asarray(mkt_train, dtype=float)
        betas_arr = np.asarray(betas_train, dtype=float)
        stocks_arr = np.asarray(stocks_train, dtype=float)
        resid = pd.DataFrame(
            stocks_arr - mkt_arr[:, None] * betas_arr[None, :],
            index=stocks_train.index,
            columns=stocks_train.columns,
        )
        cum_full = (1 + resid).prod() - 1
        cum_skip = (1 + resid.iloc[-skip_last:]).prod() - 1
    else:
        cum_full = (1 + train_returns).prod() - 1
        cum_skip = (1 + train_returns.iloc[-skip_last:]).prod() - 1

    momentum = cum_full - cum_skip
    top    = momentum[momentum >= momentum.quantile(mom_quantile_top)].index
    bottom = momentum[momentum <= momentum.quantile(mom_quantile_bottom)].index

    # --- Quality veto on shorts: drop short candidates with top-quartile quality ---
    if quality_veto_quantile is not None and len(quality_score) > 0:
        q_threshold = quality_score.quantile(quality_veto_quantile)
        high_quality = quality_score[quality_score >= q_threshold].index
        bottom = bottom.difference(high_quality)

    mom_filter = top.union(bottom)
    # --- Universe --- Vol checks out, Mom checks out, $ADV checks out, active lseg ric's
    volume_eligible_tickers = volume_eligible.columns[volume_eligible.loc[entry_date]]
    universe = vol_filter.intersection(mom_filter).intersection(volume_eligible_tickers)
    
    if len(universe) == 0:
        raise ValueError("No stocks passed the filters. Adjust parameters.")

    returns_uni = train_returns[universe]
    mkt_index   = train_returns.iloc[:,0]
    valid_idx   = mkt_index.dropna().index.intersection(returns_uni.index)

    returns_uni = returns_uni.loc[valid_idx]
    mkt_index   = mkt_index.loc[valid_idx]
    universe_list = list(universe)

    w, mu, cov, b = optimal_weights(returns_uni, mkt_index, top, bottom, universe_list, long_weight_max, short_weight_max, target_beta, target_vol, use_ewma_vol, sector_neutral, sector_map, sector_neutral_tol)
    weights = pd.Series(w, index=universe_list)

    # --- Asymmetric min-weight filtering ---
    long_keep  = (weights > 0) & (weights >= long_weight_min)
    short_keep = (weights < 0) & (weights <= -short_weight_min)
    weights = weights[long_keep | short_keep]

    if apply_screener:
        flagged, weights = screen_event_impact(
            weights=weights, 
            return_data=train_returns, 
            stock_index=stock_index, 
            reb_date=train_returns.index[-1], 
            days=screen_days, 
            z_threshold_long=screen_z_threshold_long,
            z_threshold_short=screen_z_threshold_short,
            target_beta=target_beta,
            plot=display_screener
        )
        if display_screener:
            print(flagged)

    full_weights = weights.copy()
    bench_w = float(full_weights.get(stock_index.benchmark, 0.0))
    total_shrunk_beta = portfolio_beta(full_weights, stock_index)

    # Trim to single-name positions in mu/cov/b for in-sample return/vol/sharpe
    stat_idx = weights.index.intersection(mu.index)
    mu, cov, b = mu.loc[stat_idx], cov.loc[stat_idx, stat_idx], b.loc[stat_idx]
    weights = weights.loc[stat_idx]

    insample_vol = float(np.sqrt(weights @ cov @ weights))

    stats = {
        'Portfolio Start':              str(portfolio_start),
        'In-Sample Return':             f'{mu @ weights:.2%}',
        'In-Sample Vol':                f'{insample_vol:.2%}',
        'In-Sample Sharpe':             f'{mu @ weights / insample_vol:.2f}',
        'Beta (single-name, raw)':      f'{(b * weights).sum():.3f}',
        'Beta (total, shrunk)':         f'{total_shrunk_beta:.3f}',
        'Benchmark hedge wt':           f'{bench_w:.3f}',
        'Positions':                    len(full_weights),
        'Longs':                        int((full_weights > 0).sum()),
        'Shorts':                       int((full_weights < 0).sum()),
        'Gross Leverage':               f'{full_weights.abs().sum():.2f}x',
    }

    return weights, stats

# ----- BS Reversal Strategy -----
def mean_reversion(
    return_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    stock_index: StockIndex,
    portfolio_start: str,
    lookback: int = 21,
    skip: int = 5,
    n_per_side: int = 50,
    vol_quantile: float = 0.90,
    beta_tol: float = 0.0
) -> tuple[pd.Series, dict[str, Any]]:
    """
    
    """

    all_dates = return_data.index
    start_idx = all_dates.searchsorted(pd.Timestamp(portfolio_start))
    entry_date = all_dates[min(start_idx, len(all_dates) - 1)]

    if start_idx < lookback + skip:
        raise ValueError(f"Need {lookback + skip} days before {entry_date}, have {start_idx}.")

    train_returns = return_data.iloc[start_idx - lookback - skip : start_idx - skip].dropna(axis=1, how='any')

    cum_return = (1 + train_returns).prod() - 1

    eligible_tickers = volume_eligible.columns[volume_eligible.loc[entry_date]]
    universe = train_returns.columns.intersection(eligible_tickers)
    cum_return = cum_return[universe].dropna()

    ann_vol = train_returns[cum_return.index].std() * np.sqrt(252)
    low_vol = ann_vol[ann_vol < ann_vol.quantile(vol_quantile)].index
    cum_return = cum_return[cum_return.index.intersection(low_vol)]

    # long losers, short winners - let data decide how many
    longs = cum_return.nsmallest(n_per_side).index
    shorts = cum_return.nlargest(n_per_side).index

    if len(longs) == 0 or len(shorts) == 0:
        raise ValueError("No stocks in long or short bucket.")

    # raw inverse-vol target weights
    raw = pd.Series(0.0, index=longs.union(shorts))
    long_iv = 1.0 / ann_vol[longs]
    raw[longs] = long_iv / long_iv.sum()
    short_iv = 1.0 / ann_vol[shorts]
    raw[shorts] = -short_iv / short_iv.sum()

    # fetch shrunken betas for the selected universe, drop names missing beta
    betas = pd.Series({inst: beta(inst, stock_index) for inst in raw.index})
    betas = betas.dropna()
    raw = raw.loc[betas.index]

    # per-name weight cap equal to 2x the max raw weight to keep book shape
    w_cap = float(raw.abs().max()) * 2.0

    # minimize ||w - raw||_2 s.t. dollar-neutral, shrunken-beta-neutral, sign-preserved
    w = cp.Variable(len(raw))
    raw_vec = raw.values
    beta_vec = betas.values

    sign = np.sign(raw_vec)
    constraints = [
        cp.sum(w) == 0,                              # dollar neutral
        cp.abs(beta_vec @ w) <= beta_tol,            # shrunken beta neutral (tol=0 by default)
        cp.multiply(sign, w) >= 0,                   # preserve long/short side from signal
        cp.abs(w) <= w_cap,                          # per-name cap
        cp.sum(cp.abs(w)) <= 2.0,                    # gross leverage cap at 2x (matches raw)
    ]
    objective = cp.Minimize(cp.sum_squares(w - raw_vec))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"Beta-neutralization failed: {problem.status}")

    weights = pd.Series(w.value, index=raw.index).round(8)
    # renormalize to unit gross leverage
    weights = weights / weights.abs().sum()

    stats = {
        'portfolio_start': str(portfolio_start),
        'positions': len(weights),
        'longs': len(longs),
        'shorts': len(shorts),
        'gross_leverage': f'{weights.abs().sum():.2f}x',
        'beta': round(portfolio_beta(weights, stock_index), 3)
    }

    return weights, stats

# ----- Value Strategy -----
def value_strategy(
    return_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    fundamentals: dict,
    stock_index: StockIndex,
    portfolio_start: str,
    lookback: int = 252,
    quantile_top: float = 0.85,
    quantile_bottom: float = 0.15,
    vol_quantile: float = 0.90,
    lag_days: int = 45,
) -> tuple[pd.Series, dict[str, Any]]:
    """Long cheap stocks (high earnings yield), short expensive. Inverse-vol weighted."""

    all_dates = return_data.index
    start_idx = all_dates.searchsorted(pd.Timestamp(portfolio_start))
    entry_date = all_dates[min(start_idx, len(all_dates) - 1)]
    avail_date = (pd.Timestamp(entry_date) - pd.Timedelta(days=lag_days)).date()


    if start_idx < lookback:
        raise ValueError(f"Need {lookback} days before {entry_date}, have {start_idx}.")

    train_returns = return_data.iloc[start_idx - lookback : start_idx].dropna(axis=1, how='any')

    # latest EPS with lag
    eps = fundamentals['earnings_per_share'].loc[:avail_date]
    if eps.empty:
        raise ValueError("No EPS data available.")
    eps = eps.iloc[-1]

    # price proxy from cumulative returns (relative level)
    cum_price = (1 + train_returns).prod()

    # earnings yield = EPS / price proxy
    common = eps.dropna().index.intersection(cum_price.index)
    earnings_yield = eps[common] / cum_price[common]
    earnings_yield = earnings_yield.replace([np.inf, -np.inf], np.nan).dropna()

    # filters
    eligible = volume_eligible.columns[volume_eligible.loc[entry_date]]
    earnings_yield = earnings_yield[earnings_yield.index.intersection(eligible)]

    ann_vol = train_returns[earnings_yield.index].std() * np.sqrt(252)
    low_vol = ann_vol[ann_vol < ann_vol.quantile(vol_quantile)].index
    earnings_yield = earnings_yield[earnings_yield.index.intersection(low_vol)]

    # long cheap (high EY), short expensive (low EY)
    longs = earnings_yield[earnings_yield >= earnings_yield.quantile(quantile_top)].index
    shorts = earnings_yield[earnings_yield <= earnings_yield.quantile(quantile_bottom)].index

    if len(longs) == 0 or len(shorts) == 0:
        raise ValueError("Empty long or short bucket.")

    weights = pd.Series(0.0, index=longs.union(shorts))
    inv_vol = 1.0 / ann_vol[longs]
    weights[longs] = inv_vol / inv_vol.sum()
    inv_vol = 1.0 / ann_vol[shorts]
    weights[shorts] = -inv_vol / inv_vol.sum()

    stats = {
        'portfolio_start': str(portfolio_start),
        'positions': len(weights),
        'longs': len(longs),
        'shorts': len(shorts),
        'gross_leverage': f'{weights.abs().sum():.2f}x',
    }

    return weights, stats

# ----- Optimised Quality Strategy -----
def quality_strategy(
    return_data: pd.DataFrame,
    volume_eligible: pd.DataFrame,
    fundamentals: dict,
    stock_index: StockIndex,
    portfolio_start: str,
    lookback: int = 252,
    quantile_top: float = 0.85,
    quantile_bottom: float = 0.15,
    vol_quantile: float = 0.90,
    lag_days: int = 45,
) -> tuple[pd.Series, dict[str, Any]]:
    """Long high quality (ROIC + low leverage), short low quality. Inverse-vol weighted."""

    all_dates = return_data.index
    start_idx = all_dates.searchsorted(pd.Timestamp(portfolio_start))
    entry_date = all_dates[min(start_idx, len(all_dates) - 1)]
    avail_date = (pd.Timestamp(entry_date) - pd.Timedelta(days=lag_days)).date()

    if start_idx < lookback:
        raise ValueError(f"Need {lookback} days before {entry_date}, have {start_idx}.")

    train_returns = return_data.iloc[start_idx - lookback : start_idx].dropna(axis=1, how='any')

    # latest ROIC and leverage with lag
    roic = fundamentals['return_on_invested_capital'].loc[:avail_date]
    debt_eq = fundamentals['long_term_debt_percentage_of_total_equity'].loc[:avail_date]

    if roic.empty or debt_eq.empty:
        raise ValueError("No fundamental data available.")

    roic = roic.iloc[-1]
    debt_eq = debt_eq.iloc[-1]

    common = roic.dropna().index.intersection(debt_eq.dropna().index).intersection(train_returns.columns)

    # z-score each, composite = high ROIC + low leverage
    roic_z = (roic[common] - roic[common].mean()) / roic[common].std()
    debt_z = (debt_eq[common] - debt_eq[common].mean()) / debt_eq[common].std()
    quality_score = roic_z - debt_z  # high ROIC, low debt = high score

    quality_score = quality_score.replace([np.inf, -np.inf], np.nan).dropna()

    # filters
    eligible = volume_eligible.columns[volume_eligible.loc[entry_date]]
    quality_score = quality_score[quality_score.index.intersection(eligible)]

    ann_vol = train_returns[quality_score.index].std() * np.sqrt(252)
    low_vol = ann_vol[ann_vol < ann_vol.quantile(vol_quantile)].index
    quality_score = quality_score[quality_score.index.intersection(low_vol)]

    # long high quality, short low quality
    longs = quality_score[quality_score >= quality_score.quantile(quantile_top)].index
    shorts = quality_score[quality_score <= quality_score.quantile(quantile_bottom)].index

    if len(longs) == 0 or len(shorts) == 0:
        raise ValueError("Empty long or short bucket.")

    weights = pd.Series(0.0, index=longs.union(shorts))
    inv_vol = 1.0 / ann_vol[longs]
    weights[longs] = inv_vol / inv_vol.sum()
    inv_vol = 1.0 / ann_vol[shorts]
    weights[shorts] = -inv_vol / inv_vol.sum()

    stats = {
        'portfolio_start': str(portfolio_start),
        'positions': len(weights),
        'longs': len(longs),
        'shorts': len(shorts),
        'gross_leverage': f'{weights.abs().sum():.2f}x',
    }

    return weights, stats


# ----- Small-cap Value Strategy -----
def small_cap_value():
    return 7
