# QRT

Equity trading with US (Russell 3000) and EU (Stoxx 600) universes.

## Setup

1. Connect to LSEG workspace
2. Run `download_all_data()` and `update_price_data()` in `LSEG_data.py`
3. Move `qsec-client/` to `qsec_client/` in the root directory of this project

## Files

- `testing.py`: Run tests with active and historical data to develop and iterate on new strategies
- `constants.py`: paths, data config, field definitions
- `strategies.py`: signal generation, screening, portfolio construction
- `backtest_utils.py`: backtest loop, PnL, performance metrics
- `QRT_utils.py`: data loading, plotting, SFTP execution
