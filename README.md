# QRT

Equity trading with US (Russell 3000) and EU (Stoxx 600) universes.

## Setup

1. Run `pip install -e .`
2. Run `download_all_data()` and `update_price_data()` in `qrt/data.py`
3. Move `qsec-client/` to `qsec_client/` in the root directory of this project

## Files

- `notebooks/research.ipynb`: Use active and historical index data to develop and backtest new strategies
- `qrt/data.py`: Functions to download LSEG data localy and load into notebooks
- `qrt/constants.py`: paths, data config, field definitions
- `qrt/strategies.py`: signal generation, screening, portfolio construction
- `qrt/utils.py`: backtest loop, PnL, performance metrics
- `qrt/qrt_utils.py`: data loading, plotting, SFTP execution
