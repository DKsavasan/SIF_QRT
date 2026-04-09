from typing import Literal

EXECUTION_COST_BPS = 0.0002
FINANCING_COST_ANNUAL = 0.005
TRADING_DAYS = 252

GROUP_ID = 'ICL05'
USER = 'q8576'
BACKTEST_RESULTS = 'backtest_results'


type Regions = Literal['AMER', 'EMEA']
type Markets = Literal['.SPX', '.STOXX50E']
type Instruments = Literal['RIC', 'ISIN']
