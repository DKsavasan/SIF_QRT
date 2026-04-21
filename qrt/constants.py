from typing import Literal, TypedDict
from dataclasses import dataclass

# Costs & Global Settings
EXECUTION_COST_BPS = 0.0002
FINANCING_COST_ANNUAL = 0.005
TRADING_DAYS = 252

GROUP_ID = 'ICL05'
USER = 'q8576'

BACKTEST_RESULTS = 'backtest_results'
DATA = 'data'
PRICE_VOLUME = 'price_volume'
FUNDAMENTALS = 'fundamentals'

LSEG_ACTIVE = 'active_lseg'
BB_HISTORICAL = 'historical_bb'

LSEG_ACTIVE_CONSTITUENTS_FILE = 'lseg_active_constituents.csv'
BB_HISTORICAL_CONSTITUENTS_FILE = 'bb_historical_constituents.csv'
BB_INDEX_CONSTITUENT_FOLDERS = ['Russell_3000', 'Stoxx_600']


# Index Modeling
@dataclass(frozen=True)
class StockIndex:
    name: Literal['RUA', 'STOXX']
    benchmark: Literal['.SPX', '.STOXX50E']
    region: Literal['AMER', 'EMEA']
    currency: Literal['USD', 'EUR']
    universe: Literal['0#.RUA', '0#.STOXX']


RUA = StockIndex(
    name='RUA',
    benchmark='.SPX',
    region='AMER',
    currency='USD',
    universe='0#.RUA',
)

STOXX = StockIndex(
    name='STOXX',
    benchmark='.STOXX50E',
    region='EMEA',
    currency='EUR',
    universe='0#.STOXX',
)

STOCK_INDICES = {'RUA': RUA, 'STOXX': STOXX}


# Data Types
type Region = Literal['AMER', 'EMEA']
type DataType = Literal['active', 'historical']
type DataColumns = Literal['Close', 'Volume']

class DataConfigItem(TypedDict):
    sub_dir: str
    inst_name: str

DATA_CONFIG: dict[DataType, DataConfigItem] = {
    'active': {'sub_dir': LSEG_ACTIVE, 'inst_name': 'RIC'},
    'historical': {'sub_dir': BB_HISTORICAL, 'inst_name': 'ISIN'},
}

INDEX_NAME_MAPPING = {
    'RAY': '.SPX',
    'SXXP': '.STOXX50E',
}

DAILY_DATA_FIELDS = ["TR.PriceClose", "TR.Volume"]

FUNDAMENTAL_METRICS_QUARTERLY = [
    "TR.RevenueActValue(Period=FQ0)",        # Revenue
    "TR.COGSActValue(Period=FQ0)",           # Cost of Goods Sold
    "TR.OperatingExpActual(Period=FQ0)",     # Operating Expenses
    "TR.NetProfitActValue(Period=FQ0)",      # Net Profit
    "TR.ROICActValue(Period=FQ0)",           # ROIC
    "TR.ROEActValue(Period=FQ0)",            # ROE
    "TR.ROAActValue(Period=FQ0)",            # ROA
    "TR.EPSActValue(Period=FQ0)",            # EPS
    "TR.TotalDebtActValue(Period=FQ0)",      # Total Debt
    "TR.F.LTDebtPctofTotEq(Period=FQ0)",     # Long-term debt % of total equity
    "TR.F.EarnRetentionRate(Period=FQ0)",    # Earnings Retention Rate
]
