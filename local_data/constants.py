from typing import Literal, TypedDict

BB_HISTORICAL = 'historical_bb'
LSEG_ACTIVE = 'active_lseg'
DATA = 'data'
FUNDAMENTALS = 'fundamentals'
PRICE_VOLUME = 'price_volume'


type DataType = Literal['active', 'historical']
type DataColumns = Literal['Close', 'Volume']

class DataConfigItem(TypedDict):
    sub_dir: str
    inst_name: str

DATA_CONFIG: dict[DataType, DataConfigItem] = {
    'active': {'sub_dir': LSEG_ACTIVE, 'inst_name': 'RIC'},
    'historical': {'sub_dir': BB_HISTORICAL, 'inst_name': 'ISIN'}
}


LSEG_ACTIVE_CONSTITUENTS_FILE   = 'lseg_active_constituents.csv'
BB_HISTORICAL_CONSTITUENTS_FILE = 'bb_historical_constituents.csv'
BB_INDEX_CONSTITUENT_FOLDERS    = ['Russell_3000', 'Stoxx_600']

type Markets = Literal['.SPX', '.STOXX50E']
US_UNIVERSE, EU_UNIVERSE = '0#.RUA', '0#.STOXX'
UNIVERSES = [EU_UNIVERSE, US_UNIVERSE]
US_INDEX_BENCHMARK, EU_INDEX_BENCHMARK = '.SPX', '.STOXX50E'
INDEX_NAME_MAPPING = {'RAY': US_INDEX_BENCHMARK, 'SXXP': EU_INDEX_BENCHMARK}
DAILY_DATA_FIELDS = ["TR.PriceClose", "TR.Volume"]

Y_M_D = '%Y-%m-%d'

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
