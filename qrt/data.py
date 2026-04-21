import re
import wrds
import time
import logging
import pandas as pd
import pyarrow as pa
import lseg.data as ld
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime

import qrt
from qrt.constants import (
    DataType, StockIndex, DataColumns, RUA, STOXX,
    DATA, PRICE_VOLUME, FUNDAMENTALS, LSEG_ACTIVE, BB_HISTORICAL, DATA_CONFIG,
    BB_INDEX_CONSTITUENT_FOLDERS, BB_HISTORICAL_CONSTITUENTS_FILE, LSEG_ACTIVE_CONSTITUENTS_FILE, 
    FUNDAMENTAL_METRICS_QUARTERLY, DAILY_DATA_FIELDS, INDEX_NAME_MAPPING
)


pd.set_option('future.no_silent_downcasting', True)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(qrt.__file__).resolve().parent.parent
DATA_DIR = _SCRIPT_DIR / DATA
PRICE_DIR = DATA_DIR / PRICE_VOLUME
FUNDAMENTALS_DIR = DATA_DIR / FUNDAMENTALS
WRDS_DIR = Path(DATA_DIR / "wrds")


# ---------- LSEG SDK Functions ---------- #

def get_data(instruments: list, fields: list = ["TR.PrimaryRIC", "TR.ISIN", "TR.CommonName"], date: str = None, batch_size: int = 5000):
    ld.open_session()
    try:
        if len(instruments) <= batch_size:
            return ld.get_data(
                universe=instruments, # RIC'S or ISIN's, ex: ['0#.STOXX', '0#.RUA']
                fields=fields,
                # parameters={"SDate": '2000-01-01' if date is None else date}
            )

        chunks = []
        for i in range(0, len(instruments), batch_size):
            chunk = ld.get_data(
                universe=instruments[i:i + batch_size],
                fields=fields,
                # parameters={"SDate": '2000-01-01' if date is None else date}
            )
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    finally:
        ld.close_session()

def get_history(rics: list = None, fields: list = ["TR.PriceClose", "TR.Volume"], start: str = "2026-03-20", end: str = "2026-03-28", interval: str = "1d"):
    ld.open_session()
    try:
        return ld.get_history(
            universe=rics, #['ACA.N', 'CAGR.PA'], #'0#.STOXX',
            fields=fields,
            start=start,
            end=end,
            interval=interval
        )
    finally:
        ld.close_session()
    
def discovery_search(select="RIC,ISIN,TickerSymbol,DTSubjectName,IsPrimaryRIC", filter_on=None):
    ld.open_session()
    try:
        base_filter = (
            "SearchAllCategoryv2 eq 'Equities' "
            " and MktCapCompanyUsd gt 0"
        )

        full_filter = f"{filter_on} and {base_filter}" if filter_on else base_filter

        return ld.discovery.search(
            view=ld.discovery.Views.EQUITY_QUOTES,
            top=10_000,
            filter=full_filter,
            select=select
        )
    finally:
        ld.close_session()


# ---------- Save Index Constituents: Historical and Tradeable ---------- #

def _parse_bloomberg_export(folder: str) -> pd.DataFrame:
    """
    Parse Bloomberg Excel export files from a folder into a unified DataFrame.

    The function scans a directory for Excel files ('.xlsx', '.xls') containing
    index constituent data, extracts the year from each filename, standardizes
    column names, and combines all valid files into a single DataFrame.

    Filenames are expected to contain a 4-digit year (e.g., 'RAY_2020.xlsx').
    The index name is inferred from the filename prefix and mapped using
    INDEX_NAME_MAPPING.

    Parameters
    ----------
    folder : str
        Path to the directory containing Bloomberg export files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        - Index : str
            Standardized index identifier (mapped via INDEX_NAME_MAPPING)
        - ISIN : string
            Security ISIN (non-null)
        - Year : Int64
            Year extracted from filename
        - BloombergName : str
            Security name from Bloomberg export
        - Sec Type : str (if present)
            Security type from Bloomberg export

        The DataFrame is sorted by Year (descending) and ISIN (ascending).

    Raises
    ------
    ValueError
        If no valid Excel files are found or successfully parsed.

    Notes
    -----
    - Files without a valid year in the filename are skipped.
    - Files that fail to load are skipped with a warning.
    - Rows with missing ISIN values are dropped.
    """
    historical_constituents = []
    for f in folder.iterdir():
        filename = f.name
        if not filename.endswith(('.xlsx', '.xls')):
            continue

        year_match = re.search(r'(19|20)\d{2}', filename)
        if not year_match:
            logger.warning(f"No year in {filename}, skipping")
            continue

        try:
            df = pd.read_excel(folder / filename)
        except Exception as e:
            logger.warning(f"Failed to read {filename}: {e}")
            continue

        df.columns = df.columns.str.strip()
        df['Year'] = int(year_match.group())
        df.insert(0, 'Index', filename.split('_')[0])
        historical_constituents.append(df)

    if not historical_constituents:
        raise ValueError(f"No valid files in {folder}")

    df = pd.concat(historical_constituents, ignore_index=True)
    df['Year'] = df['Year'].astype('Int64')
    df['ISIN'] = df['ISIN'].astype('string')
    df['Index'] = df['Index'].map(INDEX_NAME_MAPPING)
    df.dropna(subset=['ISIN'], inplace=True)
    return df.sort_values(['Year', 'ISIN'], ascending=[False, True])

def _has_historical_data(instruments: list, batch: int = 5000) -> list:
    """
    Filter a list of RICs or ISINs and return only those with historical price data on LSEG.

    Parameters
    ----------
    instruments : list
        List of RICs or ISINs to check.
    batch : int, optional
        Number of instruments per API request batch (default 5000).

    Returns
    -------
    list
        Subset of input instruments that have at least one historical price point.
    """
    logger.info("Searching for instruments with historical data...")
    instruments_with_data = []
    for i in range(0, len(instruments), batch):
        try:
            inst_batch = instruments[i:i + batch]
            inst_w_data_batch = get_history(
                # last data point is always retrieved, thats why this works, even for delisted RIC's
                inst_batch, fields = ["TR.PriceClose"], start=datetime.now().strftime('%Y-%m-%d')
            ).dropna(axis=1, how='all').columns.to_list()
            logger.info(f"Found {len(inst_w_data_batch)} RIC's with data in batch")
            instruments_with_data += inst_w_data_batch
        except Exception as e:
            logger.warning(f"get_history batch failed: {e}")

    return list(set(instruments_with_data))

def save_bloomberg_historical_constituents(print_only: bool = False) -> None:
    """
    Consolidate historical Bloomberg index constituents with LSEG RIC mappings and save to CSV.

    This function:
    - Reads Bloomberg constituent files from multiple folders.
    - Standardizes the data into a single DataFrame.
    - Retrieves RICs for each unique ISIN and checks which have historical price data on LSEG.
    - Adds a boolean column 'HasLsegData' indicating availability of historical data.
    - Optionally prints the resulting DataFrame and summary statistics.
    - Otherwise, saves the dataset to disk as CSV.

    Parameters
    ----------
    print_only : bool
        Print the table instead of writing it to the file.

    Returns
    -------
    None
        Writes the combined dataset to disk as a CSV file.
    """
    logger.info("Parsing bloomberg constituents excel files...")
    all_constituents = []
    for folder in BB_INDEX_CONSTITUENT_FOLDERS:
        folder_path = DATA_DIR / folder
        constituents_bb = _parse_bloomberg_export(folder=folder_path)
        all_constituents.append(constituents_bb)

    bb_constituents = pd.concat(all_constituents, ignore_index=True)[['Index', 'Name', 'ISIN', 'Year']]

    # Get unique isin's, filter on 'bb_constituents' with lseg data, assign new col for HasLsegData (bool)
    bb_isins = bb_constituents.ISIN.unique().tolist()
    logger.info(f"Found {len(bb_isins)} bloomberg constituents")
    bb_isins_with_data = _has_historical_data(instruments=bb_isins)
    bb_constituents['HasLsegData'] = bb_constituents.ISIN.isin(bb_isins_with_data)

    if print_only:
        print(bb_constituents)
        print(bb_constituents.groupby(['Index', 'Year']).nunique())
        print(f"Prepared {len(bb_constituents)} rows, and {len(bb_isins_with_data)} unique bloomberg constituents with data")
    else:
        bb_constituents.to_csv(DATA_DIR / BB_HISTORICAL_CONSTITUENTS_FILE, index=False)
        logger.info(f"Saved {len(bb_constituents)} rows, and {len(bb_isins_with_data)} unique bloomberg constituents with data to {BB_HISTORICAL_CONSTITUENTS_FILE}")

def save_lseg_active_constituents(print_only: bool = False) -> None:
    """
    Fetch and save all active LSEG RUA/STOXX constituents with RICs and ISINs.

    For each universe (US/EU), retrieves active constituents, drops duplicate RICs, 
    checks which have historical price data, and either prints the table or saves it to CSV.

    Parameters
    ----------
    print_only : bool
        If True, prints the DataFrame instead of saving to file.
    """
    logger.info("Fetching lseg active constituents...")
    # Fetch all active RUA/STOXX constituents from lseg
    active_rua = get_data([RUA.universe], fields=["TR.ISIN", "TR.CommonName"]).assign(Index=RUA.benchmark)
    active_stoxx = get_data([STOXX.universe], fields=["TR.ISIN", "TR.CommonName"]).assign(Index=STOXX.benchmark)
    lseg_active_rics = pd.concat(
        [active_rua, active_stoxx]
    ).rename(columns={'Company Common Name': 'Name', 'Instrument': 'RIC'})[['Index', 'RIC', 'ISIN', 'Name']]
    lseg_active_rics = lseg_active_rics.drop_duplicates(subset='RIC', keep='first').reset_index(drop=True)

    active_rics_with_data = _has_historical_data(instruments=lseg_active_rics.RIC.unique().tolist())
    lseg_active_rics['HasLsegData'] = lseg_active_rics.RIC.isin(active_rics_with_data)

    if print_only:
        print(lseg_active_rics)
        print(lseg_active_rics.groupby('Index').nunique())
        print(f"Prepared {len(lseg_active_rics)} rows containing {len(active_rics_with_data)} RIC's with data")
    else:
        lseg_active_rics.to_csv(DATA_DIR / LSEG_ACTIVE_CONSTITUENTS_FILE, index=False)
        logger.info(f"Saved {len(lseg_active_rics)} rows containing {len(active_rics_with_data)} RIC's with data to {LSEG_ACTIVE_CONSTITUENTS_FILE}")

def get_bloomberg_historical_constituents(has_data: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with the following columns:
    - Index        : str, benchmark index
    - Name         : str, company common name
    - ISIN         : str, ISIN identifier
    - Year         : int, year of the constituent in the index
    - HasLsegData  : bool, whether the instrument has historical price data
    """
    df = pd.read_csv(DATA_DIR / BB_HISTORICAL_CONSTITUENTS_FILE)
    return df[df.HasLsegData==has_data]

def get_lseg_active_constituents(has_data: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with the following columns:
    - Index : str, benchmark index
    - RIC   : str, LSEG RIC
    - ISIN  : str, ISIN identifier
    - Name  : str, company common name
    - HasLsegData : bool, whether the instrument has historical price data
    """
    df = pd.read_csv(DATA_DIR / LSEG_ACTIVE_CONSTITUENTS_FILE)
    return df[df.HasLsegData==has_data]


# ---------- Upsert Raw Price/Volume Data For All Constituents to Parquet Locally ---------- #

def _chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _fetch_chunk_with_retry(chunk, start_date, end_date, max_retry: int = 1, retry_delay: int = 30) -> pd.DataFrame:
    """Fetch one chunk with retry on timeout / errors."""
    for attempt in range(1, max_retry + 1):
        try:
            df = ld.get_history(
                universe=chunk,
                fields=DAILY_DATA_FIELDS,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            return df
        except Exception as e:
            if attempt < max_retry:
                time.sleep(retry_delay)
            else:
                logger.warning(f"Skipping {chunk} after {max_retry} failed attempts", exc_info=e)
                return pd.DataFrame()

def _write_parquet_incremental(df, inst, sub_folder, inst_name='RIC', upsert=False) -> None:
    """Append or create parquet file for a single instrument."""
    folder = PRICE_DIR / sub_folder / f"{inst_name}={inst}"
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"part.parquet"

    # Keep only Date, Close, Volume
    df = df[['Date', 'Close', 'Volume']]
    if df.empty:
        logger.warning(f"Empty data for {inst}")
        return

    if upsert and file_path.exists():
        try:
            existing = pq.read_table(file_path).to_pandas()
            # Read existing parquet
            existing = existing[[c for c in ['Date', 'Close', 'Volume'] if c in existing.columns]]

            # Concatenate and drop duplicates (if any)
            df = pd.concat([existing, df], ignore_index=True)
            df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
        except Exception as e:
            logger.warning(f"Could not read existing parquet {file_path}: {e}")

    # Write to parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, file_path)

def download_all_prices(instruments, sub_folder, start_date, end_date, inst_name='RIC', skip_existing=False, upsert=False, chunk_size=1, print_inst=False, sample_size=None) -> None:
    ld.open_session()
    c = 0
    try:
        for chunk in _chunk_list(instruments, chunk_size):
            if skip_existing:
                folder = PRICE_DIR / sub_folder / f"{inst_name}={chunk[0]}"
                file_path = folder / "part.parquet"

                if file_path.exists(): 
                    logger.info(f"Skipped existing: {chunk[0]}")
                    continue

            df = _fetch_chunk_with_retry(chunk=chunk, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning(f"Empty data: {chunk}")
                continue
            
            if len(chunk) > 1:
                # stack multi-RIC panel
                df = df.stack(level=0, future_stack=True).reset_index()
                df.rename(columns={'level_1': inst_name}, inplace=True)
            else:
                df[inst_name] = chunk[0]
                df.reset_index(inplace=True)
                df.columns.name = None
            
            df.rename(columns={'Price Close': 'Close'}, inplace=True)
            df.sort_values([inst_name, 'Date'], inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'])
            df.dropna(subset=['Date'], inplace=True)
            df.dropna(subset=['Close', 'Volume'], how='all', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # write each instrument incrementally
            for inst in df[inst_name].unique():
                c+=1
                if c % 1000 == 0:
                    logger.info(f"Downloaded data for {c} stocks")
                if print_inst:
                    logger.info(inst) 
                if c==sample_size:
                    logger.info(f"Downloaded all data for sample size {c} stocks")
                    return
                inst_df = df[df[inst_name] == inst].copy()

                _write_parquet_incremental(
                    df=inst_df, inst=inst, sub_folder=sub_folder, inst_name=inst_name, upsert=upsert
                )

    finally:
        ld.close_session()

def save_fundamental_data(instruments: list, sub_folder: str, start_date: str = '2000-01-01', inst_name: str = 'RIC', batch: int = 10, sample_size: int = None, skip_existing: bool = False, upsert: bool = True, max_retries: int = 3):
    """
    Rewrites all fundamentals data for the specified instruments.
    """
    logger.info(f"Fetching fundamental data for {len(instruments)} {inst_name}'s...")
    existing_insts = set()
    (FUNDAMENTALS_DIR / sub_folder).mkdir(parents=True, exist_ok=True)
    for f in (FUNDAMENTALS_DIR / sub_folder).iterdir():
        assert f.name.startswith(f"{inst_name}="), f"Unexpected file in folder: {f}"
        existing_insts.add(f.name.split(f"{inst_name}=")[1])

    c=0
    for instrument_chunk in _chunk_list(instruments, batch):

        if skip_existing:
            instrument_chunk = [inst for inst in instrument_chunk if inst not in existing_insts]
            if not instrument_chunk:
                continue

        for attempt in range(1, max_retries + 1):
            try:
                df = get_history(instrument_chunk, fields=FUNDAMENTAL_METRICS_QUARTERLY, start=start_date)
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for {instrument_chunk}: {e}. Retrying in 30s...")
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} attempts failed for {instrument_chunk}: {e}")
                    continue

        instrument_cols = df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else instrument_chunk
        df = df.apply(pd.to_numeric, errors='coerce')

        for inst in instrument_cols:
            inst_df = df[inst] if isinstance(df.columns, pd.MultiIndex) else df
            if inst_df.empty:
                continue
            inst_df.index = inst_df.index.date
            inst_df = inst_df.groupby(inst_df.index).last()
            inst_df = inst_df.dropna(how='all')
            inst_df.sort_index(inplace=True)

            folder = FUNDAMENTALS_DIR / sub_folder / f"{inst_name}={inst}"
            folder.mkdir(parents=True, exist_ok=True)
            file_path = folder / f"part.parquet"

            # Upsert
            if upsert and file_path.exists():
                existing = pd.read_parquet(file_path)
                inst_df = pd.concat([existing, inst_df])
                inst_df = inst_df[~inst_df.index.duplicated(keep='last')]
                inst_df.sort_index(inplace=True)

            table = pa.Table.from_pandas(inst_df)
            pq.write_table(table, file_path)
            c+=1
            if c==sample_size:
                logger.info(f"Downloaded fundamental data for sample size {c} stocks")
                return
            
        if c%batch == 0:
            logger.info(f"Downloaded fundamental data for {c} stocks. This chunk: {instrument_chunk}")


# ---------- Download All Data ----------

def download_all_data(
        active_price_data: bool=True,
        historical_price_data: bool=True,
        active_fundamentals: bool=False,
        historical_fundamentals: bool=False,
        fundamentals_batch: int=50,
        start_date: str = '2000-01-01',
        sample_size: int = None,
        skip_existing=True,
        update_lseg_active_constituents: bool=False,
        update_bb_historical_constituents: bool=False
    ) -> None:

    if update_lseg_active_constituents:
        save_lseg_active_constituents()

    if update_bb_historical_constituents:
        save_bloomberg_historical_constituents()

    lseg_active_rics = sorted(get_lseg_active_constituents().RIC.unique().tolist())
    if active_price_data:
        download_all_prices(
            instruments=lseg_active_rics + [RUA.benchmark, STOXX.benchmark],
            sub_folder=LSEG_ACTIVE,
            start_date=start_date,
            end_date=str(datetime.now().date()),
            inst_name='RIC',
            skip_existing=skip_existing,
            upsert=True,
            chunk_size=1,
            print_inst=True,
            sample_size=sample_size
        )

    if active_fundamentals:
        save_fundamental_data(
            instruments=lseg_active_rics,
            sub_folder=LSEG_ACTIVE,
            start_date=start_date,
            inst_name='RIC',
            sample_size=sample_size,
            batch=fundamentals_batch,
            skip_existing=skip_existing,
            upsert=True
        )
    
    bb_historical_inst = sorted(get_bloomberg_historical_constituents().ISIN.unique().tolist())
    if historical_price_data:
        download_all_prices(
            instruments=bb_historical_inst,
            sub_folder=BB_HISTORICAL,
            start_date=start_date,
            end_date=str(datetime.now().date()),
            inst_name='ISIN',
            skip_existing=skip_existing,
            upsert=True,
            chunk_size=1,
            print_inst=True,
            sample_size=sample_size
        )

    if historical_fundamentals:
        save_fundamental_data(
            instruments=bb_historical_inst,
            sub_folder=BB_HISTORICAL,
            start_date=start_date,
            inst_name='ISIN',
            sample_size=sample_size,
            batch=fundamentals_batch,
            skip_existing=skip_existing,
            upsert=True
        )

def update_price_data(batch: int=5000):
    """
    Run this once you have data downloaded up to a recent date
    """
    lseg_active = get_lseg_active_constituents()

    lseg_active_dir = PRICE_DIR / LSEG_ACTIVE
    last_date_stoxx = pd.read_parquet(lseg_active_dir / f"RIC={STOXX.benchmark}").dropna(subset=['Close']).iloc[-1].Date
    last_date_spx = pd.read_parquet(lseg_active_dir / f"RIC={RUA.benchmark}").dropna(subset=['Close']).iloc[-1].Date

    download_all_prices(
        instruments=lseg_active.RIC.unique().tolist() + [RUA.benchmark, STOXX.benchmark],
        sub_folder=LSEG_ACTIVE,
        start_date=min(last_date_stoxx, last_date_spx),
        end_date=str(datetime.now().date()),
        inst_name='RIC',
        skip_existing=False,
        upsert=True,
        chunk_size=batch,
        print_inst=False
    )


# ---------- Fetch Saved Data ----------

def get_fundamental_data(inst: str, data_type: DataType = 'active'):
    config = DATA_CONFIG[data_type]
    data_dir = FUNDAMENTALS_DIR / config['sub_dir']
    data = pd.read_parquet(data_dir / f"{config['inst_name']}={inst}/part.parquet")
    return data

def get_single_timeseries(inst: str | list, value: DataColumns = 'Close', data_type: DataType = 'active'):
    if isinstance(inst, list):
        return pd.concat({i: get_single_timeseries(i, value, data_type) for i in inst}, axis=1)
    config = DATA_CONFIG[data_type]
    data_dir = PRICE_DIR / config['sub_dir']
    df = pd.read_parquet(data_dir / f"{config['inst_name']}={inst}/part.parquet")[['Date', value]]
    df = df.drop_duplicates(subset=['Date'], keep='last')
    return df.set_index('Date')[value].dropna()

def get_timeseries(data: pd.DataFrame, value: DataColumns = 'Close', data_type: DataType = 'active', stock_index: type[StockIndex] = RUA):
    """Pivot price data into a DataFrame of stocks filtered by index constituents, with the market index in the first column.
    Single-day holiday gaps are forward-filled.

    Parameters:
        data: Long-format price data with Date, instrument ID, and price columns.
        value: Price column to pivot on.
        data_type: Active (LSEG) or historical (Bloomberg) data source.
        stock_index: Stock index: RUA or STOXX.

    Returns:
        Wide DataFrame with Date index, market index as first column, and constituent prices.
    """
    inst_name = DATA_CONFIG[data_type]['inst_name']
    if data_type=='historical':
        constituents = get_bloomberg_historical_constituents()
        constituents = constituents[constituents['Index'] == stock_index.benchmark]
        constituents_years = constituents[[inst_name, 'Year']].drop_duplicates()

        data_with_year = data.assign(Year=data['Date'].dt.year)
        data_with_year = data_with_year[data_with_year[inst_name].isin(constituents[inst_name])]
        filtered_prices = (
            data_with_year.merge(constituents_years, on=[inst_name, 'Year'], how='inner')
                .drop(columns='Year')
                .dropna(subset=[value])
        )
    else:
        constituents = get_lseg_active_constituents()
        constituents = constituents[constituents['Index'] == stock_index.benchmark]
        filtered_prices = data[data[inst_name].isin(constituents[inst_name])].dropna(subset=[value])

    # Include index data
    index_prices = get_single_timeseries(inst=stock_index.benchmark, data_type='active')
    
    df = filtered_prices.drop_duplicates(subset=[inst_name, 'Date'], keep='last')
    df = df.pivot(index="Date", columns=inst_name, values=value).sort_index()

    # Filter on only trading days, i.e. index Close price exists
    df = df.loc[df.index.intersection(index_prices.index)]

    # drop index column if it somehow exists already
    df = df.drop(columns=stock_index.benchmark, errors='ignore')

    # now safe to insert
    df.insert(0, stock_index.benchmark, index_prices)

    # Fill holidays
    df = df.ffill(limit=1)
    
    return df

def eligible_to_trade(prices_df: pd.DataFrame, volume_df: pd.DataFrame, ADV_threshold: float = 5e6, stock_index: type[StockIndex] = RUA) -> pd.DataFrame:
    """Boolean DataFrame, True where stock has sufficient liquidity to trade.
    ADV = rolling 60-day mean of currency volume (Close * Volume)."""
    
    currency_volume = volume_df * prices_df
    adv = currency_volume.shift(1).rolling(60, min_periods=1).sum() / 60

    eligible = adv >= ADV_threshold
    eligible[stock_index.benchmark] = True  # market index itself is always tradeable
    
    return eligible


# ---------- Fetch WRDS RUA and Options Data ----------

def get_connection():
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username="dccunning") # UsB)s_@=Fmt_8qy
    print("Connected.\n")
    return db

def save_wrds_daily_metrics_rua(start_date: str = '2010-01-01', end_date: str = '2025-01-01'):
    db = get_connection()

    # This gives you the full RUA-equivalent universe, any date range
    rua_daily = db.raw_sql(f"""      
        SELECT a.permno,                   -- CRSP permanent ID, never changes
            b.ticker,                   -- ticker symbol (can change over time)
            b.exchcd,                   -- 1=NYSE, 2=AMEX, 3=NASDAQ
            b.ncusip,                   -- 8-char CUSIP, links to OptionMetrics
            a.date,                     -- trading date
            a.ret,                      -- daily holding period return (split/div adjusted)
            abs(a.prc) as price,        -- closing price (abs because negative = bid/ask mid)
            a.vol as volume,            -- daily share volume
            a.shrout * 1000 as shares,  -- shares outstanding (in thousands)
            abs(a.prc) * a.shrout / 1000 as mktcap_mm,  -- market cap in $MM
            c.dlret,                    -- delisting return (NULL unless stock delists that day)
            c.dlstcd                    -- delisting code (200s=merger, 400s=bankruptcy, etc)
        FROM crsp.dsf AS a
        INNER JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND a.date BETWEEN b.namedt AND b.nameendt
        LEFT JOIN crsp.dsedelist AS c
            ON a.permno = c.permno
            AND a.date = c.dlstdt
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
        AND b.shrcd IN (10, 11)       -- common stock only (no ETFs/ADRs/REITs)
        AND b.exchcd IN (1, 2, 3)     -- major US exchanges only
        AND a.prc IS NOT NULL          -- has a price
    """)

    rua_daily.to_parquet(DATA_DIR / 'wrds/crsp_rua_daily.parquet', index=False, engine='pyarrow')
    print(f"Rows: {len(rua_daily):,}")
    print(f"Unique stocks: {rua_daily['permno'].nunique()}")
    print(f"Date range: {rua_daily['date'].min()} to {rua_daily['date'].max()}")
    print(f"Size: {Path(DATA_DIR / 'wrds/crsp_rua_daily.parquet').stat().st_size / 1e9:.2f} GB")

def pull_universe(db, n_stocks, permno_filter: list = None):
    """Top N US common stocks by market cap via CRSP."""
    print(f"=== Pulling top {n_stocks} stocks by market cap ===")

    permno_clause = ""
    if permno_filter:
        permno_str = ",".join([str(int(p)) for p in permno_filter])
        permno_clause = f"AND a.permno IN ({permno_str})"

    mktcap = db.raw_sql(f"""
        SELECT a.permno, b.ticker, b.ncusip,
               abs(a.prc) * a.shrout / 1000 as mktcap_mm
        FROM crsp.dsf AS a
        INNER JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE a.date = (SELECT max(date) FROM crsp.dsf)
          AND b.shrcd IN (10, 11)
          AND a.prc IS NOT NULL
          AND a.shrout IS NOT NULL
          {permno_clause}
        ORDER BY mktcap_mm DESC
        LIMIT {n_stocks}
    """)

    print(f"  Stocks: {len(mktcap)}")
    print(f"  Mkt cap range: ${mktcap['mktcap_mm'].min():.0f}M - ${mktcap['mktcap_mm'].max():.0f}M")
    print()
    return mktcap

def link_to_optionmetrics(db, mktcap):
    """Map CRSP ncusip -> OptionMetrics secid."""
    print("=== Linking CRSP -> OptionMetrics via CUSIP ===")

    cusip_str = ",".join([f"'{c}'" for c in mktcap['ncusip'].dropna().tolist()])

    secid_map = db.raw_sql(f"""
        SELECT DISTINCT secid, ticker, cusip
        FROM optionm.secnmd
        WHERE cusip IN ({cusip_str})
    """)

    secid_map = secid_map.drop_duplicates(subset='secid', keep='last')

    merged = secid_map.merge(
        mktcap[['ncusip', 'permno', 'ticker', 'mktcap_mm']],
        left_on='cusip',
        right_on='ncusip',
        how='inner',
        suffixes=('_om', '_crsp')
    )

    print(f"  Matched: {merged['secid'].nunique()} / {len(mktcap)} stocks")
    print()
    return merged

def pull_options(db, secids, year, min_dte, max_dte):
    """Pull options data for given secids and year."""
    secid_str = ",".join([str(int(s)) for s in secids])

    print(f"  Querying opprcd{year}...", end=" ", flush=True)
    t0 = time.time()

    df = db.raw_sql(f"""
        SELECT secid, date, exdate, cp_flag,
               strike_price / 1000 as strike,
               best_bid, best_offer,
               volume, open_interest,
               impl_volatility, delta
        FROM optionm.opprcd{year}
        WHERE secid IN ({secid_str})
          AND cp_flag IN ('C', 'P')
          AND best_bid > 0
          AND exdate - date BETWEEN {min_dte} AND {max_dte}
    """)

    elapsed = time.time() - t0
    print(f"{len(df):,} rows in {elapsed:.0f}s")
    return df

def main(n_stocks: int = 500, years: list = [2024], min_dte: int = 1, max_dte: int = 30, permno_filter: list = None):
    db = get_connection()

    try:
        mktcap = pull_universe(db, n_stocks, permno_filter)
        secid_map = link_to_optionmetrics(db, mktcap)
        secids = secid_map['secid'].unique()

        secid_map.to_parquet(WRDS_DIR / "secid_map.parquet", index=False)

        print(f"=== Pulling options data for {len(secids)} stocks across {years} ===")
        total_rows = 0

        for year in years:
            df = pull_options(db, secids, year, min_dte, max_dte)
            out_path = WRDS_DIR / "options" / f"{year}.parquet"
            df.to_parquet(out_path, index=False)
            total_rows += len(df)
            print(f"  Saved {out_path.name}: {out_path.stat().st_size / 1e6:.1f} MB")
            del df 

        print(f"\n  Total rows across all years: {total_rows:,}")
        print("Saved: secid_map.parquet")


    finally:
        db.close()


if __name__ == "__main__":
    permnos = pd.read_csv(DATA_DIR / 'tmp.csv').permno.to_list()

    main(n_stocks=1600, years=[2022, 2023, 2024, 2025], permno_filter=permnos)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    # update_price_data()

    download_all_data(
        active_price_data=False,                  # 2-3 hours?
        historical_price_data=False,              # 12-18 hours?
        active_fundamentals=False, 
        historical_fundamentals=False,
        fundamentals_batch=10,                    # Number of stocks to fetch in one call
        start_date='2000-01-01',                  # Applies to all data fetched
        skip_existing=True,
        sample_size=None,                         # Max number of stocks to write data for
        update_lseg_active_constituents=False,    # 10 min?
        update_bb_historical_constituents=False   # 30 min?
    )
