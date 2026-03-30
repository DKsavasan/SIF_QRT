import os
import re
import time
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import lseg.data as ld
from datetime import date
import pyarrow.parquet as pq


pd.set_option('future.no_silent_downcasting', True)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, 'data')
PRICE_DATA_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'data', 'lseg')
INDEX_NAME_MAPPING = {'RAY': '.RUA', 'SXXP': '.STOXX50E'}
BB_COLUMN_RENAME = {'ISIN\n': 'ISIN', 'Sec Type\n': 'SecType'}
DAILY_DATA_FIELDS = ["TR.PriceClose", "TR.Volume"]


# ---------- LSEG SDK Functions ---------- #

def get_data(instruments: list, fields: list = ["TR.PrimaryRIC", "TR.ISIN", "TR.CommonName"]):
    ld.open_session()
    try:
        return ld.get_data(
            universe=instruments,  # RIC'S or ISIN's: ['0#.STOXX', '0#.RUA']
            fields=fields,
            parameters={"SDate": '2002-01-31'}
        )
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
            # " and MktCapCompanyUsd gt 0"
        )

        full_filter = f"{filter_on} and {base_filter}" if filter_on else base_filter

        return ld.discovery.search(
            view=ld.discovery.Views.EQUITY_QUOTES,
            top=10_000,
            filter=full_filter,
            select=select # "ISIN,RIC,ListingStatus,RetireDate,TickerSymbol,DTSubjectName,RCSTRBC2012Leaf,RCSCurrencyLeaf,RCSExchangeCountryLeaf,MktCapCompanyUsd,ExchangeName,ExchangeCode,PermID,IsPrimaryRIC,AvgVol90D"
        )
    finally:
        ld.close_session()


# ---------- Save Tradeable Index Constituents Localy ---------- #

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
    for filename in os.listdir(folder):
        if not filename.endswith(('.xlsx', '.xls')):
            continue

        year_match = re.search(r'(19|20)\d{2}', filename)
        if not year_match:
            logger.warning(f"No year in {filename}, skipping")
            continue

        try:
            df = pd.read_excel(os.path.join(folder, filename))
        except Exception as e:
            logger.warning(f"Failed to read {filename}: {e}")
            continue

        df.rename(columns=BB_COLUMN_RENAME, inplace=True)
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

def save_bloomberg_index_constituents(
    data_folders: list,
    out_file: str
):
    """
    Build a consolidated index constituents dataset by merging Bloomberg exports
    with LSEG RIC mappings, and save the result to a CSV file.

    For each specified folder, the function:
    - Parses Bloomberg constituent files into a standardized DataFrame.
    - Extracts unique ISINs and retrieves corresponding RICs from LSEG in batches.
    - Merges Bloomberg data with RIC mappings on ISIN.
    - Appends results across all folders into a single dataset.

    Parameters
    ----------
    data_folders : list, optional
        List of subfolder names (under DATA_DIR) containing Bloomberg export files.
        Each folder is processed independently and then combined. Default is ['Russell_3000'].
    out_file : str, optional
        Name of the output CSV file saved in DATA_DIR. Default is 'index_constituents.csv'.
    batch : int, optional
        Batch size for LSEG ISIN queries. Default is 1000.

    Returns
    -------
    None
        Writes the combined dataset to disk as a CSV file.
    """
    all_constituents = []
    for folder in data_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        constituents_bb = _parse_bloomberg_export(folder=folder_path)
        all_constituents.append(constituents_bb)

    combined = pd.concat(all_constituents, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, out_file), index=False)
    logger.info(f"Saved bloomberg constituents: {len(combined)} rows to {out_file}")

def _get_batch_data_lseg_RICs(instruments: list, batch: int = 1000) -> pd.DataFrame:
    """
    Fetch LSEG RICs for a list of ISINs in batches and return one valid RIC per ISIN.

    For each ISIN, rows with missing RICs are discarded, and the first non-null
    RIC encountered is retained. If multiple valid RICs exist for an ISIN, the
    selection is based on the original ordering of the data returned by LSEG


    Parameters
    ----------
    isins : list
        List of ISIN strings to query.
    batch : int, optional
        Number of ISINs per request batch (default is 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per ISIN with a non-null RIC with columns:

        - Instrument
        - ISIN
        - LsegName
        - PrimaryIssueRIC
    """
    results = []
    for i in range(0, len(instruments), batch):
        try:
            firms = get_data(instruments[i:i + batch], fields=["TR.CommonName", "TR.RIC"])
            results.append(firms)
        except Exception as e:
            logger.warning(f"get_data batch failed: {e}")

    if not results:
        raise RuntimeError("All LSEG batches failed")

    df = pd.concat(results, ignore_index=True)
    df.drop_duplicates(inplace=True)
    
    # df.rename(columns={'Company Common Name': 'LsegCompanyName'}, inplace=True)
    # df['PrimaryIssueRIC'] = df['Primary Issue RIC'].str.strip().replace('', pd.NA)
    # df.drop(columns=['Primary Issue RIC'], inplace=True)

    # mark which ISINs have at least one valid RIC
    valid_inst = df.loc[df['PrimaryIssueRIC'].notna(), 'Instrument'].unique()

    # keep rows where:
    # - RIC is not null OR
    # - ISIN has no valid RIC at all
    filtered_df = df[
        (df['PrimaryIssueRIC'].notna()) | (~df['Instrument'].isin(valid_inst))
    ]
    return filtered_df[~filtered_df['PrimaryIssueRIC'].isna()]

def update_isin_ric_mapping(out_file: str, indecies: list = ['0#.RUA', '0#.STOXX']):
    """
    Fetch LSEG RICs for a list of ISINs and save the mapping to a CSV.

    Parameters
    ----------
    out_file : str
        Name of the output CSV file saved in DATA_DIR.

    Returns
    -------
    None
    """
    instruments = pd.read_csv(os.path.join(DATA_DIR, 'bloomberg_index_constituents.csv'))['ISIN'].unique().tolist()

    logger.info('Fetching LSEG RUA, STOXX constituents...')
    current_rics = get_data(indecies, fields=["TR.ISIN", "TR.CommonName"])

    ric_lseg = current_rics[current_rics.ISIN.isin(instruments)].drop_duplicates('Instrument').sort_values('Instrument').reset_index(drop=True)
    ric_lseg.rename(columns={'Instrument': 'RIC'}, inplace=True)

    instruments_not_found = set(instruments) - set(ric_lseg)

    ric_lseg.to_csv(os.path.join(DATA_DIR, out_file), index=False)
    logger.info(f"Saved ISIN -> RIC mapping: {len(ric_lseg)} rows to {out_file}")


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

def _write_parquet_incremental(df, ric, upsert=False) -> None:
    """Append or create parquet file for a single RIC."""
    folder = os.path.join(PRICE_DATA_OUTPUT_DIR, f"RIC={ric}")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"part.parquet")

    # Keep only Date, Close, Volume
    df = df[['Date', 'Close', 'Volume']]
    if upsert and os.path.exists(file_path):
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

def download_all(rics, start_date, end_date, skip_existing=False, upsert=False, chunk_size=1, print_ric=True) -> None:
    ld.open_session()
    c = 0
    try:
        for chunk in _chunk_list(rics, chunk_size):
            if skip_existing:
                folder = os.path.join(PRICE_DATA_OUTPUT_DIR, f"RIC={chunk[0]}")
                file_path = os.path.join(folder, "part.parquet")

                if os.path.exists(file_path):
                    logger.warning("Skipped: ", chunk[0])
                    continue

            df = _fetch_chunk_with_retry(chunk=chunk, start_date=start_date, end_date=end_date)
            if df.empty:
                continue
            
            if len(chunk) > 1:
                # stack multi-RIC panel
                df = df.stack(level=0, future_stack=True).reset_index()
                df.rename(columns={'level_1': 'RIC'}, inplace=True)
            else:
                df['RIC'] = chunk[0]
                df.reset_index(inplace=True)
                df.columns.name = None
            
            df.rename(columns={'Price Close': 'Close'}, inplace=True)
            df.sort_values(['RIC','Date'], inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df.dropna(subset=['Close', 'Volume'], how='all', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # write each RIC incrementally
            for ric in df['RIC'].unique():
                c+=1
                if c % 1000 == 0:
                    logger.info(f"Downloaded data for {c} stocks")
                if print_ric:
                    logger.info(ric) 
                ric_df = df[df['RIC'] == ric].copy()
                _write_parquet_incremental(df=ric_df, ric=ric, upsert=upsert)

    finally:
        ld.close_session()


# ---------- Load Local LSEG Data for Backtesting ---------- #

def get_timeseries(data: pd.DataFrame, constituents: pd.DataFrame, value_col: str = 'Close', index: str = '.RUA'):
    """All data for stocks while they were listed in the index, plus the index itself."""
    constituents.rename(columns={'ISIN': 'RIC'}, inplace=True)

    years = data['Date'].dt.year.values
    rics = data['RIC'].values 

    # All historical index constituents since 2000-01-01
    constituents = constituents[constituents['Index']==index][['RIC', 'Year']].drop_duplicates()
    constituents['RIC'] = constituents['RIC'].astype(data['RIC'].dtype)

    valid_pairs = set(zip(constituents['RIC'], constituents['Year']))
    
    mask = np.fromiter(
        ((ric, year) in valid_pairs for ric, year in zip(rics, years)),
        dtype=bool,
        count=len(data)
    )
    filtered_prices = data.loc[mask].dropna(subset=[value_col])

    # Include index data
    index_prices = data[data['RIC']==index][['Date', value_col]].set_index('Date')[value_col]
    
    df = filtered_prices.drop_duplicates(subset=['RIC', 'Date'], keep='first')
    df = df.pivot(index="Date", columns="RIC", values=value_col).sort_index()
    df.insert(0, index, index_prices)

    # Filter on only trading days, i.e. index Close price exists
    valid_days = data[data['RIC']==index][['Date', 'Close']].set_index('Date')['Close'].dropna().index
    df = df.loc[df.index.intersection(valid_days)]
    
    return df

def eligible_to_trade(prices_df: pd.DataFrame, vol_df: pd.DataFrame, adv_threshold: float = 5e6, index: str = '.RUA') -> pd.DataFrame:
    """Boolean DataFrame, True where stock has sufficient liquidity to trade.
    ADV = rolling 60-day mean of currency volume (Close * Volume)."""
    
    currency_volume = vol_df.multiply(prices_df, fill_value=0)
    adv = currency_volume.shift(1).rolling(60, min_periods=1).sum() / 60

    eligible = adv >= adv_threshold
    eligible[index] = False  # index itself is never tradeable
    
    return eligible


if __name__ == '__main__':    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    if True:
        update_isin_ric_mapping(out_file='isin_ric_mapping.csv')

    if False:
        # Daily update: Upsert daily price/volume data
        traded_instruments = pd.read_csv('data/instrument_ric_mapping.csv')['Instrument'].dropna().unique().tolist()
        all_instruments = traded_instruments + ['.RUA', '.STOXX', '.STOXX50E']
        last_date_of_data = pd.read_parquet(f"data/lseg/RIC=.STOXX50E").dropna(subset=['Close']).iloc[-1].Date

        download_all(
            rics=all_instruments, 
            start_date=last_date_of_data.strftime('%Y-%m-%d'), 
            end_date=date.today().strftime('%Y-%m-%d'), 
            upsert=True, 
            chunk_size=200, 
            skip_existing=False, 
            print_ric=False
        )



