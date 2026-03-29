import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, Lock
import logging

fuel_burn_global = None

# Sets the global variable to the value passed in as the argument. This pattern is commonly used
# in multiprocessing scenarios where worker processes need access to a large, read-only object
# (like a DataFrame) without passing it as an argument to every worker,
# which would be inefficient and could cause serialization issues.
def init_fuel_burn(fuel_burn):
    global fuel_burn_global
    fuel_burn_global = fuel_burn

# Shared logging setup
log_lock = Lock()

# Logger for missing parquet files
parquet_logger = logging.getLogger("parquet_logger")
parquet_logger.setLevel(logging.INFO)
parquet_handler = logging.FileHandler('missing_parquet_files.log')
parquet_logger.addHandler(parquet_handler)

# Logger for missing timeranges
timerange_logger = logging.getLogger("timerange_logger")
timerange_logger.setLevel(logging.INFO)
timerange_handler = logging.FileHandler('missing_timeranges.log')
timerange_logger.addHandler(timerange_handler)

# Logger for missing columns
columns_logger = logging.getLogger("columns_logger")
columns_logger.setLevel(logging.INFO)
columns_handler = logging.FileHandler('missing_columns.log')
columns_logger.addHandler(columns_handler)

def log_missing_timerange(flight_id, segment_idx, start, end):
    with log_lock:
        timerange_logger.info(f"Missing timerange: flight_id={flight_id}, segment_idx={segment_idx}, start={start}, end={end}")

def log_missing_columns(flight_id, missing_cols):
    with log_lock:
        columns_logger.info(f"Missing columns in flight_id={flight_id}: {missing_cols}")

def log_missing_parquet(parquet_path):
    with log_lock:
        parquet_logger.info(f'Parquet file not found: {parquet_path}')

def process_flight(flight_id, flights_dir):
    global fuel_burn_global
    results = []
    segments = fuel_burn_global[fuel_burn_global['flight_id'] == flight_id]
    parquet_path = f'{flights_dir}/{flight_id}.parquet'
    if not os.path.exists(parquet_path):
        log_missing_parquet(parquet_path)
        return results
    try:
        pos_df = pd.read_parquet(parquet_path)
    except Exception as e:
        log_missing_parquet(parquet_path)
        return results

    required_columns = ['timestamp', 'source', 'latitude', 'longitude', 'altitude']
    missing_cols = [col for col in required_columns if col not in pos_df.columns]
    if missing_cols:
        log_missing_columns(flight_id, missing_cols)
        return results

    for _, seg in segments.iterrows():
        start, end = seg['start'], seg['end']
        filtered = pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] <= end)]
        if filtered.empty:
            log_missing_timerange(flight_id, seg['idx'], start, end)
            continue
        for source, group in filtered.groupby('source'):
            stats = {
                'flight_id': flight_id,
                'segment_idx': seg['idx'],
                'source': source,
                'start': start,
                'end': end,
                'latitude_min': group['latitude'].min() if not group['latitude'].empty else np.nan,
                'latitude_max': group['latitude'].max() if not group['latitude'].empty else np.nan,
                'longitude_min': group['longitude'].min() if not group['longitude'].empty else np.nan,
                'longitude_max': group['longitude'].max() if not group['longitude'].empty else np.nan,
                'altitude_min': group['altitude'].min() if not group['altitude'].empty else np.nan,
                'altitude_max': group['altitude'].max() if not group['altitude'].empty else np.nan,
            }
            results.append(stats)
    return results

def run_multiprocessor(fuel_burn, flights_dir='dataset/flights_train', num_workers=10):
    unique_flight_ids = fuel_burn['flight_id'].unique()
    with Pool(num_workers, initializer=init_fuel_burn, initargs=(fuel_burn,)) as pool:
        results = pool.starmap(process_flight, [(flight_id, flights_dir) for flight_id in unique_flight_ids])
    prep_df = pd.DataFrame([item for sublist in results for item in sublist])
    return prep_df