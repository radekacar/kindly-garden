import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from openap.phase import FlightPhase
from functools import partial

_united_global = None

def init_united(united_df):
    global _united_global
    _united_global = united_df

def get_flight_phase(trajectory_df):
    ts = (trajectory_df['timestamp'] - trajectory_df['timestamp'].iloc[0]).dt.total_seconds()
    alt = trajectory_df['altitude'].values
    spd = trajectory_df['groundspeed'].values
    roc = trajectory_df['vertical_rate'].values

    fp = FlightPhase()
    fp.set_trajectory(ts, alt, spd, roc)
    labels = fp.phaselabel()
    return labels

def extract_phases(trajectory_df, min_length, min_period):
    # Always return three DataFrames, even if input is empty
    if trajectory_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    trajectory_df = trajectory_df.reset_index(drop=True)
    labels = get_flight_phase(trajectory_df)
    trajectory_df = trajectory_df.copy()
    trajectory_df['phase'] = labels

    result = {}
    for phase_label in ['CL', 'DE', 'CR']:
        phase_df = trajectory_df[trajectory_df['phase'] == phase_label]
        if len(phase_df) < min_length:
            result[phase_label] = pd.DataFrame()
            continue

        timestamps = phase_df['timestamp'].values
        time_diffs = np.insert(np.diff(timestamps) / np.timedelta64(1, 's'), 0, min_period)
        keep_mask = np.cumsum(time_diffs) >= np.arange(len(time_diffs)) * min_period
        filtered_phase = phase_df[keep_mask].reset_index(drop=True)

        if len(filtered_phase) < min_length:
            result[phase_label] = pd.DataFrame()
        else:
            result[phase_label] = filtered_phase

    # Always return a tuple of three DataFrames
    return result.get('CL', pd.DataFrame()), result.get('DE', pd.DataFrame()), result.get('CR', pd.DataFrame())

def process_flight(flight_id, min_length, min_period, flights_dir, phases_dir):
    global _united_global
    rows = []
    segments = _united_global[_united_global['flight_id'] == flight_id]
    parquet_path = f'{flights_dir}/{flight_id}.parquet'
    if not os.path.exists(parquet_path):
        return rows
    try:
        pos_df = pd.read_parquet(parquet_path)
    except Exception:
        return rows
    for idx, seg in segments.iterrows():
        start, end = seg['start'], seg['end']
        filtered = pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] <= end)]
        climb_phase_df, descend_phase_df, cruise_phase_df = extract_phases(filtered, min_length, min_period)
        result_row = seg.to_dict()
        result_row['has_climb_phase'] = not climb_phase_df.empty
        result_row['has_descend_phase'] = not descend_phase_df.empty
        result_row['has_cruise_phase'] = not cruise_phase_df.empty

        # Save phase DataFrames to disk and store file paths
        os.makedirs(phases_dir, exist_ok=True)
        climb_path = ''
        descend_path = ''
        cruise_path = ''
        if not climb_phase_df.empty:
            climb_path = os.path.join(phases_dir, f'{flight_id}_{idx}_climb.parquet')
            climb_phase_df.to_parquet(climb_path)
        if not descend_phase_df.empty:
            descend_path = os.path.join(phases_dir, f'{flight_id}_{idx}_descend.parquet')
            descend_phase_df.to_parquet(descend_path)
        if not cruise_phase_df.empty:
            cruise_path = os.path.join(phases_dir, f'{flight_id}_{idx}_cruise.parquet')
            cruise_phase_df.to_parquet(cruise_path)
        result_row['climb_phase_path'] = climb_path
        result_row['descend_phase_path'] = descend_path
        result_row['cruise_phase_path'] = cruise_path

        rows.append(result_row)
    return rows

def run_multiprocess_phases(united_df, num_workers=10, min_length=10, min_period=5, flights_dir='dataset/flights_train',
                            phases_dir='dataset/phases'):
    unique_ids = united_df['flight_id'].unique()
    with Pool(num_workers, initializer=init_united, initargs=(united_df,)) as pool:
        results = pool.map(
            partial(
                process_flight,
                min_length=min_length,
                min_period=min_period,
                flights_dir=flights_dir,
                phases_dir=phases_dir
            ),
            unique_ids
        )
    flat = [item for sublist in results for item in sublist]
    return pd.DataFrame(flat)