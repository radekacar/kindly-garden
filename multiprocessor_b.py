import pandas as pd
import numpy as np
import os
import math
import xarray as xr
from typing import Union
from typing import Sequence
from multiprocessing import Pool
from functools import partial

# Constants
GAMMA = 1.4                     # ratio of specific heats for air
R_AIR = 287.058                 # J/(kg*K)
KTS_TO_MS = 0.514444            # 1 knot = 0.514444 m/s
MS_TO_KTS = 1.0 / KTS_TO_MS     # inverse conversion

fuel_burn_global = None
flight_dir_global = None
era5_dir_global = None

def init_fuel_burn(fuel_burn, flight_dir, era5_dir):
    global fuel_burn_global, flight_dir_global, era5_dir_global
    fuel_burn_global = fuel_burn
    flight_dir_global = flight_dir
    era5_dir_global = era5_dir


def get_profile_at_alt(ds, lat, lon, alt_ft, timestamp):
    G0 = 9.80665  # Standard gravity in m/s^2
    alt_m = alt_ft * 0.3048  # Convert feet to meters
    # Ensure timestamp is pandas.Timestamp and UTC
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.to_datetime(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize('UTC')
    # Find the closest time index
    if 'time' in ds.dims:
        times = pd.to_datetime(ds['time'].values)
        if times.tz is None:
            times = times.tz_localize('UTC')
        idx = np.argmin(np.abs(times - timestamp))
        profile = ds.isel(time=idx)
    else:
        profile = ds
    # Select the nearest grid point for the given lat/lon
    profile = profile.sel(latitude=lat, longitude=lon, method='nearest')
    # Calculate geometric height from geopotential
    height_profile = (profile['z'] / G0).values
    u_profile = profile['u'].values
    v_profile = profile['v'].values
    temp_profile = profile['t'].values
    # Sort by height for interpolation
    sort_idx = np.argsort(height_profile)
    height_sorted = height_profile[sort_idx]
    u_sorted = u_profile[sort_idx]
    v_sorted = v_profile[sort_idx]
    temp_sorted = temp_profile[sort_idx]
    try:
        u_interp = np.interp(alt_m, height_sorted, u_sorted)
        v_interp = np.interp(alt_m, height_sorted, v_sorted)
        temp_interp = np.interp(alt_m, height_sorted, temp_sorted)
        return u_interp, v_interp, temp_interp
    except Exception:
        return np.nan, np.nan, np.nan


def calculate_tas(groundspeed_kts, track_deg, u_wind_ms, v_wind_ms):
    # --- Unit Conversions ---
    # Use module-level constants KTS_TO_MS and MS_TO_KTS
    # Convert groundspeed to m/s
    gs_ms = groundspeed_kts * KTS_TO_MS
    # Convert track angle to radians (math convention: 0=East)
    # Meteorological convention (0=North) to math convention: angle_math = 90 - angle_met
    track_rad = np.deg2rad(90 - track_deg)
    # --- Decompose Groundspeed into u and v components ---
    # u_gs is the Eastward component, v_gs is the Northward component
    u_gs = gs_ms * np.cos(track_rad)
    v_gs = gs_ms * np.sin(track_rad)
    # --- Calculate Airspeed Components (vector subtraction) ---
    # Airspeed = Groundspeed - Windspeed
    u_tas = u_gs - u_wind_ms
    v_tas = v_gs - v_wind_ms
    # --- Calculate TAS magnitude and convert to knots ---
    tas_ms = np.sqrt(u_tas ** 2 + v_tas ** 2)
    tas_kts = tas_ms * MS_TO_KTS

    return tas_kts


def calculate_groundspeed(tas_kts, track_deg, u_wind_ms, v_wind_ms):
    # quick NaN checks
    if np.any([tas_kts is None, track_deg is None, u_wind_ms is None, v_wind_ms is None]):
        return np.nan
    try:
        tas_kts = float(tas_kts)
        track_deg = float(track_deg)
        u_wind_ms = float(u_wind_ms)
        v_wind_ms = float(v_wind_ms)
    except Exception:
        return np.nan
    # Convert TAS to m/s
    tas_ms = tas_kts * KTS_TO_MS
    # Convert track from meteorological (0 = North) to math angle used for components
    # math angle where 0 = East and positive counter-clockwise: angle_math = 90 - track_deg
    track_rad = np.deg2rad(90.0 - track_deg)
    # Air-relative components (eastward, northward)
    u_tas = tas_ms * np.cos(track_rad)
    v_tas = tas_ms * np.sin(track_rad)
    # Groundspeed components = air-relative + wind
    u_gs = u_tas + u_wind_ms
    v_gs = v_tas + v_wind_ms
    # Resulting groundspeed magnitude (m/s) -> convert to knots
    gs_ms = np.hypot(u_gs, v_gs)
    gs_kts = gs_ms / KTS_TO_MS

    return float(gs_kts)


def isa_temperature(alt_m: float) -> float:
    T0 = 288.15
    L = 0.0065
    return T0 - L * alt_m if alt_m <= 11000 else 216.65

def speed_of_sound(temp_k: float) -> float:
    return math.sqrt(GAMMA * R_AIR * temp_k)


def tas_from_mach(mach: float,
                  out_unit: str = "m/s",
                  temp_k: float = None,
                  temp_c: float = None,
                  alt_ft: float = None) -> float:
    if temp_k is None:
        if temp_c is not None:
            temp_k = float(temp_c) + 273.15
        elif alt_ft is not None:
            alt_m = float(alt_ft) * 0.3048
            temp_k = isa_temperature(alt_m)
        else:
            raise ValueError("Provide temp_k (K) or temp_c (C) or alt_ft (ft) to compute TAS.")

    tas_ms = float(mach) * speed_of_sound(float(temp_k))
    out = out_unit.lower()
    if out in ("kt", "kts", "knot", "knots"):
        return tas_ms * MS_TO_KTS
    if out in ("m/s", "ms"):
        return tas_ms
    raise ValueError("Unsupported output unit: choose 'kts' or 'm/s'.")


def fill_tas_column(group_sorted, grib_df):
    required = ['TAS', 'CAS', 'mach', 'track', 'groundspeed', 'latitude', 'longitude', 'altitude', 'timestamp']
    for col in required:
        if col not in group_sorted.columns:
            group_sorted[col] = np.nan
    # Copy existing TAS values
    tas_vals = group_sorted['TAS'].copy()
    # Fill from CAS where TAS is missing
    cas_mask = tas_vals.isna() & group_sorted['CAS'].notna()
    tas_vals[cas_mask] = group_sorted.loc[cas_mask, 'CAS']
    # Batch process rows needing weather data
    needs_weather = tas_vals.isna() & (
        group_sorted['mach'].notna() |
        (group_sorted['groundspeed'].notna() & group_sorted['track'].notna())
    )
    if needs_weather.any():
        weather_rows = group_sorted[needs_weather]
        for idx, row in weather_rows.iterrows():
            u, v, temp_k = get_profile_at_alt(grib_df, row['latitude'], row['longitude'],
                                               row['altitude'], row['timestamp'])
            # Try Mach first
            if pd.notna(row['mach']) and pd.notna(temp_k):
                tas_vals[idx] = tas_from_mach(row['mach'], out_unit='kts', temp_k=temp_k)
            # Then try groundspeed calculation
            elif pd.notna(u) and pd.notna(v) and pd.notna(row['groundspeed']) and pd.notna(row['track']):
                tas_vals[idx] = calculate_tas(row['groundspeed'], row['track'], u, v)
    group_sorted['TAS'] = tas_vals

    return group_sorted

def fill_groundspeed_column(group_sorted, grib_df):
    # Copy existing groundspeed values
    gs_vals = group_sorted['groundspeed'].copy()
    # Only process rows where groundspeed is missing but TAS and track exist
    needs_calc = gs_vals.isna() & group_sorted['TAS'].notna() & group_sorted['track'].notna()
    if needs_calc.any():
        calc_rows = group_sorted[needs_calc]
        for idx, row in calc_rows.iterrows():
            u, v, _ = get_profile_at_alt(grib_df, row['latitude'], row['longitude'],
                                          row['altitude'], row['timestamp'])
            if pd.notna(u) and pd.notna(v):
                gs_vals[idx] = calculate_groundspeed(row['TAS'], row['track'], u, v)
    group_sorted['groundspeed'] = gs_vals

    return group_sorted

def haversine(
        lat1: Union[float, np.ndarray],
        lon1: Union[float, np.ndarray],
        lat2: Union[float, np.ndarray],
        lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    # Mean Earth radius in meters (WGS84 mean)
    R: float = 6_371_000.0
    # Convert inputs to numpy arrays for vectorized operations
    lat1_a = np.asarray(lat1, dtype=float)
    lon1_a = np.asarray(lon1, dtype=float)
    lat2_a = np.asarray(lat2, dtype=float)
    lon2_a = np.asarray(lon2, dtype=float)
    # Convert degrees to radians
    phi1: np.ndarray = np.radians(lat1_a)  # latitude1 in radians
    phi2: np.ndarray = np.radians(lat2_a)  # latitude2 in radians
    # Differences in radians
    dphi: np.ndarray = np.radians(lat2_a - lat1_a)  # delta latitude
    dlambda: np.ndarray = np.radians(lon2_a - lon1_a)  # delta longitude
    # Normalize longitude difference to range [-pi, +pi] to handle dateline crossing
    dlambda = (dlambda + np.pi) % (2 * np.pi) - np.pi
    # Haversine formula
    sin_dphi_2 = np.sin(dphi / 2.0)
    sin_dlambda_2 = np.sin(dlambda / 2.0)
    a = sin_dphi_2 ** 2 + np.cos(phi1) * np.cos(phi2) * sin_dlambda_2 ** 2
    # Numerical safety: clamp a to [0, 1]
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist = R * c  # distance in meters
    # If all inputs were scalars, return a Python float for convenience
    if dist.shape == ():  # 0-d array (scalar)
        return float(dist)

    return dist

def calculate_3d_distance(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    altitudes: Sequence[float],
    alt_unit: str = 'ft'
) -> float:
    # Convert inputs to numpy arrays of floats for vectorized arithmetic
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    alt = np.asarray(altitudes, dtype=float)
    # Validate lengths
    if lat.size != lon.size or lat.size != alt.size:
        raise ValueError("latitudes, longitudes and altitudes must have the same length")
    # If less than 2 points, distance is zero
    if lat.size < 2:
        return 0.0
    # Convert altitude to meters if provided in feet
    if alt_unit.lower() == 'ft':
        alt_m = alt * 0.3048  # 1 foot = 0.3048 meters
    elif alt_unit.lower() == 'm':
        alt_m = alt
    else:
        raise ValueError("alt_unit must be either 'ft' or 'm'")
    # Compute horizontal distances between consecutive points (meters)
    # haversine supports vectorized inputs, so provide consecutive pairs
    horiz = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])  # array of length N-1
    # Compute vertical differences between consecutive points (meters)
    vert = alt_m[1:] - alt_m[:-1]
    # Compute 3D segment distances and sum
    segments = np.sqrt(horiz**2 + vert**2)
    total_distance_m = float(np.sum(segments))

    return total_distance_m

def process_flight(flight_id, FAST):
    global fuel_burn_global, flight_dir_global, era5_dir_global
    results = []
    segments = fuel_burn_global[fuel_burn_global['flight_id'] == flight_id]
    parquet_path = f'{flight_dir_global}/{flight_id}.parquet'
    grib_path = f'{era5_dir_global}/CDS_data_{flight_id}.grib'
    if not os.path.exists(parquet_path) or not os.path.exists(grib_path):
        return results
    try:
        pos_df = pd.read_parquet(parquet_path)
        grib_df = xr.open_dataset(grib_path, engine='cfgrib')
    except Exception:
        return results

    for _, seg in segments.iterrows():
        start, end = seg['start'], seg['end']
        mask = (pos_df['timestamp'] >= start) & (pos_df['timestamp'] <= end)
        filtered = pos_df[mask]
        total_count = len(filtered)
        acars_count = (filtered['source'] == 'acars').sum()
        adsb_count = (filtered['source'] == 'adsb').sum()

        for source, group in filtered.groupby('source'):

            group_sorted = group.sort_values('timestamp')

            if FAST:
                # Calculate mean values for the group
                lat_mean = group_sorted['latitude'].mean(skipna=True)
                lon_mean = group_sorted['longitude'].mean(skipna=True)
                alt_mean = group_sorted['altitude'].mean(skipna=True)
                time_mean = pd.to_datetime(group_sorted['timestamp']).mean()
                track_mean = group_sorted['track'].mean(skipna=True)
                gs_mean = group_sorted['groundspeed'].mean(skipna=True)
                tas_mean = group_sorted['TAS'].mean(skipna=True)
                cas_mean = group_sorted['CAS'].mean(skipna=True)
                mach_mean = group_sorted['mach'].mean(skipna=True)

                # Fill TAS using fallback logic
                final_tas = np.nan
                if not np.isnan(tas_mean):
                    final_tas = tas_mean
                if np.isnan(final_tas) and not np.isnan(cas_mean):
                    final_tas = cas_mean
                if np.isnan(final_tas) and not np.isnan(mach_mean):
                    u, v, temp_k = get_profile_at_alt(grib_df, lat_mean, lon_mean, alt_mean, time_mean)
                    if not np.isnan(temp_k):
                        final_tas = tas_from_mach(mach_mean, out_unit='kts', temp_k=temp_k)
                if np.isnan(final_tas) and not np.isnan(gs_mean) and not np.isnan(track_mean):
                    u, v, temp_k = get_profile_at_alt(grib_df, lat_mean, lon_mean, alt_mean, time_mean)
                    if not np.isnan(u) and not np.isnan(v):
                        final_tas = calculate_tas(gs_mean, track_mean, u, v)

                # Fill groundspeed using fallback logic
                final_gs = np.nan
                if not np.isnan(gs_mean):
                    final_gs = gs_mean
                if np.isnan(final_gs) and not np.isnan(final_tas) and not np.isnan(track_mean):
                    u, v, temp_k = get_profile_at_alt(grib_df, lat_mean, lon_mean, alt_mean, time_mean)
                    if not np.isnan(u) and not np.isnan(v):
                        final_gs = calculate_groundspeed(final_tas, track_mean, u, v)

                stats = {
                        'flight_id': flight_id,
                        'segment_idx': seg['idx'],
                        'source': source,
                        'typecode': group_sorted['typecode'].iloc[0] if 'typecode' in group_sorted.columns else None,
                        'start': start,
                        'end': end,
                        'since_takeoff': seg['since_takeoff'],
                        'until_landed': seg['until_landed'],
                        'segment_duration': seg['segment_duration'],
                        'acars_count': acars_count,
                        'adsb_count': adsb_count,
                        'source_percentage': len(group_sorted) / total_count * 100 if total_count > 0 else 0,
                        'altitude_mean': alt_mean,
                        'altitude_std': group_sorted['altitude'].std(skipna=True),
                        'groundspeed_mean': final_gs,
                        'groundspeed_std': group_sorted['groundspeed'].std(skipna=True),
                        'vertical_rate_mean': group_sorted['vertical_rate'].mean(skipna=True),
                        'vertical_rate_std': group_sorted['vertical_rate'].std(skipna=True),
                        'TAS_mean': final_tas,
                        'TAS_std': group_sorted['TAS'].std(skipna=True),
                        'latitude_mean': lat_mean,
                        'longitude_mean': lon_mean,
                        'haversine_distance': haversine(
                            group_sorted['latitude'].iloc[0], group_sorted['longitude'].iloc[0],
                            group_sorted['latitude'].iloc[-1], group_sorted['longitude'].iloc[-1]
                        ) if len(group_sorted) >= 2 else np.nan,
                        '3d_distance': calculate_3d_distance(
                            group_sorted['latitude'], group_sorted['longitude'], group_sorted['altitude']
                        ) if len(group_sorted) >= 2 else np.nan,
                        'fuel_kg': seg['fuel_kg'],
                    }

                results.append(stats)

            else:  # FAST == False slow calculation
                # Slow: calculate TAS and groundspeed for each row
                group_sorted = fill_tas_column(group_sorted, grib_df)
                group_sorted = fill_groundspeed_column(group_sorted, grib_df)

                stats = {
                    'flight_id': flight_id,
                    'segment_idx': seg['idx'],
                    'source': source,
                    'typecode': group_sorted['typecode'].iloc[0],
                    'start': start,
                    'end': end,
                    'since_takeoff': seg['since_takeoff'],
                    'until_landed': seg['until_landed'],
                    'segment_duration': seg['segment_duration'],
                    'acars_count': acars_count,
                    'adsb_count': adsb_count,
                    'source_percentage': len(group_sorted) / total_count * 100 if total_count > 0 else 0,
                    'altitude_mean': group_sorted['altitude'].mean(),
                    'altitude_std': group_sorted['altitude'].std(),
                    'groundspeed_mean': group_sorted['groundspeed'].mean(),
                    'groundspeed_std': group_sorted['groundspeed'].std(),
                    'vertical_rate_mean': group_sorted['vertical_rate'].mean(),
                    'vertical_rate_std': group_sorted['vertical_rate'].std(),
                    'TAS_mean': group_sorted['TAS'].mean(),
                    'TAS_std': group_sorted['TAS'].std(),
                    'latitude_mean': group_sorted['latitude'].mean(),
                    'longitude_mean': group_sorted['longitude'].mean(),
                }

                if len(group_sorted) >= 2:
                    stats['haversine_distance'] = haversine(
                        group_sorted['latitude'].iloc[0], group_sorted['longitude'].iloc[0],
                        group_sorted['latitude'].iloc[-1], group_sorted['longitude'].iloc[-1]
                    )
                else:
                    stats['haversine_distance'] = np.nan

                if len(group_sorted) >= 2:
                    stats['3d_distance'] = calculate_3d_distance(
                        group_sorted['latitude'], group_sorted['longitude'], group_sorted['altitude'], alt_unit='ft'
                    )
                else:
                    stats['3d_distance'] = np.nan

                stats['fuel_kg'] = seg['fuel_kg']

                results.append(stats)

    return results

def run_multiprocessor(fuel_burn, num_workers=10, FAST=True, flight_dir='dataset/flights_train', era5_dir='dataset/ERA5'):
    unique_flight_ids = fuel_burn['flight_id'].unique()
    with Pool(num_workers, initializer=init_fuel_burn, initargs=(fuel_burn, flight_dir, era5_dir)) as pool:
        process_func = partial(process_flight, FAST=FAST)
        results = pool.map(process_func, unique_flight_ids)
    stats_df = pd.DataFrame([item for sublist in results for item in sublist])
    return stats_df