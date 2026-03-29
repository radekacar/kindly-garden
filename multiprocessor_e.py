import pandas as pd
import numpy as np
import os
import xarray as xr
from openap import drag, thrust, FuelFlow
from pitot import isa
from scipy.optimize import minimize_scalar
from typing import Tuple, Union, Optional
from numpy.typing import ArrayLike
from tqdm.notebook import tqdm
from multiprocessing import Pool

KTS_TO_MS = 0.514444            # 1 knot = 0.514444 m/s
MS_TO_KTS = 1.0 / KTS_TO_MS     # inverse conversion

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

def fill_tas_column_simplified(group_sorted, grib_df):
    """
    Fill the TAS (True Airspeed) column for a sorted group DataFrame.
    Calculates TAS from groundspeed, track, and wind (u, v) only.
    """
    required = ['TAS', 'groundspeed', 'track', 'latitude', 'longitude', 'altitude', 'timestamp']
    for col in required:
        if col not in group_sorted.columns:
            group_sorted[col] = np.nan
    tas_vals = group_sorted['TAS'].copy()
    needs_weather = tas_vals.isna() & (
        group_sorted['groundspeed'].notna() & group_sorted['track'].notna()
    )
    if needs_weather.any():
        weather_rows = group_sorted[needs_weather]
        for idx, row in weather_rows.iterrows():
            u, v, _ = get_profile_at_alt(grib_df, row['latitude'], row['longitude'],
                                         row['altitude'], row['timestamp'])
            if pd.notna(u) and pd.notna(v) and pd.notna(row['groundspeed']) and pd.notna(row['track']):
                tas_vals[idx] = calculate_tas(row['groundspeed'], row['track'], u, v)
    group_sorted['TAS'] = tas_vals
    return group_sorted

def compute_c0_c1(ac: str, tas: ArrayLike, alt_ft: ArrayLike, mass2: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    dmodel = drag.Drag(ac)
    tas = np.asarray(tas)
    alt_ft = np.asarray(alt_ft)
    c0 = np.empty_like(tas, dtype=float)
    c1 = np.empty_like(tas, dtype=float)
    # The drag.clean() function estimates the drag force when the aircraft is at the clean configuration, which means no flaps or landing gear are deployed.
    for idx, _ in np.ndenumerate(tas):
        t = tas[idx]
        a = alt_ft[idx]
        D0 = float(dmodel.clean(alt=a, tas=t, mass=0))
        Dm = float(dmodel.clean(alt=a, tas=t, mass=mass2))
        c0[idx] = D0
        c1[idx] = (Dm - D0) / (mass2 ** 2)
    return c0, c1

def compute_thrust(
    phase_flag: Union[str, ArrayLike],
    ac: str,
    tas: ArrayLike,
    alt_ft: ArrayLike,
    roc: Optional[ArrayLike] = None
) -> np.ndarray:
    tmodel = thrust.Thrust(ac)
    tas = np.asarray(tas, dtype=float)
    alt_ft = np.asarray(alt_ft, dtype=float)
    if tas.shape != alt_ft.shape:
        raise ValueError("tas and alt_ft must have the same shape")
    n = tas.size
    # Prepare roc array
    if roc is None:
        roc_arr = np.zeros(n, dtype=float)
    else:
        roc_arr = np.asarray(roc, dtype=float)
        if roc_arr.shape != tas.shape:
            raise ValueError("roc must be None or have same shape as tas/alt_ft")
    # Normalize phase_flag to array of strings
    phase_arr = np.asarray(phase_flag, dtype=object)
    if phase_arr.ndim == 0:
        phase_arr = np.full(n, str(phase_arr).upper(), dtype=object)
    else:
        if phase_arr.shape != tas.shape:
            # allow 1D array matching length
            if phase_arr.size == n:
                phase_arr = phase_arr.astype(object)
            else:
                raise ValueError("phase_flag must be scalar or array-like with same length as tas")
    # Validate and uppercase values
    def _norm(ph):
        if ph is None:
            raise ValueError("phase_flag contains None")
        s = str(ph).strip().upper()
        if s not in ("CL", "DE", "CR"):
            raise ValueError(f"Unsupported phase flag: {ph!r}. Allowed: 'CL','DE','CR'")
        return s
    phase_arr = np.vectorize(_norm)(phase_arr)
    thr = np.empty(n, dtype=float)
    for i in range(n):
        ph = phase_arr[i]
        if ph == "CL":
            thr[i] = tmodel.climb(tas=tas[i], alt=alt_ft[i], roc=roc_arr[i])
        elif ph == "DE":
            thr[i] = tmodel.descend(tas=tas[i], alt=alt_ft[i], roc=roc_arr[i])
        else:  # "CR"
            thr[i] = tmodel.cruise(tas=tas[i], alt=alt_ft[i])

    return thr

def energy_rate(alt_ft: ArrayLike, tas_kts: ArrayLike, timestamp: ArrayLike) -> np.ndarray:
    alt_ft = np.asarray(alt_ft, dtype=float)
    tas_kts = np.asarray(tas_kts, dtype=float)
    timestamp = np.asarray(timestamp)
    # Unit conversions
    alt_m = alt_ft * 0.3048           # feet -> meters
    tas_ms = tas_kts * 0.514444       # knots -> m/s
    g_0 = 9.80665   # Standard gravity in m/s^2
    # Use ISA temperature at altitude in meters (if needed)
    tempISA = isa.temperature(alt_m)
    # tau = temperature / tempISA (not available here) -> assume 1
    tau = 1.0
    # Differences
    dalt = np.diff(alt_m, prepend=alt_m[0])            # meters
    tas2 = tas_ms ** 2                                 # (m/s)^2
    dtas2 = np.diff(tas2, prepend=tas2[0])             # (m/s)^2 differences
    # Replace 'datetime64[s]' with 'datetime64[ms]' for higher resolution, to avoid zero dt
    dt = np.diff(timestamp.astype('datetime64[ms]').astype(float), prepend=timestamp[0].astype('datetime64[ms]').astype(float))
    # Avoid division by zero
    dt[dt == 0] = np.nan
    # energy rate per unit mass: (g0 * dalt * tau + 0.5 * d(tas^2)) / dt
    e = (g_0 * dalt * tau + 0.5 * dtas2) / dt

    return e

def estimate_mass_least_squares(a, b, c, mass_bounds=(1000.0, 200000.0), tol=1e-3):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    if not np.any(mask):
        return np.nan
    a, b, c = a[mask], b[mask], c[mask]
    if a.size == 0:
        return np.nan
    def obj(m):
        r = a * m**2 + b * m + c
        return np.sum(r**2)
    res = minimize_scalar(obj, bounds=mass_bounds, method='bounded', options={'xatol': tol})
    return float(res.x) if res.success else np.nan

def estimate_mass_from_energy_balance(trajectory_df, ac: str, is_climb: bool = True) -> np.ndarray:
    # Validate input type and required columns
    if trajectory_df is None:
        raise ValueError("`trajectory_df` must be provided")
    df = pd.DataFrame(trajectory_df).copy()
    required = {'TAS', 'altitude', 'vertical_rate', 'timestamp'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trajectory_df: {sorted(missing)}")
    # Extract arrays and normalize types
    tas = np.asarray(df['TAS'].values, dtype=float)  # knots
    alt_ft = np.asarray(df['altitude'].values, dtype=float)  # feet
    roc = np.asarray(df['vertical_rate'].values, dtype=float)  # ft/min
    timestamp = np.asarray(pd.to_datetime(df['timestamp'], utc=True).values)  # datetime64[ns, UTC]
    # Compute model components (these helper functions are assumed available in module)
    c0, c1 = compute_c0_c1(ac, tas, alt_ft)
    phase_flag = 'CL' if is_climb else 'DE'
    thr = compute_thrust(phase_flag, ac, tas, alt_ft, roc)  # N
    e_rate = energy_rate(alt_ft, tas, timestamp)  # W/kg (J/s per kg)
    # Convert TAS to m/s for coefficient `a`
    KTS2MS = 0.514444
    tas_ms = tas * KTS2MS
    # Build quadratic coefficients and ensure broadcasting to same shape
    a = -np.asarray(c1, dtype=float) * np.asarray(tas_ms, dtype=float)
    b = -np.asarray(e_rate, dtype=float)
    c = np.asarray(thr, dtype=float) - np.asarray(c0, dtype=float)
    # Broadcast to common shape
    a_b, b_b, c_b = np.broadcast_arrays(a, b, c)
    # Solve quadratic per-sample using existing solver
    mass = estimate_mass_least_squares(a=a_b, b=b_b, c=c_b)  # expected output in kg or np.nan
    # Ensure result is 1-D with same length as input rows
    mass = np.asarray(mass).reshape(-1)[:len(df)]

    return mass

def calculate_fuel_flow(phase_df, phase_flag, ac_type):
    ff = FuelFlow(ac_type)
    tas_kts = phase_df['TAS']
    roc_ft_min = phase_df['vertical_rate']
    alt_ft = phase_df['altitude']
    thrust_values = compute_thrust(phase_flag, ac_type, tas_kts, alt_ft, roc_ft_min)
    result = []

    return ff.at_thrust(thrust_values)

def process_row(row, resample_n=100, era5_dir='dataset/ERA5'):
    """
    Processes a single row (DataFrame) by resampling if needed.
    Assumes 'timestamp' column exists for sorting and sampling.
    """
    result = row.copy()
    estimated_mass_climb = np.nan
    estimated_mass_descend = np.nan

    phase_info = [
        ('climb_phase_path', 'CL', 'estimated_mass_climb'),
        ('descend_phase_path', 'DE', 'estimated_mass_descend'),
        ('cruise_phase_path', 'CR', None)
    ]

    for phase_path_key, phase_flag, mass_col in phase_info:
        phase_path = row.get(phase_path_key, None)
        if pd.notna(phase_path) and os.path.exists(phase_path):
            try:
                phase_df = pd.read_parquet(phase_path)
                if len(phase_df) > resample_n:
                    phase_df = phase_df.sort_values('timestamp')
                    idx = np.linspace(0, len(phase_df) - 1, resample_n).astype(int)
                    phase_df = phase_df.iloc[idx].reset_index(drop=True)
                flight_id = phase_df['flight_id'].iloc[0]
                grib_file_path = f'{era5_dir}/CDS_data_{flight_id}.grib'
                if os.path.exists(grib_file_path):
                    grib_file = xr.open_dataset(grib_file_path, engine='cfgrib')
                    phase_df = fill_tas_column_simplified(phase_df.sort_values('timestamp').reset_index(drop=True), grib_file)
                    phase_df = phase_df.dropna(subset=['TAS', 'altitude', 'vertical_rate', 'timestamp'])
                    phase_df = phase_df.drop_duplicates(subset=['timestamp'])
                    ac = phase_df['typecode'].iloc[0].lower()
                    if phase_flag in ['CL', 'DE']:
                        est_mass = estimate_mass_from_energy_balance(phase_df, ac, phase_flag == 'CL')
                        result[mass_col] = est_mass
                    fuel_flow_values = calculate_fuel_flow(phase_df, phase_flag, ac)
                    result[f'fuel_flow_{phase_flag}'] = np.nanmean(fuel_flow_values)
            except Exception:
                result[f'fuel_flow_{phase_flag}'] = np.nan
                if mass_col:
                    result[mass_col] = np.nan

    # Remove unwanted columns
    for col in ['has_climb_phase', 'has_descend_phase', 'has_cruise_phase', 'climb_phase_path', 'descend_phase_path', 'cruise_phase_path']:
        result.pop(col, None)

    return result

def _process_row_wrapper(args):
    row, resample_n, era5_dir = args
    return process_row(row, resample_n, era5_dir)

def run_mass_estimation(united_with_phases, num_workers=10, resample_n=100, era5_dir='dataset/ERA5'):
    """
    Processes all phase DataFrames in parallel, with optional resampling.
    Shows progress bar in Jupyter Notebook.
    """
    rows = united_with_phases.to_dict('records')
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_row_wrapper, [(row, resample_n, era5_dir) for row in rows]),
            total=len(rows),
            desc="Processing files"
        ))
    df = pd.DataFrame(results)
    cols = [c for c in df.columns if c != 'fuel_kg'] + ['fuel_kg'] if 'fuel_kg' in df.columns else df.columns
    return df[cols]
