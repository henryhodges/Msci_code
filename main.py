"""
main.py
==========
Main analysis pipeline for the Hénon DRO spacecraft crossover study.

Identifies periods when STEREO-A, STEREO-B, and Solar Orbiter cross a
modelled  Distant Retrograde Orbit (DRO) constellation in the Sun-Earth 
CR3BP, and lie near the Sun-Earth line. For each crossover event, in-situ 
magnetic field and plasma data are retrieved and compared with 
time-propagated L1 (OMNI) observations using ballistic propagation.

Orbit model : orbit.py (CR3BP numerical integration)
Data sources : JPL Horizons (ephemerides), pyspedas (CDF), OMNI HRO-1min
Indices      : Richardson & Cane ICME catalogue, Dst (OMNI2), Kp (OMNI2)

Configuration
-------------
    All run parameters are set in the Constants & Toggles section below.
    Set SAVE_PLOTS_TO_FILE = True to save figures to FIGURES_ROOT.

Author : Henry Hodges
"""

# ---------------------------------- imports --------------------------------- #
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob
import signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter, AutoDateLocator
import numpy as np
import pandas as pd
import pyspedas
import cdflib
from scipy.interpolate import interp1d
from tqdm import tqdm
import requests
from io import StringIO

from orbit import DROConstellation, PhysicalConstants
_const = PhysicalConstants()

try:
    from sunpy.coordinates import get_horizons_coord
    from astropy.time import Time
    import astropy.units as u
    SUNPY_AVAILABLE = True
except ImportError:
    SUNPY_AVAILABLE = False
    print("[WARN] SunPy not available; JPL Horizons ephemerides cannot be loaded.")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspedas")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")


# -------------------------------- File Paths -------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", str(PROJECT_ROOT / "figures")))

WRITE_EVENTS_CSV = False
CSV_FILENAME = "henon_dro_crossover_events.csv"


# ---------------------------- Constants & Toggles --------------------------- #

# Constants
AU_KM = _const.AU_KM()
YEAR_DAYS = _const.SIDEREAL_YEAR_DAYS
MEAN_MOTION = 2 * np.pi / YEAR_DAYS

EARTH_EPHEMERIS_MARGIN_DAYS = 60 

# Date range for spacecraft scan
START_DATE = datetime(2007, 6, 1) 
END_DATE = datetime(2008, 2, 20) 

# Tolerances from DRO for event detection
SUN_EARTH_TOL_AU= 0.3
DRO_TOL_AU = 0.5

# Time windows for analysis
DT_HOURS = 6
INSITU_WINDOW_HR = 144 
MAG_TARGET_POINTS_PER_DAY = 1440


# Event limits per spacecraft (to avoid too many plots, set to 'None' for max datapoints)
EVENT_LIMIT = {
    "STEREO-A": 3,
    "Solar Orbiter": 0,
    "Parker Solar Probe": 0,
    "STEREO-B": 0,
}

# To highlight events between Earth-Sun
FILTER_SUNWARD_ONLY = True  

# Only highlights events with a CME present
PLOT_ONLY_ICME_EVENTS = False 

# Plot toggles
PLOT_OVERVIEW = True
PLOT_HELIOCENTRIC = True
PLOT_EARTH_FRAME = True
PLOT_INSITU_DST = True

# Save toggles
SAVE_PLOTS_TO_FILE = False 
SAVE_DATA_CSV = False
SAVE_TO_PDF = False 
TRACK_CONTEXT_DAYS = 42


# ---------------------------- Utilities --------------------------- #

class TimeoutError(Exception):
    pass

def _extract_constants_block():
    """
    Extract the Constants & Toggles section directly from this file so the saved
    configuration snapshot matches the executed run settings.
    """
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return "ERROR: Could not read __file__ to extract constants block.\n"

    pattern = (
        r"#\s*-+\s*Constants\s*&\s*Toggles\s*-+\s*#\s*\n"
        r"(.*?)"
        r"(?=\n#\s*-+\s*Utilities\s*-+\s*#)"
    )
    m = re.search(pattern, src, flags=re.DOTALL)
    if not m:
        return "ERROR: Could not locate Constants & Toggles section in source.\n"

    header = "# ---------------------------- Constants & Toggles --------------------------- #\n\n"
    return header + m.group(1).strip() + "\n"

def save_run_config(output_root: str):
    """
    Write a run_config.txt snapshot to the output root and, when present, to the
    plots/ and csv/ subdirectories so the run configuration can be reconstructed later.
    """
    if not output_root:
        return

    constants_block = _extract_constants_block()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = (
        f"# Run config snapshot generated: {stamp}\n"
        f"# Script: {Path(__file__).name}\n\n"
        + constants_block
    )

    root = Path(output_root)
    targets = [root, root / "plots", root / "csv"]

    for d in targets:
        try:
            if d.exists():
                (d / "run_config.txt").write_text(payload, encoding="utf-8")
        except Exception:
            # Don't hard-fail the run if writing the config snapshot fails
            pass

def setup_output_folder():
    if not SAVE_PLOTS_TO_FILE:
        return None
    
    # Check for non-interactive mode via environment variable
    folder_name = os.environ.get('OUTPUT_FOLDER_NAME')
    
    if folder_name:
        # Non-interactive mode - auto-create folder
        print("Plot output setup (automatic mode)")
        print(f"Root directory: {FIGURES_ROOT}")
        print(f"Using folder name: {folder_name}")
        
        # Remove any path separators to ensure it's just a folder name
        folder_name = folder_name.replace('/', '_').replace('\\', '_')
        
        full_path = FIGURES_ROOT / folder_name
        
        # Create the main folder (overwrite if exists)
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated output folder: {full_path}")
        
        # Create subdirectories if CSV saving is enabled
        if SAVE_DATA_CSV:
            plots_path = full_path / "plots"
            csv_path = full_path / "csv"
            plots_path.mkdir(parents=True, exist_ok=True)
            csv_path.mkdir(parents=True, exist_ok=True)
            print(f"Created plots subfolder: {plots_path}")
            print(f"Created csv subfolder: {csv_path}")
        
        print("="*80 + "\n")
        save_run_config(full_path)
        return full_path
    
    # Interactive mode - ask user for folder name
    print("Plot output setup")
    print(f"Root directory: {FIGURES_ROOT}")
    print("\nPlots will be saved to a subfolder within the root directory.")
    
    while True:
        folder_name = input("\nEnter folder name for this run: ").strip()
        
        if not folder_name:
            print("[ERROR] Folder name cannot be empty. Please try again.")
            continue
        
        # Remove any path separators to ensure it's just a folder name
        folder_name = folder_name.replace('/', '_').replace('\\', '_')
        
        full_path = FIGURES_ROOT / folder_name
        
        # Check if folder already exists
        if full_path.exists():
            overwrite = input(f"[WARN] Folder '{folder_name}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite == 'y':
                break
            else:
                continue
        else:
            break
    
    # Create the main folder
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated output folder: {full_path}")
    
    # Create subdirectories if CSV saving is enabled
    if SAVE_DATA_CSV:
        plots_path = full_path / "plots"
        csv_path = full_path / "csv"
        plots_path.mkdir(parents=True, exist_ok=True)
        csv_path.mkdir(parents=True, exist_ok=True)
        print(f"Created plots subfolder: {plots_path}")
        print(f"Created csv subfolder: {csv_path}")
    
    print("="*80 + "\n")
    save_run_config(full_path)
    return full_path
    
def timeout_handler(signum, frame):
    raise TimeoutError("CDF read timed out")

def safe_cdf_read(cdf_file, timeout=10):
    """Read CDF with timeout protection."""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        cdf = cdflib.CDF(cdf_file)
        signal.alarm(0)
        return cdf
    except TimeoutError:
        signal.alarm(0)
        print(f"[WARN] Timed out while reading CDF file: {os.path.basename(cdf_file)}")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"[WARN] Failed to read CDF file {os.path.basename(cdf_file)}: {e}")
        return None

def ensure_datetime_array(times):
    out = []
    for t in times:
        if isinstance(t, datetime):
            out.append(t)
        elif isinstance(t, np.datetime64):
            out.append(pd.to_datetime(t).to_pydatetime())
        else:
            # Try CDF epoch
            try:
                dt = cdflib.cdfepoch.to_datetime(t)
                if isinstance(dt, (list, np.ndarray)):
                    out.append(dt[0])
                else:
                    out.append(dt)
                continue
            except Exception:
                pass
            try:
                out.append(datetime.utcfromtimestamp(float(t)))
            except Exception:
                out.append(datetime(1970, 1, 1))
    return np.array(out, dtype=object)

def write_event_csv(event, preloaded_data, spacecraft_ephemeris, epoch_date, 
                   spacecraft_name, output_folder, window_hours=INSITU_WINDOW_HR):
    """
    Write event data to CSV file.
    """
    if not SAVE_DATA_CSV or not output_folder:
        return
    
    center = event.get("time")
    if not isinstance(center, datetime):
        return
    
    d = preloaded_data.get(center)
    if not d:
        return
    
    # Extract data
    mag = d.get("mag")
    swa_moments = d.get("swa_moments")
    dst_shifted = d.get("dst_shifted")
    kp_shifted = d.get("kp_shifted")
    l1_shifted_ballistic = d.get("l1_shifted_ballistic")
    
    if not mag or "time" not in mag or "B" not in mag:
        return
    
    t0 = center - timedelta(hours=window_hours)
    t1 = center + timedelta(hours=window_hours)
    
    # Get MAG data
    t_mag = ensure_datetime_array(mag["time"])
    b_mag = mag["B"]
    mask_mag = (t_mag >= t0) & (t_mag <= t1)

    if not np.any(mask_mag):
        return
    
    t_mag = t_mag[mask_mag]
    b_mag = b_mag[mask_mag]
    
    # Build CSV data dictionary
    csv_data = {
        'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S.%f') for t in t_mag],
        'unix_timestamp': [t.timestamp() for t in t_mag],
        'spacecraft': [spacecraft_name] * len(t_mag),
        'crossover_date': [center.strftime('%Y-%m-%d')] * len(t_mag),
        'crossover_time': [center.strftime('%H:%M:%S')] * len(t_mag),
    }
    
    # Spacecraft magnetic field (RTN)
    csv_data['B_r'] = b_mag[:, 0] if b_mag.ndim > 1 else np.full(len(t_mag), np.nan)
    csv_data['B_t'] = b_mag[:, 1] if b_mag.ndim > 1 else np.full(len(t_mag), np.nan)
    csv_data['B_n'] = b_mag[:, 2] if b_mag.ndim > 1 else np.full(len(t_mag), np.nan)
    csv_data['B_mag'] = np.linalg.norm(b_mag, axis=1) if b_mag.ndim > 1 else np.abs(b_mag)
    
    # Spacecraft plasma - interpolate to MAG times
    mag_times_sec = np.array([t.timestamp() for t in t_mag])
    
    if swa_moments and "time" in swa_moments:
        swa_times = ensure_datetime_array(swa_moments["time"])
        swa_times_sec = np.array([t.timestamp() for t in swa_times])
        
        # Density
        if "n" in swa_moments:
            n_swa = np.array(swa_moments["n"], dtype=float)
            f_n = interp1d(swa_times_sec, n_swa, kind='linear', bounds_error=False, fill_value=np.nan)
            csv_data['swa_n'] = f_n(mag_times_sec)
        else:
            csv_data['swa_n'] = np.full(len(t_mag), np.nan)
        
        # Velocity
        if "V" in swa_moments:
            v_swa = np.array(swa_moments["V"], dtype=float)
            if v_swa.ndim > 1 and v_swa.shape[1] == 3:
                for i, comp in enumerate(['swa_V_r', 'swa_V_t', 'swa_V_n']):
                    f_v = interp1d(swa_times_sec, v_swa[:, i], kind='linear', bounds_error=False, fill_value=np.nan)
                    csv_data[comp] = f_v(mag_times_sec)
                csv_data['swa_V_mag'] = np.sqrt(csv_data['swa_V_r']**2 + csv_data['swa_V_t']**2 + csv_data['swa_V_n']**2)
            else:
                for comp in ['swa_V_r', 'swa_V_t', 'swa_V_n', 'swa_V_mag']:
                    csv_data[comp] = np.full(len(t_mag), np.nan)
        else:
            for comp in ['swa_V_r', 'swa_V_t', 'swa_V_n', 'swa_V_mag']:
                csv_data[comp] = np.full(len(t_mag), np.nan)
        
        # Temperature
        if "T" in swa_moments:
            t_swa = np.array(swa_moments["T"], dtype=float)
            f_t = interp1d(swa_times_sec, t_swa, kind='linear', bounds_error=False, fill_value=np.nan)
            csv_data['swa_T'] = f_t(mag_times_sec)
        else:
            csv_data['swa_T'] = np.full(len(t_mag), np.nan)
    else:
        for col in ['swa_n', 'swa_V_r', 'swa_V_t', 'swa_V_n', 'swa_V_mag', 'swa_T']:
            csv_data[col] = np.full(len(t_mag), np.nan)
    
    # L1 DATA - BALLISTIC PROPAGATION
    if l1_shifted_ballistic and "B_gse_shifted" in l1_shifted_ballistic:
        l1_times = ensure_datetime_array(l1_shifted_ballistic["time"])
        l1_times_sec = np.array([t.timestamp() for t in l1_times])
        
        # B field
        b_l1 = l1_shifted_ballistic["B_gse_shifted"]
        if b_l1.ndim > 1 and b_l1.shape[1] == 3:
            for i, comp in enumerate(['l1_B_x_gse_ballistic', 'l1_B_y_gse_ballistic', 'l1_B_z_gse_ballistic']):
                f_b = interp1d(l1_times_sec, b_l1[:, i], kind='linear', bounds_error=False, fill_value=np.nan)
                csv_data[comp] = f_b(mag_times_sec)
            csv_data['l1_B_mag_ballistic'] = np.sqrt(
                csv_data['l1_B_x_gse_ballistic']**2 + 
                csv_data['l1_B_y_gse_ballistic']**2 + 
                csv_data['l1_B_z_gse_ballistic']**2
            )
        else:
            for comp in ['l1_B_x_gse_ballistic', 'l1_B_y_gse_ballistic', 'l1_B_z_gse_ballistic', 'l1_B_mag_ballistic']:
                csv_data[comp] = np.full(len(t_mag), np.nan)
        

        # Velocity
        if "V_shifted" in l1_shifted_ballistic and l1_shifted_ballistic["V_shifted"] is not None:
            v_l1 = np.array(l1_shifted_ballistic["V_shifted"], dtype=float)

            if v_l1.ndim > 1 and v_l1.shape[1] == 3:
                for i, comp in enumerate(['l1_V_x_ballistic', 'l1_V_y_ballistic', 'l1_V_z_ballistic']):
                    f_v = interp1d(l1_times_sec, v_l1[:, i], kind='linear', bounds_error=False, fill_value=np.nan)
                    csv_data[comp] = f_v(mag_times_sec)
                csv_data['l1_V_mag_ballistic'] = np.sqrt(
                    csv_data['l1_V_x_ballistic']**2 +
                    csv_data['l1_V_y_ballistic']**2 +
                    csv_data['l1_V_z_ballistic']**2
                )

            else:
                v_speed = v_l1.reshape(-1) if v_l1.ndim == 1 else v_l1[:, 0]
                f_vm = interp1d(l1_times_sec, v_speed, kind='linear', bounds_error=False, fill_value=np.nan)
                csv_data['l1_V_mag_ballistic'] = f_vm(mag_times_sec)
                csv_data['l1_V_x_ballistic'] = np.full(len(t_mag), np.nan)
                csv_data['l1_V_y_ballistic'] = np.full(len(t_mag), np.nan)
                csv_data['l1_V_z_ballistic'] = np.full(len(t_mag), np.nan)
        else:
            csv_data['l1_V_x_ballistic'] = np.full(len(t_mag), np.nan)
            csv_data['l1_V_y_ballistic'] = np.full(len(t_mag), np.nan)
            csv_data['l1_V_z_ballistic'] = np.full(len(t_mag), np.nan)
            csv_data['l1_V_mag_ballistic'] = np.full(len(t_mag), np.nan)
        
        # Density
        if "n_shifted" in l1_shifted_ballistic:
            n_l1 = l1_shifted_ballistic["n_shifted"]
            f_n = interp1d(l1_times_sec, n_l1, kind='linear', bounds_error=False, fill_value=np.nan)
            csv_data['l1_n_ballistic'] = f_n(mag_times_sec)
        else:
            csv_data['l1_n_ballistic'] = np.full(len(t_mag), np.nan)
        
        # Temperature
        if "T_shifted" in l1_shifted_ballistic:
            t_l1 = l1_shifted_ballistic["T_shifted"]
            f_t = interp1d(l1_times_sec, t_l1, kind='linear', bounds_error=False, fill_value=np.nan)
            csv_data['l1_T_ballistic'] = f_t(mag_times_sec)
        else:
            csv_data['l1_T_ballistic'] = np.full(len(t_mag), np.nan)
        
        # Propagation delay and distance
        if "propagation_delay_hours" in l1_shifted_ballistic:
            delay = l1_shifted_ballistic["propagation_delay_hours"]
            f_delay = interp1d(l1_times_sec, delay, kind='linear', bounds_error=False, fill_value=np.nan)
            csv_data['l1_propagation_delay_hours_ballistic'] = f_delay(mag_times_sec)
        else:
            csv_data['l1_propagation_delay_hours_ballistic'] = np.full(len(t_mag), np.nan)
        
        if "propagation_distance_au" in l1_shifted_ballistic:
            dist = l1_shifted_ballistic["propagation_distance_au"]
            f_dist = interp1d(l1_times_sec, dist, kind='linear', bounds_error=False, fill_value=np.nan)
            csv_data['l1_propagation_distance_au_ballistic'] = f_dist(mag_times_sec)
        else:
            csv_data['l1_propagation_distance_au_ballistic'] = np.full(len(t_mag), np.nan)
    else:
        for col in ['l1_B_x_gse_ballistic', 'l1_B_y_gse_ballistic', 'l1_B_z_gse_ballistic', 'l1_B_mag_ballistic',
                    'l1_V_x_ballistic', 'l1_V_y_ballistic', 'l1_V_z_ballistic', 'l1_V_mag_ballistic',
                    'l1_n_ballistic', 'l1_T_ballistic', 
                    'l1_propagation_delay_hours_ballistic', 'l1_propagation_distance_au_ballistic']:
            csv_data[col] = np.full(len(t_mag), np.nan)
    
    
    # Geomagnetic indices (ballistic shifted)
    if dst_shifted and "dst_shifted" in dst_shifted:
        dst_times = ensure_datetime_array(dst_shifted["time"])
        dst_times_sec = np.array([t.timestamp() for t in dst_times])
        dst_vals = np.array(dst_shifted["dst_shifted"], dtype=float)
        f_dst = interp1d(dst_times_sec, dst_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        csv_data['dst_shifted'] = f_dst(mag_times_sec)
    else:
        csv_data['dst_shifted'] = np.full(len(t_mag), np.nan)
    
    if kp_shifted and "kp_shifted" in kp_shifted:
        kp_times = ensure_datetime_array(kp_shifted["time"])
        kp_times_sec = np.array([t.timestamp() for t in kp_times])
        kp_vals = np.array(kp_shifted["kp_shifted"], dtype=float)
        f_kp = interp1d(kp_times_sec, kp_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        csv_data['kp_shifted'] = f_kp(mag_times_sec)
    else:
        csv_data['kp_shifted'] = np.full(len(t_mag), np.nan)
    
    # Spacecraft ephemeris - interpolate to MAG times
    sc_times = spacecraft_ephemeris["times"]
    sc_positions = spacecraft_ephemeris["positions"]
    sc_times_sec = np.array([t.timestamp() for t in sc_times])
    
    for i, coord in enumerate(['sc_x_hci', 'sc_y_hci', 'sc_z_hci']):
        f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind='linear', bounds_error=False, fill_value=np.nan)
        csv_data[coord] = f_pos(mag_times_sec)
    
    # Convert to AU
    csv_data['sc_x_au'] = csv_data['sc_x_hci'] / AU_KM
    csv_data['sc_y_au'] = csv_data['sc_y_hci'] / AU_KM
    csv_data['sc_z_au'] = csv_data['sc_z_hci'] / AU_KM
    
    # Distance to Earth and angle
    earth_positions = np.array([SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date) for t in t_mag])
    sc_pos_array = np.column_stack([csv_data['sc_x_hci'], csv_data['sc_y_hci'], csv_data['sc_z_hci']])
    
    distances = np.linalg.norm(sc_pos_array - earth_positions, axis=1) / AU_KM
    csv_data['sc_distance_to_earth_au'] = distances
    
    angles = []
    for i in range(len(t_mag)):
        earth_hat = earth_positions[i] / np.linalg.norm(earth_positions[i])
        sc_hat = sc_pos_array[i] / np.linalg.norm(sc_pos_array[i])
        cos_angle = np.clip(np.dot(earth_hat, sc_hat), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
    csv_data['sc_angle_from_sun_earth_line_deg'] = np.array(angles)
    
    # Earth position
    csv_data['earth_x_hci'] = earth_positions[:, 0]
    csv_data['earth_y_hci'] = earth_positions[:, 1]
    csv_data['earth_z_hci'] = earth_positions[:, 2]
    csv_data['earth_x_au'] = earth_positions[:, 0] / AU_KM
    csv_data['earth_y_au'] = earth_positions[:, 1] / AU_KM
    csv_data['earth_z_au'] = earth_positions[:, 2] / AU_KM
    
    # Data quality flags
    csv_data['mag_data_available'] = 1
    csv_data['swa_data_available'] = 1 if (swa_moments and "n" in swa_moments) else 0
    csv_data['l1_ballistic_available'] = 1 if (l1_shifted_ballistic and "B_gse_shifted" in l1_shifted_ballistic) else 0
    csv_data['dst_data_available'] = 1 if (dst_shifted and "dst_shifted" in dst_shifted) else 0
    csv_data['kp_data_available'] = 1 if (kp_shifted and "kp_shifted" in kp_shifted) else 0
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    
    # Save to csv subfolder
    csv_folder = Path(output_folder) / "csv"
    safe_sc_name = spacecraft_name.replace(" ", "_").replace("-", "")
    timestamp = center.strftime('%Y%m%d')
    csv_filename = f"event_{safe_sc_name}_{timestamp}.csv"
    csv_path = csv_folder / csv_filename
    
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"[INFO] Saved CSV: {csv_path} ({len(df)} rows).")
    
def _warn_out_of_range(name, arr, units="", expected_min=None, expected_max=None):
    arr = np.asarray(arr, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    
    if expected_min is not None or expected_max is not None:
        bad_low = bad_high = 0
        if expected_min is not None:
            bad_low = int(np.sum(finite < expected_min))
        if expected_max is not None:
            bad_high = int(np.sum(finite > expected_max))
        if bad_low or bad_high:
            print(f"[WARN] {name}: {bad_low} values below {expected_min} {units}, "
                f"{bad_high} values above {expected_max} {units}.")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
})


# ----------------------- Ephemerides + geometry -------------------------- #

class SpacecraftData:
    HORIZONS_IDS = {
        "STEREO-A":          "-234",
        "STEREO-B":          "-235",
        "Solar Orbiter":     "-144",
        "Parker Solar Probe":"-96",
        "Earth":             "399", 
    }

    @staticmethod
    def get_positions(spacecraft_name, start_date, end_date, dt_hours=6):
        if not SUNPY_AVAILABLE:
            return None
        if spacecraft_name not in SpacecraftData.HORIZONS_IDS:
            return None

        horizons_id   = SpacecraftData.HORIZONS_IDS[spacecraft_name]
        chunk_days    = 365
        all_times     = []
        all_positions = []

        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            times       = []
            current     = current_start
            while current <= current_end:
                times.append(current)
                current += timedelta(hours=dt_hours)
            if not times:
                break

            try:
                t_astropy = Time([t.isoformat() for t in times])
                coords    = get_horizons_coord(horizons_id, t_astropy)
                ecliptic  = coords.transform_to("heliocentriceclipticiau76")

                r_km = ecliptic.distance.to(u.km).value
                lon  = ecliptic.lon.to(u.rad).value
                lat  = ecliptic.lat.to(u.rad).value

                x = r_km * np.cos(lat) * np.cos(lon)
                y = r_km * np.cos(lat) * np.sin(lon)
                z = r_km * np.sin(lat)

                positions = np.column_stack([x, y, z])
                all_times.extend(times)
                all_positions.append(positions)
            except Exception:
                pass

            current_start = current_end + timedelta(hours=dt_hours)

        if not all_positions:
            return None

        combined_positions = np.vstack(all_positions)
        return {"times": all_times, "positions": combined_positions}

class SunEarthLineAnalyzer:
    _earth_t_sec = None
    _earth_x = None
    _earth_y = None
    _earth_z = None

    @staticmethod
    def set_earth_ephemeris(times, positions):
        if times is None or positions is None or len(times) == 0:
            raise ValueError("Empty Earth ephemeris passed to set_earth_ephemeris")

        t_sec = np.array([t.timestamp() for t in times], dtype=float)
        pos   = np.asarray(positions, dtype=float)

        SunEarthLineAnalyzer._earth_t_sec = t_sec
        SunEarthLineAnalyzer._earth_x = pos[:, 0]
        SunEarthLineAnalyzer._earth_y = pos[:, 1]
        SunEarthLineAnalyzer._earth_z = pos[:, 2]


    @staticmethod
    def earth_position_at_time(time_dt, epoch_dt=None):
        if SunEarthLineAnalyzer._earth_t_sec is not None:
            t = float(time_dt.timestamp())

            # 1D linear interpolation for each component
            x = np.interp(t, SunEarthLineAnalyzer._earth_t_sec, SunEarthLineAnalyzer._earth_x)
            y = np.interp(t, SunEarthLineAnalyzer._earth_t_sec, SunEarthLineAnalyzer._earth_y)
            z = np.interp(t, SunEarthLineAnalyzer._earth_t_sec, SunEarthLineAnalyzer._earth_z)
            return np.array([x, y, z])

        if epoch_dt is None:
            raise RuntimeError(
                "Earth ephemeris not set and no epoch_dt provided for fallback."
            )
        days  = (time_dt - epoch_dt).total_seconds() / 86400.0
        theta = MEAN_MOTION * days
        return np.array([AU_KM * np.cos(theta), AU_KM * np.sin(theta), 0.0])

    @staticmethod
    def distance_to_sun_earth_line(spacecraft_pos, earth_pos):
        v      = earth_pos
        v_hat  = v / np.linalg.norm(v)
        w      = spacecraft_pos
        s      = np.dot(w, v_hat)
        proj   = s * v_hat
        perp   = w - proj
        is_between = 0 < s < np.linalg.norm(v)
        return np.linalg.norm(perp), s, is_between
    
class CrossoverFinder:
    def __init__(self, constellation, sun_earth_tolerance_au=0.05, dro_tolerance_au=0.05):
        self.constellation = constellation
        self.se_tol_km     = sun_earth_tolerance_au * AU_KM
        self.dro_tol_km    = dro_tolerance_au * AU_KM

    def find_events(self, spacecraft_data, epoch_date):
        events = []
        for time, pos in zip(spacecraft_data["times"], spacecraft_data["positions"]):
            earth_pos  = SunEarthLineAnalyzer.earth_position_at_time(time, epoch_date)
            perp, s, is_between = SunEarthLineAnalyzer.distance_to_sun_earth_line(pos, earth_pos)
            if perp > self.se_tol_km:
                continue
            for sat in self.constellation.satellites:
                dro_positions = sat["orbit"]["dro_helio"]
                dists         = np.linalg.norm(dro_positions - pos[:2], axis=1)
                j             = np.argmin(dists)
                dmin          = dists[j]
                if dmin < self.dro_tol_km:
                    events.append({
                        "time":                 time,
                        "spacecraft_pos":       pos,
                        "earth_pos":            earth_pos,
                        "dro_id":               sat["id"],
                        "dro_pos":              dro_positions[j],
                        "perp_to_se_line_au":   perp / AU_KM,
                        "along_se_line_au":     s / AU_KM,
                        "dist_to_dro_au":       dmin / AU_KM,
                        "is_between_sun_earth": is_between,
                    })
        return events

    def group_events_by_day(self, events):
        """
        Group events by week and select the best candidate per week.

        Events are first filtered to retain only those between the Sun and Earth
        (if FILTER_SUNWARD_ONLY is set), then grouped into 7-day windows.
        The event with the smallest distance to the DRO is selected from each group
        """
        if not events:
            return []
        

        events.sort(key=lambda x: x["time"])
        
        print(f"\n[FILTER] Starting with {len(events)} raw crossover detections")
        
        if FILTER_SUNWARD_ONLY:
            sunward_events = [e for e in events if e["is_between_sun_earth"]]
            print(f"[FILTER] After sunward filter: {len(sunward_events)} events")
            print(f"         (removed {len(events) - len(sunward_events)} events behind Earth or beyond Sun)")
            events = sunward_events
        
        if not events:
            print("[FILTER] No events remaining after filters")
            return []
        
        groups = []
        cur = [events[0]]
        for e in events[1:]:
            if (e["time"] - cur[0]["time"]).days < 7:
                cur.append(e)
            else:
                groups.append(cur)
                cur = [e]
        if cur:
            groups.append(cur)
        
        best = [min(g, key=lambda x: x["dist_to_dro_au"]) for g in groups]
        
        return best


# ------------------------- L1 OMNI loader + ballistic -------------------- #

class L1DataLoader:
    @staticmethod
    def load_omni_data(t0_dt, t1_dt):
        """
        Load OMNI high-res OMNI2 data with inline cleaning per variable.
        """
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                    t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            files = pyspedas.omni.data(trange=trange,
                                    level='hro',
                                    time_clip=True,
                                    downloadonly=True)
            if not files:
                return None

            all_times = []
            all_B = []
            all_V = []
            all_n = []
            all_T = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info = cdf.cdf_info()
                    vars_list = cdf_info.zVariables

                    if 'Epoch' in vars_list:
                        epoch = cdf.varget('Epoch')
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        all_times.extend(times_dt)
                    else:
                        continue

                    # B-field
                    b_data = None
                    for b_var in ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'B']:
                        if b_var in vars_list:
                            try:
                                if b_var == 'B':
                                    b_data = cdf.varget(b_var)
                                    if b_data.ndim == 1:
                                        b_data = b_data.reshape(-1, 1)
                                    break
                                else:
                                    bx = cdf.varget('BX_GSE')
                                    by = cdf.varget('BY_GSE')
                                    bz = cdf.varget('BZ_GSE')
                                    bx = -bx
                                    by = -by
                                    b_data = np.column_stack([bx, by, bz])
                                    break
                            except Exception:
                                continue

                    # Velocity
                    v_data = None
                    for v_var in ['flow_speed', 'V', 'Vx']:
                        if v_var in vars_list:
                            try:
                                v_data = cdf.varget(v_var)
                                if len(v_data.shape) == 1:
                                    v_data = v_data.reshape(-1, 1)
                                break
                            except Exception:
                                continue

                    # Density
                    n_data = None
                    for n_var in ['proton_density', 'Np', 'N']:
                        if n_var in vars_list:
                            try:
                                n_data = cdf.varget(n_var)
                                break
                            except Exception:
                                continue

                    # Temperature
                    t_data = None
                    for t_var in ['T', 'Tp', 'proton_temperature']:
                        if t_var in vars_list:
                            try:
                                t_data = cdf.varget(t_var)
                                break
                            except Exception:
                                continue

                    if b_data is not None:
                        all_B.append(b_data)
                    if v_data is not None:
                        all_V.append(v_data)
                    if n_data is not None:
                        all_n.append(n_data)
                    if t_data is not None:
                        all_T.append(t_data)

                except Exception as e:
                    print(f"[WARN] Failed to read OMNI data from {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_times:
                return None

            time_arr = ensure_datetime_array(all_times)
            result = {"time": time_arr}

            # B-FIELD CLEANING
            if all_B:
                B_combined = np.vstack(all_B).astype(float)
                
                # Mask B-field sentinels and bad values
                mask_B = (
                    (~np.isfinite(B_combined)) |
                    (np.abs(B_combined) >= 1e29) |
                    (B_combined == 99999) |
                    (B_combined == 9999) |
                    (np.abs(B_combined) > 200.0)  # Physical limit
                )
                
                n_bad_B = np.sum(np.any(mask_B, axis=1))
                if n_bad_B > 0:
                    print(f"[WARN] L1 magnetic field data: masking {n_bad_B} invalid samples.")
                
                B_combined[mask_B] = np.nan
                result["B_gse"] = B_combined
                
                if B_combined.ndim == 1 or B_combined.shape[1] == 1:
                    _warn_out_of_range("L1 B (clean)", B_combined, "nT",
                            expected_min=-100, expected_max=100)
                else:
                    Bmag_clean = np.linalg.norm(B_combined, axis=1)
                    _warn_out_of_range("L1 |B| (clean)", Bmag_clean, "nT",
                            expected_min=0, expected_max=100)

            # VELOCITY CLEANING 
            if all_V:
                V_combined = np.vstack(all_V).astype(float)
                
                # Mask velocity sentinels and bad values
                mask_V = (
                    (~np.isfinite(V_combined)) |
                    (np.abs(V_combined) >= 1e29) |
                    (V_combined == 9999) |
                    (V_combined == 99999) |
                    (np.abs(V_combined) > 3000.0) |  # Physical limit
                    (V_combined < 0)
                )
                
                n_bad_V = np.sum(np.any(mask_V, axis=1))
                if n_bad_V > 0:
                    print(f"[WARN] L1 solar wind speed data: masking {n_bad_V} invalid samples.")
                
                V_combined[mask_V] = np.nan
                result["V"] = V_combined
                
                Vmag_clean = np.linalg.norm(V_combined, axis=1) if V_combined.ndim > 1 else V_combined
                _warn_out_of_range("L1 V (clean)", Vmag_clean, "km/s",
                        expected_min=0, expected_max=2000)

            # DENSITY CLEANING
            if all_n:
                n_combined = np.hstack(all_n).astype(float)
                
                # Mask density sentinels and bad values
                mask_n = (
                    (~np.isfinite(n_combined)) |
                    (np.abs(n_combined) >= 1e29) |
                    (n_combined >= 999.0) |
                    (n_combined < 0)
                )
                
                n_bad_n = np.sum(mask_n)
                if n_bad_n > 0:
                    print(f"[WARN] L1 density data: masking {n_bad_n} invalid samples.")
                
                n_combined[mask_n] = np.nan
                result["n"] = n_combined
                
                _warn_out_of_range("L1 n (clean)", n_combined, "cm^-3",
                        expected_min=0, expected_max=100)

            # TEMPERATURE CLEANING
            if all_T:
                T_combined = np.hstack(all_T).astype(float)
                
                mask_T = (
                    (~np.isfinite(T_combined)) |
                    (np.abs(T_combined) >= 1e29) |
                    (T_combined == 9999999) |
                    (T_combined > 1e7) |
                    (T_combined < 0)
                )
                
                n_bad_T = np.sum(mask_T)
                if n_bad_T > 0:
                    print(f"[WARN] L1 temperature data: masking {n_bad_T} invalid samples before conversion from Kelvin to eV.")
                
                T_combined[mask_T] = np.nan
                # Convert from Kelvin to eV
                T_eV = T_combined / 11604.5
                
                result["T"] = T_eV
                
                _warn_out_of_range("L1 T (clean)", T_eV, "eV",
                        expected_min=0, expected_max=200)

            return result if len(result) > 1 else None

        except Exception as e:
            print(f"[WARN] OMNI data loading failed: {e}")
            return None
    
    @staticmethod
    def get_l1_position(time_dt, epoch_date):
        earth_pos = SunEarthLineAnalyzer.earth_position_at_time(time_dt, epoch_date)
        earth_sun_dir = -earth_pos / np.linalg.norm(earth_pos)
        l1_pos = earth_pos + earth_sun_dir * (0.01 * AU_KM)
        return l1_pos

class BallisticPropagation:
    @staticmethod
    def calculate_propagation_delay(sc_pos, l1_pos, v_sw):
        """
        Ballistic propagation: assumes radial propagation at solar wind speed.
        """
        delta_r = l1_pos - sc_pos
        v_mag = np.linalg.norm(v_sw)
        
        # Fallback to typical solar wind speed if velocity is invalid
        if v_mag < 1.0:
            v_mag = 400.0
            v_sw = np.array([v_mag, 0, 0])
        
        full_distance = np.linalg.norm(delta_r)
        delay_seconds = full_distance / v_mag
        
        return delay_seconds, full_distance

    @staticmethod
    def apply_ballistic_shift_to_l1(crossover_data, l1_data, sc_ephemeris, epoch_date):
        if not l1_data or not crossover_data:
            return None

        sc_times_sec = np.array([t.timestamp() for t in sc_ephemeris["times"]])
        sc_positions = sc_ephemeris["positions"]

        crossover_times_sec = np.array([t.timestamp() for t in crossover_data["time"]])

        sc_pos_interp = np.zeros((len(crossover_times_sec), 3))
        for i in range(3):
            f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind="linear",
                             bounds_error=False, fill_value=np.nan)
            sc_pos_interp[:, i] = f_pos(crossover_times_sec)

        l1_positions = np.array([
            L1DataLoader.get_l1_position(t, epoch_date)
            for t in crossover_data["time"]
        ])

        propagation_delays = np.zeros(len(crossover_times_sec))
        propagation_distances = np.zeros(len(crossover_times_sec))

        for i in range(len(crossover_times_sec)):
            if np.any(np.isnan(sc_pos_interp[i])):
                propagation_delays[i] = np.nan
                continue
            v_sw = crossover_data["V"][i]
            delay_sec, dist_km = BallisticPropagation.calculate_propagation_delay(
                sc_pos_interp[i], l1_positions[i], v_sw
            )
            propagation_delays[i] = delay_sec
            propagation_distances[i] = dist_km

        l1_times_sec = np.array([t.timestamp() for t in l1_data["time"]])
        shifted_times_sec = crossover_times_sec + propagation_delays

        result = {
            "time": crossover_data["time"],
            "propagation_delay_hours": propagation_delays / 3600.0,
            "propagation_distance_au": propagation_distances / AU_KM,
        }

        if "B_gse" in l1_data and l1_data["B_gse"] is not None:
            B_l1 = l1_data["B_gse"]
            if len(B_l1.shape) == 1:
                B_l1 = B_l1.reshape(-1, 1)

            B_shifted = np.zeros((len(shifted_times_sec), B_l1.shape[1]))
            for j in range(B_l1.shape[1]):
                f_b = interp1d(l1_times_sec, B_l1[:, j], kind="linear",
                               bounds_error=False, fill_value=np.nan)
                B_shifted[:, j] = f_b(shifted_times_sec)
            result["B_gse_shifted"] = B_shifted

        if "V" in l1_data and l1_data["V"] is not None:
            V_l1 = l1_data["V"]
            if len(V_l1.shape) == 1:
                V_l1 = V_l1.reshape(-1, 1)

            V_shifted = np.zeros((len(shifted_times_sec), V_l1.shape[1]))
            for j in range(V_l1.shape[1]):
                f_v = interp1d(l1_times_sec, V_l1[:, j], kind="linear",
                               bounds_error=False, fill_value=np.nan)
                V_shifted[:, j] = f_v(shifted_times_sec)
            result["V_shifted"] = V_shifted

        if "n" in l1_data and l1_data["n"] is not None:
            f_n = interp1d(l1_times_sec, l1_data["n"], kind="linear",
                           bounds_error=False, fill_value=np.nan)
            result["n_shifted"] = f_n(shifted_times_sec)

        if "T" in l1_data and l1_data["T"] is not None:
            f_t = interp1d(l1_times_sec, l1_data["T"], kind="linear",
                           bounds_error=False, fill_value=np.nan)
            result["T_shifted"] = f_t(shifted_times_sec)

        has_b = "B_gse_shifted" in result and not np.all(np.isnan(result["B_gse_shifted"]))
        if not has_b:
            return None
        return result


# ------------------------------ ICME Catalogue --------------------------- #

class CME_Catalogue:   
    @staticmethod
    def load_catalouge():
        print("\n[INFO] Loading Richardson & Cane ICME catalogue...")
        
        url = "http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.content.decode('cp775')

            # Convert HTML content to a string buffer for pandas
            dfs = pd.read_html(StringIO(html_content))
            
            if not dfs:
                return None
            
            df = dfs[0]
            icme_list = CME_Catalogue._parse_catalouge(df)
            
            if icme_list:
                print(f"[INFO] Parsed {len(icme_list)} ICME events from the catalogue.")
                return icme_list
            else:
                print("[WARN] ICME catalogue parsing completed, but no events were extracted.")
                return None
                
        except Exception as e:
            print(f"[WARN] Failed to download or parse the ICME catalogue: {e}")
            return None
    
    @staticmethod
    def _parse_catalouge(df):
        icme_list = []
        
        # Flatten multi-level columns by taking first level
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Column indices:
        # 0: Disturbance time
        # 1: ICME start
        # 2: ICME end
        # 14: Magnetic cloud flag (1/2/3 = magnetic cloud, else = ejecta only)
        
        for idx, row in df.iterrows():
            try:
                # Skip header rows (first ~28 rows are headers/metadata)
                if idx < 28:
                    continue
                
                # Get start and end times (columns 1 and 2)
                start_str = str(row.iloc[1]).strip()
                end_str = str(row.iloc[2]).strip()
                
                # Skip if invalid
                if start_str == 'nan' or end_str == 'nan' or start_str == '...' or end_str == '...':
                    continue
                
                # Parse datetime (format is "YYYY/MM/DD HHMM")
                start_dt = datetime.strptime(start_str, "%Y/%m/%d %H%M")
                end_dt = datetime.strptime(end_str, "%Y/%m/%d %H%M")
                
                # Get MC flag (column 14)
                mc_flag = str(row.iloc[14]).strip()
                icme_type = 'MC' if mc_flag in ['1', '2', '3'] else 'E'
                
                # Get shock flag (from disturbance time vs ICME start)
                disturbance_str = str(row.iloc[0]).strip()
                shock_flag = 'N'
                if disturbance_str != 'nan' and disturbance_str != start_str:
                    shock_flag = 'Y'
                
                icme_list.append({
                    'start': start_dt,
                    'end': end_dt,
                    'type': icme_type,
                    'shock': shock_flag,
                    'v_max': 400,  # Default (not always in table)
                    'b_max': 10,   # Default (not always in table)
                    'comment': f"{icme_type} event"
                })
                
            except Exception as e:
                # Skip rows that can't be parsed
                continue
        
        return icme_list
    
    @staticmethod
    def load_from_local_file(filepath=None):
        if filepath is None:
            filepath = PROJECT_ROOT / "icme_catalogue.html"
        else:
            filepath = Path(filepath)

        print(f"\n[INFO] Loading ICME catalogue from local file: {filepath}")
        
        try:
            from io import StringIO
            
            with filepath.open('rb') as f:
                html_content = f.read().decode('cp775')
            
            dfs = pd.read_html(StringIO(html_content))
            
            if not dfs:
                print("[WARN] No tables were found in the local ICME catalogue file.")
                return None
            
            df = dfs[0]
            print(f"[INFO] Found ICME catalogue table with {len(df)} rows.")
            
            icme_list = CME_Catalogue._parse_catalouge(df)
            
            if icme_list:
                print(f"[INFO] Parsed {len(icme_list)} ICME events from the local catalogue.")
                return icme_list
            else:
                print("[WARN] Failed to parse the local ICME catalogue.")
                return None
                
        except Exception as e:
            print(f"[WARN] Failed to load the local ICME catalogue file: {e}")
            return None
    
    @staticmethod
    def backpropagate_icmes(event, spacecraft_ephemeris, epoch_date, icme_list,
                        common_data=None, tolerance_hours=24):
        """
        Back-propagate ICMEs from Earth to spacecraft using ballistic propagation.
        
        Args:
            event: Crossover event dictionary
            spacecraft_ephemeris: S/C ephemeris data
            epoch_date: Epoch datetime
            icme_list: List of ICME events from catalogue
            common_data: Common spacecraft data (for velocity)
            tolerance_hours: Matching tolerance window
        
        Returns:
            List of matched ICMEs with back-propagated times, or None
        """
        print(f"\n[INFO] Back-propagating ICME arrival for event at {event['time']}.")
        
        if not icme_list:
            print("[INFO] No ICME catalogue available")
            return None
        
        # Get spacecraft position at crossover
        t_crossover = event["time"]
        sc_pos = event["spacecraft_pos"]
        
        # Calculate distance to Earth
        earth_pos = SunEarthLineAnalyzer.earth_position_at_time(t_crossover, epoch_date)
        distance_km = np.linalg.norm(sc_pos - earth_pos)
        distance_au = distance_km / AU_KM
            
        # Get solar wind velocity from measurements if available
        if common_data and "V" in common_data:
            times_common = common_data["time"]
            time_diffs = [abs((t - t_crossover).total_seconds()) for t in times_common]
            idx_closest = np.argmin(time_diffs)
            V_vec = common_data["V"][idx_closest]
            v_sw_kms = np.linalg.norm(V_vec)
            print(f"[INFO] Using measured solar wind speed: {v_sw_kms:.1f} km/s")
        else:
            v_sw_kms = 400.0
            print(f"[INFO] Using default solar wind speed: {v_sw_kms:.1f} km/s (no in-situ plasma speed available).")
        
        # Calculate propagation time
        propagation_time_sec = distance_km / v_sw_kms
        propagation_time_hrs = propagation_time_sec / 3600.0
        
        # Predicted arrival time at Earth
        t_predicted_earth = t_crossover + timedelta(seconds=propagation_time_sec)

        # Find matching ICMEs (within tolerance)
        matched_icmes = []
        tolerance_td = timedelta(hours=tolerance_hours)
        
        print(f"[INFO] Checking predicted arrival against {len(icme_list)} ICME catalogue entries.")
        
        for i, icme in enumerate(icme_list, 1):
            time_diff_start = abs(t_predicted_earth - icme['start'])
            time_diff_end = abs(t_predicted_earth - icme['end'])
            
            
            # Match if predicted arrival is within ICME duration ± tolerance
            if (icme['start'] - tolerance_td <= t_predicted_earth <= icme['end'] + tolerance_td or
                time_diff_start <= tolerance_td or time_diff_end <= tolerance_td):
                
                # Back-propagate ICME times to spacecraft
                sc_start = icme['start'] - timedelta(seconds=propagation_time_sec)
                sc_end = icme['end'] - timedelta(seconds=propagation_time_sec)
                
                print(f"[INFO] Matched ICME: Earth {icme['start']} to {icme['end']}; spacecraft {sc_start} to {sc_end}.")
                
                matched_icmes.append({
                    'earth_start': icme['start'],
                    'earth_end': icme['end'],
                    'sc_start': sc_start,
                    'sc_end': sc_end,
                    'type': icme['type'],
                    'shock': icme['shock'],
                    'comment': icme['comment'],
                    'distance_au': distance_au,
                    'propagation_hours': propagation_time_hrs,
                    'v_sw_used': v_sw_kms,
                    'method': 'Ballistic'
                })
        
        if matched_icmes:
            print(f"[INFO] Found {len(matched_icmes)} ICME match(es) using ballistic propagation.")
        else:
            print("[INFO] No matching ICME events were found for this crossover.")
        
        return matched_icmes if matched_icmes else None


# ------------------------ Solo in-situ + geomag -------------------------- #

class SoloDataLoader:
    @staticmethod
    def load_mag(t0_dt, t1_dt):
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                      t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            files = pyspedas.projects.solo.mag(
                trange=trange,
                datatype="rtn-normal",
                level="l2",
                time_clip=True,
                downloadonly=True,
            )
            if not files:
                return None

            print(f"[INFO] Loading {len(files)} Solar Orbiter MAG files")
            all_times = []
            all_B = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info  = cdf.cdf_info()
                    vars_list = cdf_info.zVariables

                    b_var = next((v for v in vars_list
                                  if 'B_RTN' in v or 'B_rtn' in v), None)

                    if b_var and 'EPOCH' in vars_list:
                        epoch = cdf.varget('EPOCH')
                        b_data = cdf.varget(b_var)

                        n = len(epoch)
                        if n == 0:
                            continue

                        # Decimate per file
                        step = max(1, n // MAG_TARGET_POINTS_PER_DAY)
                        idx = np.arange(0, n, step, dtype=int)

                        epoch_sel = epoch[idx]
                        b_sel     = b_data[idx, :] if b_data.ndim > 1 else b_data[idx]

                        times_dt = cdflib.cdfepoch.to_datetime(epoch_sel)
                        all_times.extend(times_dt)
                        all_B.append(b_sel)

                        if b_sel.ndim == 1:
                            bmag = np.abs(b_sel)
                        else:
                            bmag = np.linalg.norm(b_sel, axis=1)

                except Exception as e:
                    print(f"[WARN] Failed to read Solar Orbiter MAG file {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_B:
                print("[WARN] No valid Solar Orbiter MAG data could be read from the downloaded files.")
                return None

            B_combined = np.vstack(all_B)
            times_arr  = ensure_datetime_array(all_times)

            # Check field magnitude range for possible unit or scaling issues
            if B_combined.ndim == 1:
                Bmag = np.abs(B_combined)
            else:
                Bmag = np.linalg.norm(B_combined, axis=1)


            if np.nanmax(Bmag) > 1e3:
                print("[WARN] Solar Orbiter MAG field magnitude exceeds expected range; units or scaling may be incorrect.")

            return {"time": times_arr, "B": B_combined}

        except Exception as e:
            print(f"[WARN] Solar Orbiter MAG loading failed: {e}")
            return None

    @staticmethod
    def load_swa_moments(t0_dt, t1_dt):
        """
        Load Solar Orbiter SWA PAS ground moments (L2 PAS-GRND-MOM).
        Reads CDF files directly since pyspedas.projects.solo.swa() doesn't load into pytplot.
        
        Returns:
            {
                "time": list of datetime,
                "n": 1D array (density),
                "V": 2D array [ntimes, 3] (velocity RTN),
                "T": 1D array (temperature)
            }
            or None on failure.
        """
        try:
            trange = [
                t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                t1_dt.strftime("%Y-%m-%d/%H:%M:%S")
            ]

            # Download the requested CDF files
            files = pyspedas.projects.solo.swa(
                trange=trange,
                datatype="pas-grnd-mom",
                level="l2",
                time_clip=True,
                downloadonly=True 
            )
            
            if not files:
                print("[INFO] No Solar Orbiter SWA moments files were available for the requested interval.")
                return None

            print(f"[INFO] Loading {len(files)} Solar Orbiter SWA moments files.")
            
            all_times = []
            all_n = []
            all_v = []
            all_t = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    # Read variables directly from CDF
                    epoch = cdf.varget('Epoch')
                    times_dt = cdflib.cdfepoch.to_datetime(epoch)
                    
                    n_data = cdf.varget('N')          # Density
                    v_data = cdf.varget('V_RTN')      # Velocity [n×3]
                    t_data = cdf.varget('T')          # Temperature
                    
                    # Accumulate data
                    all_times.extend(times_dt)
                    all_n.extend(n_data)
                    all_v.extend(v_data)
                    all_t.extend(t_data)
                    
                except Exception as e:
                    print(f"[WARN] Could not read SWA moments file {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_times:
                print("[WARN] SWA moments files were found, but no readable data were extracted.")
                return None

            # Convert to arrays
            times_array = ensure_datetime_array(all_times)
            n_array = np.array(all_n, dtype=float)
            v_array = np.array(all_v, dtype=float)
            t_array = np.array(all_t, dtype=float)
            
            # Sort by time (in case files were out of order)
            sort_idx = np.argsort([t.timestamp() for t in times_array])
            times_array = times_array[sort_idx]
            n_array = n_array[sort_idx]
            v_array = v_array[sort_idx]
            t_array = t_array[sort_idx]
            
            print(f"[INFO] Loaded Solar Orbiter SWA moments ({len(times_array)} samples).")
            _warn_out_of_range("SWA n", n_array, "cm^-3", expected_min=0, expected_max=100)
            _warn_out_of_range("SWA |V|", np.linalg.norm(v_array, axis=1), "km/s", 
                    expected_min=0, expected_max=1000)
            _warn_out_of_range("SWA T", t_array, "eV", expected_min=0, expected_max=500)

            return {
                "time": times_array,
                "n": n_array,
                "V": v_array,
                "T": t_array
            }

        except Exception as e:
            print(f"[WARN] Solar Orbiter SWA moments loading failed: {e}")
            return None

    @staticmethod
    def load_swa_eflux(t0_dt, t1_dt):
        """
        Load Solar Orbiter SWA PAS energy flux (L2 PAS-EFLUX).
        Reads CDF files directly since pyspedas.projects.solo.swa() doesn't load into pytplot.
        
        Returns:
            {
                "time": list of datetime,
                "eflux": 2D array [ntimes, nenergies],
                "energy": 1D array [nenergies] (energy bin centers)
            }
            or None on failure.
        """
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"), t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]

            files = pyspedas.projects.solo.swa(
                trange=trange,
                datatype="pas-eflux",
                level="l2",
                time_clip=True,
                downloadonly=True
            )
            
            if not files:
                print("[INFO] No Solar Orbiter SWA energy flux files were available for the requested interval.")
                return None

            print(f"[INFO] Loading {len(files)} Solar Orbiter SWA energy flux files.")
            
            all_times = []
            all_eflux = []
            energy_bins = None

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    epoch = cdf.varget('Epoch')
                    times_dt = cdflib.cdfepoch.to_datetime(epoch)
                    eflux_data = cdf.varget('eflux')
                    
                    if energy_bins is None:
                        energy_bins = cdf.varget('Energy')
                    
                    all_times.extend(times_dt)
                    all_eflux.extend(eflux_data)
                    
                except Exception as e:
                    print(f"[WARN] Could not read SWA energy flux file {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_times:
                print("[WARN] SWA energy flux files were found, but no readable data were extracted.")
                return None

            times_array = ensure_datetime_array(all_times)
            eflux_array = np.array(all_eflux, dtype=float)
            
            sort_idx = np.argsort([t.timestamp() for t in times_array])
            times_array = times_array[sort_idx]
            eflux_array = eflux_array[sort_idx]
            
            print(f"[INFO] Loaded Solar Orbiter SWA energy flux data ({len(times_array)} times, {len(energy_bins)} energy bins).")
            
            return {
                "time": times_array,
                "eflux": eflux_array,
                "energy": energy_bins
            }

        except Exception as e:
            print(f"[WARN] Solar Orbiter SWA energy flux loading failed: {e}")
            return None
    
    @staticmethod
    def load_kyoto_dst(t0_dt, t1_dt):
        """
        Load Dst index from OMNI2 hourly data.
        
        CRITICAL: Dst is in hourly files (datatype='1hour'), NOT in HRO!
        Files contain ~6 months of data, so we time-clip after loading.
        """
        try:
            # Simple date format (no time component) for hourly data
            trange = [t0_dt.strftime("%Y-%m-%d"),
                    t1_dt.strftime("%Y-%m-%d")]
            
            # Use hourly data, NOT HRO
            files = pyspedas.omni.data(
                trange=trange,
                datatype='1hour',
                time_clip=True,
                downloadonly=True
            )
            
            if not files:
                return None

            all_times = []
            all_dst = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info = cdf.cdf_info()
                    

                    vars_list = cdf_info.zVariables + cdf_info.rVariables
                    
                    # Try multiple possible names (DST is most common)
                    dst_var = None
                    for candidate in ['DST', 'Dst', 'dst', 'SYM_H']:
                        if candidate in vars_list:
                            dst_var = candidate
                            break
                    
                    if dst_var and 'Epoch' in vars_list:
                        epoch = cdf.varget('Epoch')
                        dst_data = cdf.varget(dst_var)
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        
                        # (files contain ~6 months, not just requested period)
                        times_np = np.array(times_dt)
                        mask_time = (times_np >= np.datetime64(t0_dt)) & (times_np <= np.datetime64(t1_dt))
                        
                        if np.any(mask_time):
                            all_times.extend(times_np[mask_time])
                            all_dst.append(dst_data[mask_time])
                            
                except Exception as e:
                    print(f"[WARN] Failed to read Dst data from {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_dst:
                return None

            times_array = ensure_datetime_array(all_times)
            dst_array = np.hstack(all_dst).astype(float)
            
            # Clean sentinels
            mask_bad = (
                (~np.isfinite(dst_array)) |
                (np.abs(dst_array) >= 9999)
            )
            dst_array[mask_bad] = np.nan

            print(f"[INFO] Loaded Dst data ({np.sum(~np.isnan(dst_array))} valid points).")
            
            return {
                "time": times_array,
                "dst":  dst_array
            }

        except Exception as e:
            print(f"[WARN] Dst loading failed: {e}")
            return None
    
    @staticmethod
    def load_kp_index(t0_dt, t1_dt):
        """
        Load Kp index from OMNI2 hourly data.
        
        CRITICAL: 
        - Kp is in hourly files (datatype='1hour'), NOT in HRO
        - Kp is stored as Kp * 10 (e.g., Kp=1.3 to stored as 13)
        - Must divide by 10 to get actual Kp values
        """
        try:
            # Simple date format for hourly data
            trange = [t0_dt.strftime("%Y-%m-%d"),
                    t1_dt.strftime("%Y-%m-%d")]
            
            # Use hourly data, NOT HRO
            files = pyspedas.omni.data(
                trange=trange,
                datatype='1hour',
                time_clip=True,
                downloadonly=True
            )
            
            if not files:
                return None

            all_times = []
            all_kp = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info = cdf.cdf_info()
                    
                    # Kp is stored in rVariables for this OMNI product
                    vars_list = cdf_info.zVariables + cdf_info.rVariables
                    
                    kp_var = None
                    for candidate in ['KP', 'Kp', 'kp', 'KP_INDEX', 'Kp_index']:
                        if candidate in vars_list:
                            kp_var = candidate
                            break
                    
                    if kp_var and 'Epoch' in vars_list:
                        epoch = cdf.varget('Epoch')
                        kp_data = cdf.varget(kp_var)
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        
                        times_np = np.array(times_dt)
                        mask_time = (times_np >= np.datetime64(t0_dt)) & (times_np <= np.datetime64(t1_dt))
                        
                        if np.any(mask_time):
                            all_times.extend(times_np[mask_time])
                            all_kp.append(kp_data[mask_time])
                            
                except Exception as e:
                    print(f"[WARN] Failed to read Kp data from {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_kp:
                return None

            times_array = ensure_datetime_array(all_times)
            kp_raw = np.hstack(all_kp).astype(float)
            
            mask_bad = (
                (~np.isfinite(kp_raw)) |
                (kp_raw < 0) |
                (kp_raw > 90) |
                (np.abs(kp_raw - 999) < 1) |
                (np.abs(kp_raw - 9999) < 1)
            )
            
            kp_raw[mask_bad] = np.nan
            
            # OMNI stores Kp multiplied by 10; divide by 10 to recover physical values
            kp_actual = kp_raw / 10.0
            
            n_valid = np.sum(~np.isnan(kp_actual))
            print(f"[INFO] Loaded Kp data ({n_valid} valid points; range {np.nanmin(kp_actual):.1f} to {np.nanmax(kp_actual):.1f}).")

            return {
                "time": times_array,
                "kp":   kp_actual
            }

        except Exception as e:
            print(f"[WARN] Kp loading failed: {e}")
            return None
        
    @staticmethod
    def interpolate_to_mag_time(mag_data, swa_moments):
        """Interpolate SWA moments onto the MAG time grid."""
        mag_t = np.array([t.timestamp() for t in mag_data["time"]])
        swa_t = np.array([t.timestamp() for t in swa_moments["time"]])

        if mag_t.size == 0 or swa_t.size == 0:
            print("[WARN] Cannot interpolate plasma data to MAG timestamps: empty time arrays.")
            return None

        if mag_t[0] > swa_t[-1] or mag_t[-1] < swa_t[0]:
            print("[WARN] Cannot interpolate plasma data to MAG timestamps: no time overlap.")
            return None

        V_interp = np.zeros((len(mag_t), 3))
        for i in range(3):
            f = interp1d(
                swa_t, swa_moments["V"][:, i], kind="linear",
                bounds_error=False, fill_value=np.nan
            )
            V_interp[:, i] = f(mag_t)

        f_n = interp1d(
            swa_t, swa_moments["n"], kind="linear",
            bounds_error=False, fill_value=np.nan
        )
        n_interp = f_n(mag_t)

        valid = ~(np.isnan(V_interp).any(axis=1) | np.isnan(n_interp))
        if not valid.any():
            print("[WARN] Interpolation to MAG timestamps produced no valid plasma samples.")
            return None

        # Retain only time samples with valid interpolated plasma values
        mag_times_valid = [mag_data["time"][i] for i in range(len(mag_t)) if valid[i]]
        B_valid         = mag_data["B"][valid]
        V_valid         = V_interp[valid]
        n_valid         = n_interp[valid]

        # Check interpolated plasma values for possible unit or scaling issues
        if B_valid.ndim == 1:
            Bmag = np.abs(B_valid)
        else:
            Bmag = np.linalg.norm(B_valid, axis=1)
        Vmag = np.linalg.norm(V_valid, axis=1)

        if np.nanmax(Vmag) > 2_000:
            print("[WARN] Interpolated SWA speed exceeds expected range; units may be incorrect.")
        if np.nanmax(n_valid) > 1_000:
            print("[WARN] Interpolated SWA density exceeds expected range; units or flags may be incorrect.")

        return {
            "time": mag_times_valid,
            "B":    B_valid,
            "V":    V_valid,
            "n":    n_valid,
        }

    @staticmethod
    def transform_rtn_to_gse(B_rtn, spacecraft_positions, times_dt, epoch_date):
        N = len(B_rtn)
        B_gse = np.zeros_like(B_rtn)

        for i in range(N):
            sc_pos = spacecraft_positions[i]
            earth_pos = SunEarthLineAnalyzer.earth_position_at_time(times_dt[i], epoch_date)

            sc_xy = sc_pos[:2]
            theta_sc = np.arctan2(sc_xy[1], sc_xy[0])
            R_hat = np.array([np.cos(theta_sc), np.sin(theta_sc), 0])
            T_hat = np.array([-np.sin(theta_sc), np.cos(theta_sc), 0])
            N_hat = np.array([0, 0, 1])

            earth_xy = earth_pos[:2]
            theta_earth = np.arctan2(earth_xy[1], earth_xy[0])
            X_hat = np.array([-np.cos(theta_earth), -np.sin(theta_earth), 0])
            Y_hat = np.array([np.sin(theta_earth), -np.cos(theta_earth), 0])
            Z_hat = np.array([0, 0, 1])

            R = np.array([
                [np.dot(X_hat, R_hat), np.dot(X_hat, T_hat), np.dot(X_hat, N_hat)],
                [np.dot(Y_hat, R_hat), np.dot(Y_hat, T_hat), np.dot(Y_hat, N_hat)],
                [np.dot(Z_hat, R_hat), np.dot(Z_hat, T_hat), np.dot(Z_hat, N_hat)],
            ])

            B_gse[i] = R @ B_rtn[i]

        return B_gse

    @staticmethod
    def average_to_3hour_bins(times_dt, values):
        if len(times_dt) == 0:
            return {"time": [], "values": np.array([])}

        start_time = times_dt[0].replace(hour=(times_dt[0].hour // 3) * 3,
                                         minute=0, second=0, microsecond=0)
        end_time = times_dt[-1]

        bin_times = []
        bin_values = []

        current_bin_start = start_time
        while current_bin_start <= end_time:
            current_bin_end = current_bin_start + timedelta(hours=3)
            mask = [(current_bin_start <= t < current_bin_end) for t in times_dt]
            if any(mask):
                bin_values.append(np.mean(values[mask]))
                bin_times.append(current_bin_start + timedelta(hours=1.5))
            current_bin_start = current_bin_end

        return {"time": bin_times, "values": np.array(bin_values)}

    @staticmethod
    def predict_dst_obrien(B_rtn, V_rtn, n, times_dt):
        a        = -4.4
        b        = 0.5
        tau_main = 7.7 * 3600

        V_mag = np.linalg.norm(V_rtn, axis=1)
        Bn    = B_rtn[:, 2]
        Bs    = np.where(Bn < 0, -Bn, 0)
        Ey    = V_mag * Bs / 1000.0

        Q      = np.where(Ey > b, a * (Ey - b), 0.0)
        Q_sec  = Q / 3600.0

        m_p  = 1.67e-27
        n_si = n * 1e6
        V_si = V_mag * 1000.0
        Pdyn = m_p * n_si * V_si**2 * 1e9

        P0            = 2.0
        P_correction  = 7.26 * np.sqrt(Pdyn) - 7.26 * np.sqrt(P0)

        tsec = np.diff([t.timestamp() for t in times_dt])
        tsec = np.insert(tsec, 0, tsec[0] if len(tsec) > 0 else 60.0)

        Dst = np.zeros(len(times_dt))
        for i in range(1, len(times_dt)):
            dDst  = (Q_sec[i] - Dst[i - 1] / tau_main) * tsec[i]
            Dst[i] = Dst[i - 1] + dDst

        Dst_pred = Dst + P_correction
        return {"time": times_dt, "dst_predicted": Dst_pred}

    @staticmethod
    def predict_kp_newell(B_rtn, V_rtn, times_dt, spacecraft_positions, epoch_date):
        B_gse = SoloDataLoader.transform_rtn_to_gse(B_rtn, spacecraft_positions,
                                                    times_dt, epoch_date)

        B_Y = B_gse[:, 1]
        B_Z = B_gse[:, 2]
        theta_clock = np.arctan2(B_Y, B_Z)
        B_T = np.sqrt(B_Y**2 + B_Z**2)

        V_mag = np.linalg.norm(V_rtn, axis=1)

        sin_half_theta = np.abs(np.sin(theta_clock / 2.0))
        V_safe = np.maximum(V_mag, 1.0)
        B_T_safe = np.maximum(B_T, 0.1)

        dPhi_dt = (V_safe**(4.0/3.0)) * (B_T_safe**(2.0/3.0)) * (sin_half_theta**8) * 1e-4

        a_kp = 0.3
        b_kp = 2.5

        Kp_raw = a_kp + b_kp * np.log10(dPhi_dt + 1e-10)
        Kp_instantaneous = np.clip(Kp_raw, 0, 9)

        Kp_binned = SoloDataLoader.average_to_3hour_bins(times_dt, Kp_instantaneous)

        return {
            "time": Kp_binned["time"],
            "kp_predicted": Kp_binned["values"]
        }

    @staticmethod
    def apply_propagation_delay(common_data, dst_measured, spacecraft_ephemeris, epoch_date):
        if not dst_measured or not common_data:
            return None

        sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
        sc_positions = spacecraft_ephemeris["positions"]

        common_times_sec = np.array([t.timestamp() for t in common_data["time"]])

        sc_pos_interp = np.zeros((len(common_times_sec), 3))
        for i in range(3):
            f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind="linear",
                             bounds_error=False, fill_value=np.nan)
            sc_pos_interp[:, i] = f_pos(common_times_sec)

        earth_positions = np.array([
            SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
            for t in common_data["time"]
        ])

        distances = np.linalg.norm(sc_pos_interp - earth_positions, axis=1)
        V_mag = np.linalg.norm(common_data["V"], axis=1)
        propagation_delays_sec = distances / V_mag
        propagation_delays_hours = propagation_delays_sec / 3600.0

        dst_times_sec = np.array([t.timestamp() for t in dst_measured["time"]])
        dst_values = dst_measured["dst"].flatten() if len(dst_measured["dst"].shape) > 1 else dst_measured["dst"]

        f_dst = interp1d(dst_times_sec, dst_values, kind="linear",
                         bounds_error=False, fill_value=np.nan)

        shifted_times_sec = common_times_sec + propagation_delays_sec
        dst_shifted = f_dst(shifted_times_sec)

        valid = ~np.isnan(dst_shifted)
        if not valid.any():
            return None

        return {
            "time": [common_data["time"][i] for i in range(len(valid)) if valid[i]],
            "dst_shifted": dst_shifted[valid],
            "propagation_delay_hours": propagation_delays_hours[valid]
        }

    @staticmethod
    def apply_propagation_delay_kp(common_data, kp_measured, spacecraft_ephemeris, epoch_date):
        if not kp_measured or not common_data:
            return None

        sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
        sc_positions = spacecraft_ephemeris["positions"]

        common_times_sec = np.array([t.timestamp() for t in common_data["time"]])

        sc_pos_interp = np.zeros((len(common_times_sec), 3))
        for i in range(3):
            f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind="linear",
                             bounds_error=False, fill_value=np.nan)
            sc_pos_interp[:, i] = f_pos(common_times_sec)

        earth_positions = np.array([
            SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
            for t in common_data["time"]
        ])

        distances = np.linalg.norm(sc_pos_interp - earth_positions, axis=1)
        V_mag = np.linalg.norm(common_data["V"], axis=1)
        propagation_delays_sec = distances / V_mag

        kp_times_sec = np.array([t.timestamp() for t in kp_measured["time"]])
        kp_values = kp_measured["kp"].flatten() if len(kp_measured["kp"].shape) > 1 else kp_measured["kp"]

        f_kp = interp1d(kp_times_sec, kp_values, kind="linear",
                        bounds_error=False, fill_value=np.nan)

        shifted_times_sec = common_times_sec + propagation_delays_sec
        kp_shifted = f_kp(shifted_times_sec)

        valid = ~np.isnan(kp_shifted)
        if not valid.any():
            return None

        return {
            "time": [common_data["time"][i] for i in range(len(valid)) if valid[i]],
            "kp_shifted": kp_shifted[valid]
        }

    @staticmethod
    def preload_all(events, spacecraft_ephemeris, epoch_date, icme_catalogue, window_hours=72):
        all_data = {}
        for event in tqdm(events, desc="Loading", ncols=80):
            t_ev  = event["time"]
            t0_dt = t_ev - timedelta(hours=window_hours)
            t1_dt = t_ev + timedelta(hours=window_hours)

            print(f"\n  {t_ev}:")
            print(f"    In-situ window: {t0_dt} to {t1_dt}")

            mag  = SoloDataLoader.load_mag(t0_dt, t1_dt)
            swam = SoloDataLoader.load_swa_moments(t0_dt, t1_dt)
            swae = SoloDataLoader.load_swa_eflux(t0_dt, t1_dt)

            ext_t0 = t0_dt
            ext_t1 = t1_dt + timedelta(days=5)

            dst = SoloDataLoader.load_kyoto_dst(ext_t0, ext_t1)
            kp  = SoloDataLoader.load_kp_index(ext_t0, ext_t1)
            l1  = L1DataLoader.load_omni_data(ext_t0, ext_t1)

            dst_pred = None
            dst_shifted = None
            kp_pred = None
            kp_shifted = None
            l1_shifted = None
            l1_shifted_ballistic = None
            common = None

            if mag and swam and ("V" in swam) and ("n" in swam):
                common = SoloDataLoader.interpolate_to_mag_time(mag, swam)
                if common:
                    dst_pred = SoloDataLoader.predict_dst_obrien(
                        common["B"], common["V"], common["n"], common["time"]
                    )
                    dst_shifted = SoloDataLoader.apply_propagation_delay(
                        common, dst, spacecraft_ephemeris, epoch_date
                    )

                    sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
                    sc_positions = spacecraft_ephemeris["positions"]
                    common_times_sec = np.array([t.timestamp() for t in common["time"]])

                    sc_pos_interp = np.zeros((len(common_times_sec), 3))
                    for i in range(3):
                        f_pos = interp1d(sc_times_sec, sc_positions[:, i],
                                        kind="linear", bounds_error=False,
                                        fill_value=np.nan)
                        sc_pos_interp[:, i] = f_pos(common_times_sec)

                    kp_pred = SoloDataLoader.predict_kp_newell(
                        common["B"], common["V"], common["time"],
                        sc_pos_interp, epoch_date
                    )
                    kp_shifted = SoloDataLoader.apply_propagation_delay_kp(
                        common, kp, spacecraft_ephemeris, epoch_date
                    )


                    if l1:
                        l1_shifted_ballistic = BallisticPropagation.apply_ballistic_shift_to_l1(
                            common, l1, spacecraft_ephemeris, epoch_date
                        )
                        l1_shifted = l1_shifted_ballistic
                            
        
            # Back-propagate ICME arrival times from Earth to spacecraft
            icmes_ballistic = None

            if icme_catalogue:
                print(f"\n[PRELOAD] Checking ICME catalogue for event at {t_ev}")
                icmes_ballistic = CME_Catalogue.backpropagate_icmes(
                    event, spacecraft_ephemeris, epoch_date, icme_catalogue,
                    common_data=common,
                )
                
            all_data[t_ev] = {
                "mag":                mag,
                "swa":                swae,
                "swa_moments":        swam,
                "dst_measured":       dst,
                "dst_predicted":      dst_pred,
                "dst_shifted":        dst_shifted,
                "kp_measured":        kp,
                "kp_predicted":       kp_pred,
                "kp_shifted":         kp_shifted,
                "l1_data":            l1,
                "l1_shifted":         l1_shifted,
                "l1_shifted_ballistic": l1_shifted_ballistic,
                "icmes_ballistic":    icmes_ballistic,
            }
        return all_data


# ------------------------ STEREO in-situ loaders -------------------------- #

class STEREODataLoader:
    """Data loader for STEREO-A and STEREO-B spacecraft."""
    
    @staticmethod
    def load_mag(spacecraft, t0_dt, t1_dt):
        """
        Load STEREO IMPACT MAG data with decimation.
        
        Args:
            spacecraft: "STEREO-A" or "STEREO-B"
            t0_dt: Start datetime
            t1_dt: End datetime
            
        Returns:
            {"time": datetime array, "B": [N×3] RTN B-field} or None
        """
        try:
            probe = 'a' if spacecraft == "STEREO-A" else 'b'
            
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                     t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            
            # Download files (no level/coord args for STEREO)
            files = pyspedas.projects.stereo.mag(trange=trange,probe=probe,time_clip=True,downloadonly=True)
            
            if not files:
                print(f"[WARN] No {spacecraft} MAG files were available for the requested interval.")
                return None
            
            all_times = []
            all_B = []
            
            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    vars_list = cdf.cdf_info().zVariables
                    
                    # STEREO variable names
                    epoch_var = next((v for v in ['Epoch', 'EPOCH', 'epoch'] if v in vars_list), None)
                    b_var = next((v for v in ['BFIELD', 'B_RTN', 'BRTN'] if v in vars_list), None)
                    
                    if not epoch_var or not b_var:
                        continue
                    
                    epoch = cdf.varget(epoch_var)
                    b_data = cdf.varget(b_var)
                    
                    n = len(epoch)
                    if n == 0:
                        continue
                    
                    # Decimate per file
                    step = max(1, n // MAG_TARGET_POINTS_PER_DAY)
                    idx = np.arange(0, n, step, dtype=int)
                    
                    epoch_sel = epoch[idx]
                    b_sel = b_data[idx, :] if b_data.ndim > 1 else b_data[idx]
                    
                    times_dt = cdflib.cdfepoch.to_datetime(epoch_sel)
                    all_times.extend(times_dt)
                    all_B.append(b_sel)
                    
                    # Check per-file field magnitude range
                    if b_sel.ndim == 1:
                        bmag = np.abs(b_sel)
                    else:
                        bmag = np.linalg.norm(b_sel, axis=1)
                
                except Exception as e:
                    print(f"[WARN] Failed to read {spacecraft} MAG file {os.path.basename(cdf_file)}: {e}")
                    continue
            
            if not all_B:
                print(f"[WARN] No valid {spacecraft} MAG data were extracted from the downloaded files.")
                return None
            
            B_combined = np.vstack(all_B).astype(float)
            times_arr = ensure_datetime_array(all_times)

            if B_combined.ndim == 2 and B_combined.shape[1] == 4:
                B_combined = B_combined[:, :3] 

            mask_bad = (
                (~np.isfinite(B_combined)) |
                (np.abs(B_combined) >= 1e30) |
                (np.abs(B_combined) > 500.0)
            )
            B_combined[mask_bad] = np.nan
            
            sort_idx = np.argsort([t.timestamp() for t in times_arr])
            times_arr = times_arr[sort_idx]
            B_combined = B_combined[sort_idx]
            

            if B_combined.ndim == 1:
                Bmag = np.abs(B_combined)
            else:
                Bmag = np.linalg.norm(B_combined, axis=1)
            
            print(f"[INFO] Loaded {spacecraft} MAG data ({len(times_arr)} samples after decimation).")
            
            if np.nanmax(Bmag) > 1e3:
                print(f"[WARN] {spacecraft} MAG field magnitude exceeds the expected range; units or scaling may be incorrect.")
            
            return {"time": times_arr, "B": B_combined}
        
        except Exception as e:
            print(f"[WARN] {spacecraft} MAG loading failed: {e}")
            return None
    
    @staticmethod
    def load_plasma(spacecraft, t0_dt, t1_dt):
        """
        Load STEREO PLASTIC plasma moments (L2, 1-minute cadence) using flexible
        variable-name matching across supported file variants.
        
        Key features:
        - Tries multiple candidate variable names for density, velocity, and temperature
        - Converts scalar speed measurements into a three-component placeholder vector
        - Detects temperature values stored in Kelvin and converts them to eV
        
        Args:
            spacecraft: "STEREO-A" or "STEREO-B"
            t0_dt: Start datetime
            t1_dt: End datetime
            
        Returns:
            {"time": datetime array, "n": density [cm^-3], "V": [N×3] velocity [km/s], "T": temperature [eV]} or None
        """
        try:
            probe = 'a' if spacecraft == "STEREO-A" else 'b'
            
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                    t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            
            files = pyspedas.projects.stereo.plastic(
                trange=trange,
                probe=probe,
                time_clip=True,
                downloadonly=True
            )
            
            if not files:
                print(f"[WARN] No PLASTIC files were returned by pyspedas for {spacecraft}.")
                return None
            
            print(f"[INFO] Loading {spacecraft} PLASTIC plasma data from {len(files)} files.")
            
            all_times = []
            all_n = []
            all_V = []
            all_T = []
            
            n_candidates = ['proton_number_density', 'Np', 'N', 'n_p', 'density']
            v_candidates = ['proton_bulk_speed', 'Vp_RTN', 'Vp', 'V', 'v_p', 'velocity']
            t_candidates = ['proton_temperature', 'Tp', 'T', 't_p', 'temperature']
            
            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    cdf_info = cdf.cdf_info()
                    vars_list = cdf_info.zVariables
                    
                    # Find epoch
                    epoch_var = next((v for v in ['epoch', 'Epoch', 'EPOCH'] if v in vars_list), None)
                    if not epoch_var:
                        print(f"[WARN] No epoch variable was found in a {spacecraft} PLASTIC file; skipping file.")
                        continue
                    
                    epoch = cdf.varget(epoch_var)
                    times_dt = cdflib.cdfepoch.to_datetime(epoch)

                    # Find density
                    n_data = None
                    for n_var in n_candidates:
                        if n_var in vars_list:
                            try:
                                n_data = cdf.varget(n_var)
                                break
                            except Exception as e:

                                continue
                    
                    
                    # Find velocity
                    v_data = None
                    for v_var in v_candidates:
                        if v_var in vars_list:
                            try:
                                v_data = cdf.varget(v_var)
                                
                                # Handle 1D velocity (scalar speed)
                                if v_data.ndim == 1:
                                    v_data = np.column_stack([v_data, np.zeros(len(v_data)), np.zeros(len(v_data))])
                                
                                break
                            except Exception as e:
                                continue
                    
                    
                    # Find temperature
                    t_data = None
                    for t_var in t_candidates:
                        if t_var in vars_list:
                            try:
                                t_data = cdf.varget(t_var)
                            except Exception as e:
                                continue
                    
                    # Accumulate data
                    if n_data is not None or v_data is not None or t_data is not None:
                        n_points = len(times_dt)
                        all_times.extend(times_dt)
                        
                        if n_data is not None:
                            all_n.extend(n_data)
                        else:
                            all_n.extend([np.nan] * n_points)
                        
                        if v_data is not None:
                            all_V.extend(v_data)
                        else:
                            all_V.extend([[np.nan, np.nan, np.nan]] * n_points)
                        
                        if t_data is not None:
                            all_T.extend(t_data)
                        else:
                            all_T.extend([np.nan] * n_points)
                        

                    else:
                        print(f"[WARN] No plasma data were found in file {os.path.basename(cdf_file)}.")
                
                except Exception as e:
                    print(f"[WARN] Failed to read {spacecraft} PLASTIC file {os.path.basename(cdf_file)}: {e}")
                    continue
            
            if not all_times:
                print(f"[WARN] No {spacecraft} PLASTIC plasma data were loaded.")
                return None
            
            print(f"[INFO] Loaded {spacecraft} PLASTIC plasma data ({len(all_times)} samples).")
            
            # Convert to arrays
            times_arr = ensure_datetime_array(all_times)
            n_arr = np.array(all_n, dtype=float)
            v_arr = np.array(all_V, dtype=float)
            t_arr = np.array(all_T, dtype=float)
        
            # Density cleaning
            n_arr[(~np.isfinite(n_arr)) | (n_arr < 0) | (n_arr > 1000) | (n_arr == 9999) | (n_arr == 99999)] = np.nan
            
            # Velocity cleaning
            v_arr[(~np.isfinite(v_arr)) | (np.abs(v_arr) > 3000) | (v_arr == 9999) | (v_arr == 99999)] = np.nan
            
            # Temperature cleaning + Kelvin to eV conversion
            t_arr[(~np.isfinite(t_arr)) | (t_arr < 0) | (t_arr > 1e7) | (t_arr == 9999) | (t_arr == 99999)] = np.nan
            
            # Auto-detect Kelvin and convert to eV
            valid_t = t_arr[np.isfinite(t_arr)]
            if len(valid_t) > 0:
                median_t = np.median(valid_t)
                if median_t > 1000:
                    print(f"[INFO] Converted {spacecraft} PLASTIC temperature from Kelvin to eV.")
                    t_arr = t_arr / 11604.5
            
            sort_idx = np.argsort([t.timestamp() for t in times_arr])
            times_arr = times_arr[sort_idx]
            n_arr = n_arr[sort_idx]
            v_arr = v_arr[sort_idx]
            t_arr = t_arr[sort_idx]
            
            return {
                "time": times_arr,
                "n": n_arr,
                "V": v_arr,
                "T": t_arr
            }
        
        except Exception as e:
            print(f"[WARN] {spacecraft} PLASTIC plasma loading failed: {e}")
            return None

    @staticmethod
    def load_eflux(spacecraft, t0_dt, t1_dt):
        """
        Load STEREO PLASTIC energy flux spectrogram data, if available.
        
        Note:
            STEREO PLASTIC L2 files often do not contain full energy-flux spectrograms.
            In those cases, the absence of energy-flux data is expected and does not
            indicate a pipeline failure.
            
        Returns:
            {"time": datetime array, "eflux": [N×E] array, "energy": energy bins} or None
        """
        try:
            probe = 'a' if spacecraft == "STEREO-A" else 'b'
            
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                    t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            
            files = pyspedas.projects.stereo.plastic(
                trange=trange,
                probe=probe,
                time_clip=True,
                downloadonly=True
            )
            
            if not files:
                print(f"[INFO] No {spacecraft} PLASTIC energy flux files were available for the requested interval.")
                return None
            
            all_times = []
            all_eflux = []
            energy_bins = None
            
            eflux_candidates = ['eflux', 'EFLUX', 'diff_en_fluxes', 'energy_flux']
            energy_candidates = ['energy', 'Energy', 'ENERGY', 'energy_vals']
            
            found_any = False
            
            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    vars_list = cdf.cdf_info().zVariables
                    
                    # Look for eflux-like variables
                    eflux_var = None
                    for candidate in eflux_candidates:
                        if candidate in vars_list:
                            eflux_var = candidate
                            break
                    
                    if eflux_var:
                        found_any = True
                        
                        epoch_var = next((v for v in ['epoch', 'Epoch', 'EPOCH'] if v in vars_list), None)
                        if not epoch_var:
                            continue
                        
                        epoch = cdf.varget(epoch_var)
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        eflux_data = cdf.varget(eflux_var)
                        
                        # Try to get energy bins
                        if energy_bins is None:
                            for e_var in energy_candidates:
                                if e_var in vars_list:
                                    try:
                                        energy_bins = cdf.varget(e_var)
                                        break
                                    except:
                                        continue
                        
                        all_times.extend(times_dt)
                        all_eflux.append(eflux_data)
                
                except Exception as e:
                    print(f"[WARN] Failed while checking {spacecraft} PLASTIC energy flux file {os.path.basename(cdf_file)}: {e}")
                    continue
            
            if not found_any:
                print(f"[INFO] No {spacecraft} PLASTIC energy flux data are available for this interval.")
                return None
            
            if not all_times or not all_eflux:
                print(f"[WARN] {spacecraft} PLASTIC energy flux files were found, but no usable data were extracted.")
                return None
            
            times_arr = ensure_datetime_array(all_times)
            eflux_arr = np.vstack(all_eflux).astype(float)
            
            # Sort by time
            sort_idx = np.argsort([t.timestamp() for t in times_arr])
            times_arr = times_arr[sort_idx]
            eflux_arr = eflux_arr[sort_idx]
            
            print(f"[INFO] Loaded {spacecraft} energy flux data ({len(times_arr)} times, {eflux_arr.shape[1]} energy bins).")
            
            return {
                "time": times_arr,
                "eflux": eflux_arr,
                "energy": energy_bins if energy_bins is not None else np.logspace(2, 4, eflux_arr.shape[1])
            }
        
        except Exception as e:
            print(f"[WARN] Error loading eflux data for {spacecraft}: {e}")
            return None

    @staticmethod
    def interpolate_to_mag_time(mag_data, plasma_data):
        """
        Interpolate PLASTIC plasma moments onto the MAG time grid.
        """
        mag_t = np.array([t.timestamp() for t in mag_data["time"]])
        plasma_t = np.array([t.timestamp() for t in plasma_data["time"]])
        
        if mag_t.size == 0 or plasma_t.size == 0:
            print("[WARN] Cannot interpolate PLASTIC plasma data to MAG timestamps: empty time arrays.")
            return None
        
        if mag_t[0] > plasma_t[-1] or mag_t[-1] < plasma_t[0]:
            print("[WARN] Cannot interpolate PLASTIC plasma data to MAG timestamps: no time overlap.")
            return None
        
        
        V_interp = np.zeros((len(mag_t), 3))
        for i in range(3):
            f = interp1d(plasma_t, plasma_data["V"][:, i], kind="linear",
                        bounds_error=False, fill_value=np.nan)
            V_interp[:, i] = f(mag_t)
        
        f_n = interp1d(plasma_t, plasma_data["n"], kind="linear",
                      bounds_error=False, fill_value=np.nan)
        n_interp = f_n(mag_t)
        
        valid = ~(np.isnan(V_interp).any(axis=1) | np.isnan(n_interp))
        if not valid.any():
            print("[WARN] Interpolation of PLASTIC plasma data to MAG timestamps produced no valid samples.")
            return None
        
        mag_times_valid = [mag_data["time"][i] for i in range(len(mag_t)) if valid[i]]
        B_valid = mag_data["B"][valid]
        V_valid = V_interp[valid]
        n_valid = n_interp[valid]
        

        if B_valid.ndim == 1:
            Bmag = np.abs(B_valid)
        else:
            Bmag = np.linalg.norm(B_valid, axis=1)
        Vmag = np.linalg.norm(V_valid, axis=1)
    
        
        if np.nanmax(Vmag) > 2_000:
            print("[WARN] Interpolated PLASTIC speed exceeds expected range; units may be incorrect.")
        if np.nanmax(n_valid) > 1_000:
            print("[WARN] Interpolated PLASTIC density exceeds expected range; units or flags may be incorrect.")

        
        return {
            "time": mag_times_valid,
            "B": B_valid,
            "V": V_valid,
            "n": n_valid,
        }
    
    @staticmethod
    def preload_all(spacecraft, events, spacecraft_ephemeris, epoch_date, icme_catalogue, window_hours=72):
        """
        Preload all STEREO data for given events.
        Similar to SoloDataLoader.preload_all() but for STEREO spacecraft.
        """
        all_data = {}
        for event in tqdm(events, desc=f"Loading {spacecraft}", ncols=80):
            t_ev = event["time"]
            t0_dt = t_ev - timedelta(hours=window_hours)
            t1_dt = t_ev + timedelta(hours=window_hours)
            
            mag = STEREODataLoader.load_mag(spacecraft, t0_dt, t1_dt)
            plasma = STEREODataLoader.load_plasma(spacecraft, t0_dt, t1_dt)
            eflux = STEREODataLoader.load_eflux(spacecraft, t0_dt, t1_dt)
            
            ext_t0 = t0_dt
            ext_t1 = t1_dt + timedelta(days=5)
            
            dst = SoloDataLoader.load_kyoto_dst(ext_t0, ext_t1)
            kp = SoloDataLoader.load_kp_index(ext_t0, ext_t1)
            l1 = L1DataLoader.load_omni_data(ext_t0, ext_t1)
            
            dst_pred = None
            dst_shifted = None
            kp_pred = None
            kp_shifted = None
            l1_shifted = None
            l1_shifted_ballistic = None
            common = None
            
            if mag and plasma and ("V" in plasma) and ("n" in plasma):
                common = STEREODataLoader.interpolate_to_mag_time(mag, plasma)
                if common:
                    dst_pred = SoloDataLoader.predict_dst_obrien(
                        common["B"], common["V"], common["n"], common["time"]
                    )
                    dst_shifted = SoloDataLoader.apply_propagation_delay(
                        common, dst, spacecraft_ephemeris, epoch_date
                    )
                    
                    sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
                    sc_positions = spacecraft_ephemeris["positions"]
                    common_times_sec = np.array([t.timestamp() for t in common["time"]])
                    
                    sc_pos_interp = np.zeros((len(common_times_sec), 3))
                    for i in range(3):
                        f_pos = interp1d(sc_times_sec, sc_positions[:, i],
                                        kind="linear", bounds_error=False,
                                        fill_value=np.nan)
                        sc_pos_interp[:, i] = f_pos(common_times_sec)
                    
                    kp_pred = SoloDataLoader.predict_kp_newell(
                        common["B"], common["V"], common["time"],
                        sc_pos_interp, epoch_date
                    )
                    kp_shifted = SoloDataLoader.apply_propagation_delay_kp(
                        common, kp, spacecraft_ephemeris, epoch_date
                    )
                    
                    if l1:
                        l1_shifted_ballistic = BallisticPropagation.apply_ballistic_shift_to_l1(
                            common, l1, spacecraft_ephemeris, epoch_date
                        )
                        l1_shifted = l1_shifted_ballistic
            
            # Back-propagate ICMEs using ballistic propagation
            icmes_ballistic = None

            if icme_catalogue:
                print(f"\n[PRELOAD] Checking ICME catalogue for event at {t_ev}")
                icmes_ballistic = CME_Catalogue.backpropagate_icmes(
                    event, spacecraft_ephemeris, epoch_date, icme_catalogue,
                    common_data=common,
                )
                if icmes_ballistic:
                    print(f"[INFO] Ballistic propagation matched {len(icmes_ballistic)} ICME event(s).")
                else:
                    print("[INFO] Ballistic propagation did not match any ICME events.")

            all_data[t_ev] = {
                "mag": mag,
                "swa": eflux,
                "swa_moments": plasma,
                "dst_measured": dst,
                "dst_predicted": dst_pred,
                "dst_shifted": dst_shifted,
                "kp_measured": kp,
                "kp_predicted": kp_pred,
                "kp_shifted": kp_shifted,
                "l1_data": l1,
                "l1_shifted": l1_shifted,
                "l1_shifted_ballistic": l1_shifted_ballistic,
                "icmes_ballistic": icmes_ballistic,
            }
        
        return all_data

    
# ------------------------------- Plotting -------------------------------- #

class Plotter:

    @staticmethod
    def average_to_5min_bins(times_dt, values, is_vector=False):
        """
        Average data to 5-minute bins.

        Converts all times to Unix timestamps (float) internally to avoid
        dtype comparison issues with numpy object arrays of datetime objects.

        Args:
            times_dt: Array-like of datetime objects (may be numpy object array)
            values:   Array of data values (1D scalar or 2D vector [N, 3])
            is_vector: If True, values is [N×3]; if False, 1D scalar array

        Returns:
            {"time": np.ndarray of datetime bin centres,
             "data": np.ndarray of averaged values}
        """
        if len(times_dt) == 0:
            empty_data = np.zeros((0, 3)) if is_vector else np.array([])
            return {"time": np.array([]), "data": empty_data}

        values = np.array(values)

        # Convert all times to Unix timestamps (float64) to avoid object-array
        # comparison issues that cause silent all-False masks
        t_sec = np.array([t.timestamp() for t in times_dt], dtype=float)

        bin_width = 5 * 60.0  # 5 minutes in seconds

        # Align first bin to nearest 5-minute boundary
        t_start = times_dt[0]
        bin_start_sec = datetime(
            t_start.year, t_start.month, t_start.day,
            t_start.hour, (t_start.minute // 5) * 5, 0
        ).timestamp()
        t_end_sec = t_sec[-1]

        bin_times = []
        bin_values = []

        current = bin_start_sec
        while current <= t_end_sec:
            next_bin = current + bin_width
            mask = (t_sec >= current) & (t_sec < next_bin)

            if np.any(mask):
                if is_vector:
                    subset = values[mask]
                    if subset.ndim == 1:
                        subset = subset.reshape(-1, 3)
                    bin_avg = np.nanmean(subset, axis=0)
                else:
                    bin_avg = float(np.nanmean(values[mask]))

                bin_values.append(bin_avg)
                centre_sec = current + bin_width / 2.0
                bin_times.append(datetime.utcfromtimestamp(centre_sec))

            current = next_bin

        if not bin_values:
            empty_data = np.zeros((0, 3)) if is_vector else np.array([])
            return {"time": np.array([]), "data": empty_data}

        data_out = np.array(bin_values)
        if is_vector and data_out.ndim == 1:
            data_out = data_out.reshape(-1, 3)

        return {
            "time": np.array(bin_times),
            "data": data_out
        }
    

    @staticmethod
    def average_spectrogram_to_5min(times_dt, eflux_2d):
        """
        Average 2D spectrogram data to 5-minute bins.
        
        Args:
            times_dt: Array of datetime objects [N]
            eflux_2d: 2D array [N × E] where E is number of energy bins
        
        Returns:
            {"time": averaged times, "eflux": averaged 2D array}
        """
        if len(times_dt) == 0 or eflux_2d.size == 0:
            return {"time": np.array([]), "eflux": np.array([])}
        
        times_dt = np.array(times_dt)
        eflux_2d = np.array(eflux_2d)
        
        start_time = times_dt[0].replace(minute=(times_dt[0].minute // 5) * 5,
                                         second=0, microsecond=0)
        end_time = times_dt[-1]
        
        bin_times = []
        bin_eflux = []
        
        current_bin_start = start_time
        while current_bin_start <= end_time:
            current_bin_end = current_bin_start + timedelta(minutes=5)
            
            mask = (times_dt >= current_bin_start) & (times_dt < current_bin_end)
            
            if np.any(mask):
                # Average along time axis for each energy bin
                bin_avg = np.nanmean(eflux_2d[mask], axis=0)
                bin_eflux.append(bin_avg)
                bin_times.append(current_bin_start + timedelta(minutes=2.5))
            
            current_bin_start = current_bin_end
        
        return {
            "time": np.array(bin_times),
            "eflux": np.array(bin_eflux)
        }

    @staticmethod
    def plot_constellation_overview(events, constellation, output_folder=None):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal")

        ax.plot(0, 0, "o", ms=16, color="#FDB813", mec="#F37021", mew=2, label="Sun", zorder=10)

        theta = np.linspace(0, 2 * np.pi, 1000)
        ax.plot(np.cos(theta), np.sin(theta), "b--", lw=1.5, alpha=0.4, label="Earth orbit")

        colors = ["#E74C3C", "#3498DB", "#2ECC71"]
        for i, sat in enumerate(constellation.satellites):
            orbit = sat["orbit"]["dro_helio"] / AU_KM
            ax.plot(orbit[:, 0], orbit[:, 1], color=colors[i], lw=2, alpha=0.7, label=f"DRO-{sat['id']} (phase {sat['phase_offset']:.2f} rad)")

        sc_colors = {"STEREO-A": "#FF6B6B", "STEREO-B": "#9B59B6", "Solar Orbiter": "#45B7D1"}
        plotted_sc = set()

        for sc_name, sc_events in events.items():
            if not sc_events:
                continue
            color = sc_colors.get(sc_name, "#888888")
            for ev in sc_events:
                label = sc_name if sc_name not in plotted_sc else None
                plotted_sc.add(sc_name)
                pos = ev["spacecraft_pos"][:2] / AU_KM
                ax.plot(pos[0], pos[1], "o", ms=8, color=color, mec="white", mew=1.5, alpha=0.9, label=label)
                text = f"{ev['time'].strftime('%Y-%m-%d')}\nDRO-{ev['dro_id']}"
                ax.annotate(text, xy=(pos[0], pos[1]), xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_title("Hénon DRO Constellation — Crossover Overview")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_folder:
            plot_folder = os.path.join(output_folder, "plots") if SAVE_DATA_CSV else output_folder
            filepath = os.path.join(plot_folder, "00_constellation_overview.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved: {filepath}")
            plt.close(fig)
        
        return fig

    @staticmethod
    def plot_event_detail(event, spacecraft_name, constellation, spacecraft_data, epoch_date, context_days=TRACK_CONTEXT_DAYS, output_folder=None):
        """
        Plot event detail in EARTH-CENTERED ROTATING FRAME.
        Earth is always at (0, -1), Sun at (0, 0).
        Frame rotates with Earth's orbital motion.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal")

        dro_sat = next((sat for sat in constellation.satellites if sat["id"] == event["dro_id"]), None)
        if not dro_sat:
            return None
        
        earth_pos_event = SunEarthLineAnalyzer.earth_position_at_time(event["time"], epoch_date)
        earth_x_event = earth_pos_event[0] / AU_KM
        earth_y_event = earth_pos_event[1] / AU_KM
        
        # Calculate rotation angle to put Earth at (0, -1)
        theta_earth = np.arctan2(earth_y_event, earth_x_event)
        rotation_angle = -theta_earth - np.pi/2
        
        cos_rot = np.cos(rotation_angle)
        sin_rot = np.sin(rotation_angle)
        
        def rotate_xy(x, y):
            """Apply rotation to put Earth at (0, -1)."""
            x_rot = x * cos_rot - y * sin_rot
            y_rot = x * sin_rot + y * cos_rot
            return x_rot, y_rot
        
        
        dro_orbit = dro_sat["orbit"]["dro_helio"] / AU_KM
        dro_x_rot, dro_y_rot = rotate_xy(dro_orbit[:, 0], dro_orbit[:, 1])
        ax.plot(dro_x_rot, dro_y_rot, "r-", lw=2, alpha=0.7, label=f"DRO-{event['dro_id']}")

        
        ax.plot(0, 0, "o", ms=12, color="#FDB813", mec="#F37021", mew=1.5, label="Sun")
        
        
        theta = np.linspace(0, 2 * np.pi, 1000)
        ax.plot(np.cos(theta), np.sin(theta), "b--", lw=1, alpha=0.2, label="Earth orbit")
        
        # Earth at event (should be at (0, -1) after rotation)
        earth_x_rot, earth_y_rot = rotate_xy(earth_x_event, earth_y_event)
        ax.plot(earth_x_rot, earth_y_rot, "o", ms=10, color="blue", mec="white", mew=1.5, label="Earth @ event")

        
        sc_pos = event["spacecraft_pos"][:2] / AU_KM
        half = context_days / 2.0
        t0 = event["time"] - timedelta(days=half)
        t1 = event["time"] + timedelta(days=half)

        times = spacecraft_data["times"]
        pos = spacecraft_data["positions"]
        mask = [(t0 <= t <= t1) for t in times]
        radii = []

        if any(mask):
            sc_xy = (pos[mask] / AU_KM)[:, :2]
            sc_x_rot, sc_y_rot = rotate_xy(sc_xy[:, 0], sc_xy[:, 1])
            ax.plot(sc_x_rot, sc_y_rot, "-", lw=2, alpha=0.9, color="#45B7D1", 
                    label=f"{spacecraft_name} (±{half:.0f} d)")
            radii.extend(np.sqrt(sc_x_rot**2 + sc_y_rot**2).tolist())

            # Plot Earth trajectory over same period
            earth_xy = np.array([SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)[:2] / AU_KM 
                                for t in np.array(times)[mask]])
            earth_traj_x_rot, earth_traj_y_rot = rotate_xy(earth_xy[:, 0], earth_xy[:, 1])
            ax.plot(earth_traj_x_rot, earth_traj_y_rot, "-", lw=2, alpha=0.9, color="gold", 
                    label=f"Earth (±{half:.0f} d)")
            radii.extend(np.sqrt(earth_traj_x_rot**2 + earth_traj_y_rot**2).tolist())
            
            # Plot L1 trajectory
            l1_xy = []
            for t in np.array(times)[mask]:
                epos = SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
                earth_sun_dir = -epos / np.linalg.norm(epos)
                l1_pos = epos + earth_sun_dir * (0.01 * AU_KM)
                l1_xy.append(l1_pos[:2] / AU_KM)
            l1_xy = np.array(l1_xy)
            l1_x_rot, l1_y_rot = rotate_xy(l1_xy[:, 0], l1_xy[:, 1])
            ax.plot(l1_x_rot, l1_y_rot, "-", lw=2, alpha=0.9, color="lime", 
                    label=f"L1 (±{half:.0f} d)")
            radii.extend(np.sqrt(l1_x_rot**2 + l1_y_rot**2).tolist())

        radii.extend(np.linalg.norm(dro_orbit[:, :2], axis=1).tolist())

        
        sc_x_rot, sc_y_rot = rotate_xy(sc_pos[0], sc_pos[1])
        ax.plot(sc_x_rot, sc_y_rot, "*", ms=15, color="gold", mec="black", mew=1.5, 
                label=f"{spacecraft_name} @ event")
        
        earth_sun_dir = -earth_pos_event / np.linalg.norm(earth_pos_event)
        l1_event = (earth_pos_event + earth_sun_dir * (0.01 * AU_KM)) / AU_KM
        l1_x_rot, l1_y_rot = rotate_xy(l1_event[0], l1_event[1])
        ax.plot(l1_x_rot, l1_y_rot, "D", ms=8, color="lime", mec="darkgreen", mew=1.5, label="L1 @ event")
        
        # Plot Sun-Earth line in rotated frame (should be vertical)
        ax.plot([0, earth_x_rot], [0, earth_y_rot], "g--", lw=1.5, alpha=0.5, label="Sun–Earth line")

        
        rmax = max(radii) if radii else max(np.linalg.norm([earth_x_rot, earth_y_rot]), 
                                            np.linalg.norm([sc_x_rot, sc_y_rot]), 1.2)
        span = rmax + 0.10 * rmax

        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xlabel("X [AU] (Earth-Centered Rotating Frame)", fontsize=11)
        ax.set_ylabel("Y [AU] (Earth-Centered Rotating Frame)", fontsize=11)
        ax.set_title(f"{spacecraft_name} — Earth-Centered Rotating Frame — {event['time'].strftime('%Y-%m-%d')}\n" +
                    f"(Earth fixed at (0, -1), frame rotates with Earth's orbit)", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_folder:
            plot_folder = os.path.join(output_folder, "plots") if SAVE_DATA_CSV else output_folder
            safe_sc_name = spacecraft_name.replace(" ", "_").replace("-", "")
            filename = f"helio_{safe_sc_name}_{event['time'].strftime('%Y%m%d')}.png"
            filepath = os.path.join(plot_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved: {filepath}")
            plt.close(fig)
        
        return fig

    @staticmethod
    def plot_event_detail_earth_frame(event, spacecraft_name, constellation, spacecraft_data, epoch_date, context_days=TRACK_CONTEXT_DAYS, output_folder=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal")

        dro_sat = next((sat for sat in constellation.satellites if sat["id"] == event["dro_id"]), None)
        if not dro_sat:
            return None

        dro_1 = constellation.satellites[0]
        dro_orbit_earth = dro_1["orbit"]["dro_earth"]
        

        dro_orbit_earth = dro_orbit_earth.copy()
        dro_orbit_earth[:, 0] = -dro_orbit_earth[:, 0]
        
        ax.plot(dro_orbit_earth[:, 0], dro_orbit_earth[:, 1], "r-", lw=2, alpha=0.7, label="DRO")

        half = context_days / 2.0
        t0 = event["time"] - timedelta(days=half)
        t1 = event["time"] + timedelta(days=half)

        times = spacecraft_data["times"]
        pos_helio = spacecraft_data["positions"]
        mask = [(t0 <= t <= t1) for t in times]
        radii = []

        if any(mask):
            sc_e = []
            for t, pxyz in zip(np.array(times)[mask], pos_helio[mask]):
                epos = SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)  # Earth heliocentric (km)
                rel = pxyz - epos  # Earth-centered translation (km)

                # Compute rotation from the *actual* Sun direction seen from Earth at time t
                sun_rel = -epos  # vector Earth -> Sun in Earth-centered coords
                theta = np.arctan2(sun_rel[1], sun_rel[0])  # angle of Sun direction in XY plane

                c, s = np.cos(-theta), np.sin(-theta)
                x_e = rel[0] * c - rel[1] * s
                y_e = rel[0] * s + rel[1] * c
                sc_e.append([x_e, y_e])

        sc_e = np.array(sc_e) / AU_KM
        
        ax.plot(sc_e[:, 0], sc_e[:, 1], "-", lw=2, color="#45B7D1", alpha=0.9, label=f"{spacecraft_name} (±{half:.0f} d)")
        radii.extend(np.linalg.norm(sc_e, axis=1).tolist())
        
        l1_e = np.array([[+0.01, 0.0] for _ in range(len(np.array(times)[mask]))])
        ax.plot(l1_e[:, 0], l1_e[:, 1], "-", lw=2, color="lime", alpha=0.9, label=f"L1 (±{half:.0f} d)")
        radii.extend(np.abs(l1_e[:, 0]).tolist())

        sc_h = event["spacecraft_pos"]
        e_h  = event["earth_pos"]

        rel_ev = sc_h - e_h

        sun_rel_ev = -e_h
        theta_ev = np.arctan2(sun_rel_ev[1], sun_rel_ev[0])

        c_ev, s_ev = np.cos(-theta_ev), np.sin(-theta_ev)
        sc_x_e = rel_ev[0] * c_ev - rel_ev[1] * s_ev
        sc_y_e = rel_ev[0] * s_ev + rel_ev[1] * c_ev
        sc_e_pt = np.array([sc_x_e, sc_y_e]) / AU_KM
            
        ax.plot(sc_e_pt[0], sc_e_pt[1], "*", ms=15, color="gold", mec="black", mew=1.5, label=f"{spacecraft_name} @ event")
        ax.plot(+0.01, 0.0, "D", ms=8, color="lime", mec="darkgreen", mew=1.5, label="L1 @ event")

        sun_rel = -e_h
        sun_x_e = sun_rel[0] * c_ev - sun_rel[1] * s_ev
        sun_y_e = sun_rel[0] * s_ev + sun_rel[1] * c_ev
        sun_e = np.array([sun_x_e, sun_y_e]) / AU_KM

        ax.plot(sun_e[0], sun_e[1], "o", ms=14, color="#FDB813", mec="#F37021", mew=2, label="Sun @ event", zorder=9)
        radii.append(np.linalg.norm(sun_e))
        ax.plot([0, sun_e[0]], [0, sun_e[1]], "g--", lw=1.5, alpha=0.5, label="Sun–Earth line")

        rmax = max(radii) if radii else 1.2
        span = rmax + 0.10 * rmax

        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xlabel("X [AU] (Earth Frame)")
        ax.set_ylabel("Y [AU] (Earth Frame)")
        ax.set_title(f"{spacecraft_name} — Earth-Centered — {event['time'].strftime('%Y-%m-%d')}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_folder:
            plot_folder = os.path.join(output_folder, "plots") if SAVE_DATA_CSV else output_folder
            safe_sc_name = spacecraft_name.replace(" ", "_").replace("-", "")
            filename = f"earth_{safe_sc_name}_{event['time'].strftime('%Y%m%d')}.png"
            filepath = os.path.join(plot_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            plt.close(fig)
        
        return fig

    @staticmethod
    def plot_all_icme_orbits_earth_frame(all_icme_events, constellation, spacecraft_data_dict, epoch_date, context_days=TRACK_CONTEXT_DAYS, output_folder=None):
        """
        Plot all ICME event orbits on a single Earth-centered plot.
        
        Args:
            all_icme_events: Dictionary {spacecraft: [events]} containing only ICME events
            constellation: DRO constellation
            spacecraft_data_dict: Dictionary {spacecraft: ephemeris_data}
            epoch_date: Epoch datetime
            context_days: Days before/after event to plot
            output_folder: Output folder path
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_aspect("equal")
        
        # Plot Earth at origin
        ax.plot(0, 0, "o", ms=12, color="#4A90E2", mec="white", mew=1.5, label="Earth", zorder=10)
        
        # Plot DRO orbit (use first DRO as reference)
        dro_1 = constellation.satellites[0]
        dro_orbit_earth = dro_1["orbit"]["dro_earth"]

        dro_orbit_earth = dro_orbit_earth.copy()
        dro_orbit_earth[:, 0] = -dro_orbit_earth[:, 0]

        ax.plot(dro_orbit_earth[:, 0], dro_orbit_earth[:, 1], "r-", lw=2, alpha=0.7, label="DRO")
        
        # Colors for different spacecraft
        sc_colors = {
            "Solar Orbiter": "#45B7D1",
            "STEREO-A": "#FF6B6B",
            "STEREO-B": "#9B59B6"
        }
        
        radii = []
        plotted_sc = set()
        
        half = context_days / 2.0
        
        # Plot each ICME event's orbit
        for sc_name, events in all_icme_events.items():
            if not events or sc_name not in spacecraft_data_dict:
                continue
            
            color = sc_colors.get(sc_name, "#888888")
            spacecraft_data = spacecraft_data_dict[sc_name]
            
            for event in events:
                t_ev = event["time"]
                t0 = t_ev - timedelta(days=half)
                t1 = t_ev + timedelta(days=half)
                
                times = spacecraft_data["times"]
                pos_helio = spacecraft_data["positions"]
                mask = [(t0 <= t <= t1) for t in times]
                
                if not any(mask):
                    continue
                
                # Transform to Earth frame (same logic as individual plots)
                sc_e = []
                for t, pxyz in zip(np.array(times)[mask], pos_helio[mask]):
                    epos = SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
                    rel = pxyz - epos
                    
                    # Rotation to align Sun-Earth line
                    sun_rel = -epos
                    theta = np.arctan2(sun_rel[1], sun_rel[0])
                    c, s = np.cos(-theta), np.sin(-theta)
                    x_e = rel[0] * c - rel[1] * s
                    y_e = rel[0] * s + rel[1] * c
                    sc_e.append([x_e, y_e])
                
                sc_e = np.array(sc_e) / AU_KM
                
                label = sc_name if sc_name not in plotted_sc else None
                plotted_sc.add(sc_name)
                ax.plot(sc_e[:, 0], sc_e[:, 1], "-", lw=1.5, color=color, alpha=0.6, label=label)
                
                # Plot event position marker
                sc_h = event["spacecraft_pos"]
                e_h = event["earth_pos"]
                rel_ev = sc_h - e_h
                sun_rel_ev = -e_h
                theta_ev = np.arctan2(sun_rel_ev[1], sun_rel_ev[0])
                c_ev, s_ev = np.cos(-theta_ev), np.sin(-theta_ev)
                sc_x_e = rel_ev[0] * c_ev - rel_ev[1] * s_ev
                sc_y_e = rel_ev[0] * s_ev + rel_ev[1] * c_ev
                sc_e_pt = np.array([sc_x_e, sc_y_e]) / AU_KM
                
                ax.plot(sc_e_pt[0], sc_e_pt[1], "o", ms=6, color=color, mec="white", mew=1, alpha=0.8)
                
                radii.extend(np.linalg.norm(sc_e, axis=1).tolist())
        

        ax.plot(+0.01, 0.0, "D", ms=8, color="lime", mec="darkgreen", mew=1.5, label="L1", zorder=5)
        
        # Plot Sun (use first event as reference for position)
        first_event = None
        for events in all_icme_events.values():
            if events:
                first_event = events[0]
                break
        
        if first_event:
            e_h = first_event["earth_pos"]
            sun_rel = -e_h
            theta_ev = np.arctan2(sun_rel[1], sun_rel[0])
            c_ev, s_ev = np.cos(-theta_ev), np.sin(-theta_ev)
            sun_x_e = sun_rel[0] * c_ev - sun_rel[1] * s_ev
            sun_y_e = sun_rel[0] * s_ev + sun_rel[1] * c_ev
            sun_e = np.array([sun_x_e, sun_y_e]) / AU_KM
            
            ax.plot(sun_e[0], sun_e[1], "o", ms=14, color="#FDB813", mec="#F37021", mew=2, label="Sun", zorder=9)
            ax.plot([0, sun_e[0]], [0, sun_e[1]], "g--", lw=1.5, alpha=0.5, label="Sun–Earth line")
            radii.append(np.linalg.norm(sun_e))
        
        radii.extend(np.linalg.norm(dro_orbit_earth, axis=1).tolist())
        
        # Set axis limits
        rmax = max(radii) if radii else 1.2
        span = rmax + 0.10 * rmax
        
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xlabel("X [AU] (Earth Frame)", fontsize=12)
        ax.set_ylabel("Y [AU] (Earth Frame)", fontsize=12)
        
        # Count total ICME events
        total_icme = sum(len(events) for events in all_icme_events.values())
        ax.set_title(f"All ICME Events — Earth-Centered Frame\n{total_icme} ICME events (±{half:.0f} days)", fontsize=13)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_folder:
            plot_folder = os.path.join(output_folder, "plots") if SAVE_DATA_CSV else output_folder
            filepath = os.path.join(plot_folder, "00_all_icme_orbits_earth_frame.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            plt.close(fig)
        
        return fig

    @staticmethod
    def _add_icme_shading(axes, icmes, t0, t1, base_color='red', label_suffix=''):
        """
        Add ICME shading to plot axes.
        
        Args:
            axes: List of matplotlib axes to add shading to
            icmes: List of ICME dictionaries
            t0, t1: Time bounds for filtering
            base_color: Base color for shading
            label_suffix: Suffix for ICME labels
        """
        if not icmes:
            return
        
        # Color variations based on base color
        if base_color == 'red':
            icme_colors = ['red', 'orange', 'darkred', 'coral', 'indianred']
        elif base_color == 'blue':
            icme_colors = ['blue', 'dodgerblue', 'darkblue', 'steelblue', 'cornflowerblue']
        else:
            icme_colors = [base_color]
        
        for i, icme in enumerate(icmes):
            color = icme_colors[i % len(icme_colors)]
            sc_start = np.datetime64(icme['sc_start'])
            sc_end = np.datetime64(icme['sc_end'])
            
            if sc_end < np.datetime64(t0) or sc_start > np.datetime64(t1):
                continue
            
            for ax in axes:
                ax.axvspan(sc_start, sc_end, alpha=0.15, color=color, zorder=0)
                ax.axvline(sc_start, color=color, linestyle=':', linewidth=1.0, alpha=0.5, zorder=1)
                ax.axvline(sc_end, color=color, linestyle=':', linewidth=1.0, alpha=0.5, zorder=1)
            
            mid_time = sc_start + (sc_end - sc_start) / 2
            y_pos = axes[0].get_ylim()[1] * (0.95 - i * 0.08)
            label_text = f"ICME {i+1}: {icme['type']}{label_suffix}"
            axes[0].text(mid_time, y_pos, label_text, ha='center', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

    @staticmethod
    def plot_solo_insitu_with_dst(event, preloaded, spacecraft_ephemeris, epoch_date, window_hours=INSITU_WINDOW_HR, spacecraft_name="Solar Orbiter", output_folder=None, return_figure=False):
        center = event.get("time")

        if not isinstance(center, datetime):
            return None
        
        # Calculate spacecraft position in Earth-centered frame (Earth at origin, Sun at (1,0) AU)
        sc_pos_hci = event["spacecraft_pos"]  # Spacecraft position in HCI (km)
        earth_pos_hci = SunEarthLineAnalyzer.earth_position_at_time(center, epoch_date)  # Earth position in HCI (km)
        
        # Relative position: spacecraft - Earth (in km)
        rel_pos = sc_pos_hci - earth_pos_hci
        
        # Direction from Earth to Sun (negative of Earth position vector)
        sun_direction = -earth_pos_hci
        theta_sun = np.arctan2(sun_direction[1], sun_direction[0])
        
        # Rotate so Sun is at (1, 0) from Earth's perspective
        cos_rot = np.cos(-theta_sun)
        sin_rot = np.sin(-theta_sun)
        
        x_earth_frame = (rel_pos[0] * cos_rot - rel_pos[1] * sin_rot) / AU_KM  # Convert to AU
        y_earth_frame = (rel_pos[0] * sin_rot + rel_pos[1] * cos_rot) / AU_KM
        
        position_str = f"S/C pos: ({x_earth_frame:.2f}, {y_earth_frame:.2f}) AU"

        d = preloaded.get(center)
        if not d:
            return None
            
        mag = d.get("mag")
        swa = d.get("swa")
        swa_moments = d.get("swa_moments")
        dst_measured = d.get("dst_measured")
        dst_pred = d.get("dst_predicted")
        dst_shifted = d.get("dst_shifted")
        kp_measured = d.get("kp_measured")
        kp_pred = d.get("kp_predicted")
        kp_shifted = d.get("kp_shifted")
        l1_shifted = d.get("l1_shifted")
        icmes_ballistic = d.get("icmes_ballistic")

        if not mag or "time" not in mag or "B" not in mag:
            print(f"[WARN] Plot generation for {spacecraft_name} at {center} failed because MAG data were unavailable.")
            return None
        t0 = center - timedelta(hours=window_hours)
        t1 = center + timedelta(hours=window_hours)

        # =====================================================================
        # AVERAGE ALL DATA TO 5-MINUTE BINS
        # =====================================================================
    
        
        # --- Average MAG data ---
        t_mag_full = ensure_datetime_array(mag["time"])
        b_full = mag["B"]
        mask_mag = (t_mag_full >= t0) & (t_mag_full <= t1)

        if not np.any(mask_mag):
            print(f"[WARN] Could not generate {spacecraft_name} plot for {center}: no MAG data in the ±{window_hours} h window.")
            return None
        
        
        t_mag_1min = t_mag_full[mask_mag]
        b_1min = b_full[mask_mag]
        
        # Decimate if still too many points (>200k)
        max_points = 200_000
        if len(t_mag_1min) > max_points:
            step = int(np.ceil(len(t_mag_1min) / float(max_points)))
            t_mag_1min = t_mag_1min[::step]
            b_1min = b_1min[::step]
        

        # Average to 5-minute bins
        b_5min = Plotter.average_to_5min_bins(t_mag_1min, b_1min, is_vector=True)
        t_mag = b_5min["time"]
        b = b_5min["data"]

        l1_b_5min = None
        t_l1 = None
        if l1_shifted and "B_gse_shifted" in l1_shifted:
            t_l1_full = ensure_datetime_array(l1_shifted["time"])
            b_l1_full = l1_shifted["B_gse_shifted"]
            mask_l1 = (t_l1_full >= t0) & (t_l1_full <= t1)
            if np.any(mask_l1):
                t_l1_1min = t_l1_full[mask_l1]
                b_l1_1min = b_l1_full[mask_l1]
                l1_b_5min = Plotter.average_to_5min_bins(t_l1_1min, b_l1_1min, is_vector=True)
                t_l1 = l1_b_5min["time"]
        
        l1_v_5min = None
        if l1_shifted and "V_shifted" in l1_shifted:
            t_l1_full = ensure_datetime_array(l1_shifted["time"])
            v_l1_full = l1_shifted["V_shifted"]
            mask_l1 = (t_l1_full >= t0) & (t_l1_full <= t1)
            if np.any(mask_l1):
                t_l1_1min = t_l1_full[mask_l1]
                v_l1_1min = v_l1_full[mask_l1]
                l1_v_5min = Plotter.average_to_5min_bins(t_l1_1min, v_l1_1min, is_vector=True)
        
        l1_n_5min = None
        if l1_shifted and "n_shifted" in l1_shifted:
            t_l1_full = ensure_datetime_array(l1_shifted["time"])
            n_l1_full = l1_shifted["n_shifted"]
            mask_l1 = (t_l1_full >= t0) & (t_l1_full <= t1)
            if np.any(mask_l1):
                t_l1_1min = t_l1_full[mask_l1]
                n_l1_1min = n_l1_full[mask_l1]
                l1_n_5min = Plotter.average_to_5min_bins(t_l1_1min, n_l1_1min, is_vector=False)
        
        l1_t_5min = None
        if l1_shifted and "T_shifted" in l1_shifted:
            t_l1_full = ensure_datetime_array(l1_shifted["time"])
            t_l1_full_data = l1_shifted["T_shifted"]
            mask_l1 = (t_l1_full >= t0) & (t_l1_full <= t1)
            if np.any(mask_l1):
                t_l1_1min = t_l1_full[mask_l1]
                t_l1_1min_data = t_l1_full_data[mask_l1]
                l1_t_5min = Plotter.average_to_5min_bins(t_l1_1min, t_l1_1min_data, is_vector=False)
        
        swa_n_5min = None
        swa_v_5min = None
        swa_t_5min = None
        if swa_moments:
            if "n" in swa_moments:
                t_swa_full = ensure_datetime_array(swa_moments["time"])
                n_swa_full = np.array(swa_moments["n"])
                mask_swa = (t_swa_full >= t0) & (t_swa_full <= t1)
                if np.any(mask_swa):
                    t_swa_1min = t_swa_full[mask_swa]
                    n_swa_1min = n_swa_full[mask_swa]
                    swa_n_5min = Plotter.average_to_5min_bins(t_swa_1min, n_swa_1min, is_vector=False)
            
            if "V" in swa_moments:
                t_swa_full = ensure_datetime_array(swa_moments["time"])
                v_swa_full = np.array(swa_moments["V"])
                mask_swa = (t_swa_full >= t0) & (t_swa_full <= t1)
                if np.any(mask_swa):
                    t_swa_1min = t_swa_full[mask_swa]
                    v_swa_1min = v_swa_full[mask_swa]
                    swa_v_5min = Plotter.average_to_5min_bins(t_swa_1min, v_swa_1min, is_vector=True)
            
            if "T" in swa_moments:
                t_swa_full = ensure_datetime_array(swa_moments["time"])
                t_swa_full_data = np.array(swa_moments["T"])
                mask_swa = (t_swa_full >= t0) & (t_swa_full <= t1)
                if np.any(mask_swa):
                    t_swa_1min = t_swa_full[mask_swa]
                    t_swa_1min_data = t_swa_full_data[mask_swa]
                    swa_t_5min = Plotter.average_to_5min_bins(t_swa_1min, t_swa_1min_data, is_vector=False)
        
        swa_eflux_5min = None
        energy_bins = None
        if swa and "eflux" in swa and swa.get("eflux") is not None:
            t_ef_full = ensure_datetime_array(swa["time"])
            mask_ef = (t_ef_full >= t0) & (t_ef_full <= t1)
            if np.any(mask_ef):
                t_ef_1min = t_ef_full[mask_ef]
                eflux_1min = np.array(swa["eflux"])[mask_ef]
                energy_bins = swa.get("energy")
                
                if eflux_1min.ndim == 2:
                    swa_eflux_5min = Plotter.average_spectrogram_to_5min(t_ef_1min, eflux_1min)
        
        # --- Average Dst data ---
        dst_shifted_5min = None
        if dst_shifted:
            dst_times = np.array(dst_shifted["time"])
            dst_vals = np.array(dst_shifted["dst_shifted"])
            dst_shifted_5min = Plotter.average_to_5min_bins(dst_times, dst_vals, is_vector=False)
        
        dst_pred_5min = None
        if dst_pred:
            dst_pred_times = np.array(dst_pred["time"])
            dst_pred_vals = np.array(dst_pred["dst_predicted"])
            dst_pred_5min = Plotter.average_to_5min_bins(dst_pred_times, dst_pred_vals, is_vector=False)
        
        # --- Average Kp data ---
        kp_shifted_5min = None
        if kp_shifted:
            kp_times = np.array(kp_shifted["time"])
            kp_vals = np.array(kp_shifted["kp_shifted"])
            kp_shifted_5min = Plotter.average_to_5min_bins(kp_times, kp_vals, is_vector=False)
        
        kp_pred_5min = None
        if kp_pred:
            kp_pred_times = np.array(kp_pred["time"])
            kp_pred_vals = np.array(kp_pred["kp_predicted"])
            kp_pred_5min = Plotter.average_to_5min_bins(kp_pred_times, kp_pred_vals, is_vector=False)
        
        sc_times = spacecraft_ephemeris["times"]
        sc_positions = spacecraft_ephemeris["positions"]
        mask_sc = [(t0 <= t <= t1) for t in sc_times]
        
        distances_5min = None
        angles_5min = None
        t_sc_5min = None
        
        if np.any(mask_sc):
            t_sc = np.array(sc_times)[mask_sc]
            pos_sc = sc_positions[mask_sc]
            
            earth_positions = np.array([SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date) for t in t_sc])
            distances = np.linalg.norm(pos_sc - earth_positions, axis=1) / AU_KM
            
            angles = []
            for i in range(len(t_sc)):
                earth_vec = earth_positions[i]
                sc_vec = pos_sc[i]
                earth_hat = earth_vec / np.linalg.norm(earth_vec)
                sc_hat = sc_vec / np.linalg.norm(sc_vec)
                cos_angle = np.clip(np.dot(earth_hat, sc_hat), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                angles.append(angle_deg)
            angles = np.array(angles)
            
            distances_5min = Plotter.average_to_5min_bins(t_sc, distances, is_vector=False)
            angles_5min = Plotter.average_to_5min_bins(t_sc, angles, is_vector=False)
            t_sc_5min = distances_5min["time"]




        # FIGURE 1: IN-SITU DATA (|B|, Br, Bt, Bn, n, V, T)
        fig1 = plt.figure(figsize=(14, 14))
        gs1 = GridSpec(7, 1, figure=fig1, hspace=0.08)
        ax_insitu = [fig1.add_subplot(gs1[i, 0]) for i in range(7)]

        if icmes_ballistic:
            Plotter._add_icme_shading(ax_insitu, icmes_ballistic, t0, t1)

        # Panel 0: |B| (magnitude)
        b_mag = np.linalg.norm(b, axis=1)
        ax_insitu[0].plot(t_mag, b_mag, linewidth=1.2, color='black', label=f"{spacecraft_name} |B|")
        
        if l1_b_5min is not None:
            b_l1_5min = l1_b_5min["data"]
            t_l1_5min = l1_b_5min["time"]
            b_l1_mag = np.linalg.norm(b_l1_5min, axis=1)
            ax_insitu[0].plot(t_l1_5min, b_l1_mag, linewidth=1.0, color='gray', linestyle=':', alpha=0.7, label="L1 |B| (shifted)")
        
        ax_insitu[0].set_ylabel("|B| [nT]", fontsize=11, fontweight='bold')
        ax_insitu[0].grid(True, alpha=0.3)
        ax_insitu[0].legend(loc="upper right", fontsize=8)
        ax_insitu[0].set_ylim(bottom=0)

        # Panels 1-3: Br, Bt, Bn
        b_labels = ["r", "t", "n"]
        b_colors = ["red", "green", "blue"]

        for i in range(min(3, b.shape[1])):
            ax_insitu[i+1].plot(t_mag, b[:, i], linewidth=1.2, color=b_colors[i], label=f"{spacecraft_name} B{b_labels[i]}")
            
            if l1_b_5min is not None and l1_b_5min["data"].shape[1] > i:
                b_l1_5min = l1_b_5min["data"]
                t_l1_5min = l1_b_5min["time"]
                b_l1_i = b_l1_5min[:, i]
                ax_insitu[i+1].plot(t_l1_5min, b_l1_i, linewidth=1.0, color='black', linestyle=':', alpha=0.7, label=f"L1 B{b_labels[i]} (shifted)")
            
            ax_insitu[i+1].set_ylabel(f"B{b_labels[i]} [nT]", fontsize=11, fontweight='bold')
            ax_insitu[i+1].grid(True, alpha=0.3)
            ax_insitu[i+1].legend(loc="upper right", fontsize=8)
            ax_insitu[i+1].axhline(0, color="black", linestyle=":", alpha=0.5, linewidth=0.8)

        # Panel 4: Density (n)
        if swa_n_5min is not None:
            t_swa_5min = swa_n_5min["time"]
            n_data_5min = swa_n_5min["data"]
            ax_insitu[4].plot(t_swa_5min, n_data_5min, linewidth=1.2, color="orange", label=f"{spacecraft_name} n")
            
            if l1_n_5min is not None:
                t_l1_n_5min = l1_n_5min["time"]
                n_l1_data_5min = l1_n_5min["data"]
                ax_insitu[4].plot(t_l1_n_5min, n_l1_data_5min, linewidth=1.0, color='black', linestyle=':', alpha=0.7, label="L1 n (shifted)")
            
            ax_insitu[4].set_ylabel("n [cm⁻³]", fontsize=11, fontweight='bold')
            ax_insitu[4].legend(loc="upper right", fontsize=8)
            ax_insitu[4].set_ylim(bottom=0)
        else:
            ax_insitu[4].text(0.5, 0.5, "No n", ha="center", va="center", transform=ax_insitu[4].transAxes, fontsize=10)
        ax_insitu[4].grid(True, alpha=0.3)

        # Panel 5: Velocity (V)
        if swa_v_5min is not None:
            t_swa_v_5min = swa_v_5min["time"]
            v_data_5min = swa_v_5min["data"]
            v_mag_5min = np.linalg.norm(v_data_5min, axis=1)
            ax_insitu[5].plot(t_swa_v_5min, v_mag_5min, linewidth=1.2, color="cyan", label=f"{spacecraft_name} V")
            
            if l1_v_5min is not None:
                t_l1_v_5min = l1_v_5min["time"]
                v_l1_5min = l1_v_5min["data"]
                v_l1_mag_5min = np.linalg.norm(v_l1_5min, axis=1)
                ax_insitu[5].plot(t_l1_v_5min, v_l1_mag_5min, linewidth=1.0, color='black', linestyle=':', alpha=0.7, label="L1 V (shifted)")
            
            ax_insitu[5].set_ylabel("V [km/s]", fontsize=11, fontweight='bold')
            ax_insitu[5].legend(loc="upper right", fontsize=8)
            ax_insitu[5].set_ylim(bottom=0)
        else:
            ax_insitu[5].text(0.5, 0.5, "No V", ha="center", va="center", transform=ax_insitu[5].transAxes, fontsize=10)
        ax_insitu[5].grid(True, alpha=0.3)

        # Panel 6: Temperature (T)
        if swa_t_5min is not None:
            t_swa_t_5min = swa_t_5min["time"]
            t_data_5min = swa_t_5min["data"]
            ax_insitu[6].plot(t_swa_t_5min, t_data_5min, linewidth=1.2, color="purple", label=f"{spacecraft_name} T")
            
            if l1_t_5min is not None:
                t_l1_t_5min = l1_t_5min["time"]
                t_l1_data_5min = l1_t_5min["data"]
                ax_insitu[6].plot(t_l1_t_5min, t_l1_data_5min, linewidth=1.0, color='black', linestyle=':', alpha=0.7, label="L1 T (shifted)")
            
            ax_insitu[6].set_ylabel("T [eV]", fontsize=11, fontweight='bold')
            ax_insitu[6].legend(loc="upper right", fontsize=8)
            ax_insitu[6].set_ylim(bottom=0)
        else:
            ax_insitu[6].text(0.5, 0.5, "No T", ha="center", va="center", transform=ax_insitu[6].transAxes, fontsize=10)
        ax_insitu[6].grid(True, alpha=0.3)

        # Format x-axis (only on bottom panel)
        ax_insitu[-1].set_xlabel("Time (UTC)", fontsize=11)
        locator = AutoDateLocator()
        formatter = DateFormatter("%Y-%b-%d\n%H:%M")
        ax_insitu[-1].xaxis.set_major_locator(locator)
        ax_insitu[-1].xaxis.set_major_formatter(formatter)
        ax_insitu[-1].set_xlim(t_mag[0], t_mag[-1])
        
        # Share x-axis for all panels
        for i in range(6):
            ax_insitu[i].sharex(ax_insitu[6])
            ax_insitu[i].tick_params(labelbottom=False)

        plt.suptitle(f"{spacecraft_name} In-Situ Measurements – {center.strftime('%Y-%m-%d %H:%M:%S')}\n{position_str}", fontsize=14, y=0.998)
        fig1.autofmt_xdate()
        plt.tight_layout()

        fig2 = plt.figure(figsize=(14, 12))
        gs2 = GridSpec(4, 1, figure=fig2, hspace=0.12)
        ax_context = [fig2.add_subplot(gs2[i, 0]) for i in range(4)]

        if icmes_ballistic:
            Plotter._add_icme_shading(ax_context, icmes_ballistic, t0, t1)

        # Panel 0: Eflux
        if swa_eflux_5min is not None and energy_bins is not None:
            from matplotlib.dates import date2num
            import matplotlib.colors as mcolors

            t_ef_5min = swa_eflux_5min["time"]
            eflux_5min = swa_eflux_5min["eflux"]

            if eflux_5min.ndim == 2:
                times_mpl = date2num(t_ef_5min)
                t_mesh, e_mesh = np.meshgrid(times_mpl, energy_bins)

                pcm = ax_context[0].pcolormesh(t_mesh, e_mesh, eflux_5min.T, shading="auto", cmap="jet", 
                                               norm=mcolors.LogNorm(vmin=1e5, vmax=1e10))
                ax_context[0].set_yscale("log")
                ax_context[0].set_ylabel("Energy [eV]", fontsize=11, fontweight='bold')
                ax_context[0].set_ylim([100, 2e4])

                pos = ax_context[0].get_position()
                cax = fig2.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
                cbar = fig2.colorbar(pcm, cax=cax)
                cbar.set_label(r"eflux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$]", rotation=270, labelpad=20, fontsize=9)
            else:
                ax_context[0].text(0.5, 0.5, "No 2D eflux", ha="center", va="center", 
                                  transform=ax_context[0].transAxes, fontsize=10)
        else:
            ax_context[0].text(0.5, 0.5, "No Eflux", ha="center", va="center", 
                              transform=ax_context[0].transAxes, fontsize=10)
        ax_context[0].grid(True, alpha=0.3)

        # Panel 1: Dst
        has_dst = False
        if dst_shifted_5min is not None:
            ax_context[1].plot(dst_shifted_5min["time"], dst_shifted_5min["data"], linewidth=1.5, 
                             color="blue", label="Measured Dst (Time-Shifted)", alpha=0.8)
            has_dst = True
        if dst_pred_5min is not None:
            ax_context[1].plot(dst_pred_5min["time"], dst_pred_5min["data"], linewidth=1.5, 
                             color="red", linestyle="--", label="Predicted Dst (Crossover)", alpha=0.8)
            has_dst = True

        if has_dst:
            ax_context[1].set_ylabel("Dst [nT]", fontsize=11, fontweight='bold')
            ax_context[1].axhline(0, color="black", linestyle=":", alpha=0.5, linewidth=0.8)
            ax_context[1].axhline(-50, color="orange", linestyle="--", alpha=0.3, linewidth=0.8, label="Moderate storm")
            ax_context[1].axhline(-100, color="red", linestyle="--", alpha=0.3, linewidth=0.8, label="Strong storm")
            ax_context[1].legend(loc="upper right", fontsize=8)
        else:
            ax_context[1].text(0.5, 0.5, "No Dst", ha="center", va="center", 
                              transform=ax_context[1].transAxes, fontsize=10)
        ax_context[1].grid(True, alpha=0.3)

        # Panel 2: Kp
        has_kp = False
        if kp_shifted_5min is not None:
            ax_context[2].plot(kp_shifted_5min["time"], kp_shifted_5min["data"], linewidth=1.5, 
                             color="blue", label="Measured Kp (Time-Shifted)", alpha=0.8)
            has_kp = True
        if kp_pred_5min is not None:
            ax_context[2].plot(kp_pred_5min["time"], kp_pred_5min["data"], linewidth=1.5, 
                             color="red", linestyle="--", label="Predicted Kp (Crossover)", alpha=0.8)
            has_kp = True

        if has_kp:
            ax_context[2].set_ylabel("Kp", fontsize=11, fontweight='bold')
            ax_context[2].set_ylim([0, 9])
            ax_context[2].axhline(5, color="orange", linestyle="--", alpha=0.3, linewidth=0.8, label="Storm threshold")
            ax_context[2].axhline(7, color="red", linestyle="--", alpha=0.3, linewidth=0.8, label="Strong storm")
            ax_context[2].legend(loc="upper right", fontsize=8)
        else:
            ax_context[2].text(0.5, 0.5, "No Kp", ha="center", va="center", 
                              transform=ax_context[2].transAxes, fontsize=10)
        ax_context[2].grid(True, alpha=0.3)

        # Panel 3: Position (Distance + Angle)
        if distances_5min is not None and angles_5min is not None:
            t_sc_5min = distances_5min["time"]
            dist_5min = distances_5min["data"]
            ang_5min = angles_5min["data"]
            
            ax_context[3].plot(t_sc_5min, dist_5min, 'k-', linewidth=1.5, label='Distance to Earth')
            ax_context[3].set_ylabel('Distance [AU]', color='k', fontsize=11, fontweight='bold')
            ax_context[3].tick_params(axis='y', labelcolor='k')
            ax_context[3].grid(True, alpha=0.3)
            ax_context[3].set_ylim(bottom=0)
            
            ax_angle = ax_context[3].twinx()
            ax_angle.plot(t_sc_5min, ang_5min, 'r-', linewidth=1.5, label='Angle from Sun-Earth line')
            ax_angle.set_ylabel('Angle [deg]', color='r', fontsize=11, fontweight='bold')
            ax_angle.tick_params(axis='y', labelcolor='r')
            ax_angle.set_ylim([0, 180])
            
            lines1, labels1 = ax_context[3].get_legend_handles_labels()
            lines2, labels2 = ax_angle.get_legend_handles_labels()
            ax_context[3].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        else:
            ax_context[3].text(0.5, 0.5, "No ephemeris in window", ha="center", va="center", 
                              transform=ax_context[3].transAxes, fontsize=10)
            ax_context[3].grid(True, alpha=0.3)

        # Format x-axis (only on bottom panel)
        ax_context[-1].set_xlabel("Time (UTC)", fontsize=11)
        ax_context[-1].xaxis.set_major_locator(locator)
        ax_context[-1].xaxis.set_major_formatter(formatter)
        ax_context[-1].set_xlim(t_mag[0], t_mag[-1])
        
        # Share x-axis for all panels
        for i in range(3):
            ax_context[i].sharex(ax_context[3])
            ax_context[i].tick_params(labelbottom=False)

        plt.suptitle(f"{spacecraft_name} Context Data — {center.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14, y=0.995)
        fig2.autofmt_xdate()
        plt.tight_layout()

        if output_folder:
            safe_sc_name = spacecraft_name.replace(" ", "_").replace("-", "")
            timestamp = center.strftime('%Y%m%d')
            
            filepath1 = os.path.join(output_folder, f"insitu_{safe_sc_name}_{timestamp}.png")
            fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath1}")
            
            filepath2 = os.path.join(output_folder, f"context_{safe_sc_name}_{timestamp}.png")
            fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath2}")


        if return_figure:
            return fig1, fig2
        else:
            return fig1, fig2


# ------------------------------ Main driver ----------------------------- #

def main():
    print("Hénon DRO Constellation Crossover Analysis with L1 Comparison")
    print(f"Period: {START_DATE:%Y-%m-%d} to {END_DATE:%Y-%m-%d}")

    # Setup output folder if saving plots
    output_folder = setup_output_folder()

    # Load ICME catalogue
    icme_catalogue = CME_Catalogue.load_catalouge()
    if not icme_catalogue:
        print("[INFO] Attempting to load the ICME catalogue from a local file.")
        icme_catalogue = CME_Catalogue.load_from_local_file()
    
    if not icme_catalogue:
        print("[WARN] ICME catalogue could not be loaded; continuing without ICME matching.")
    else:
        icme_catalogue = [icme for icme in icme_catalogue 
                         if START_DATE <= icme['start'] <= END_DATE]
        print(f"[INFO] Found {len(icme_catalogue)} ICME events in the analysis date range.")


    constellation = DROConstellation()

    spacecraft_list = ["STEREO-A", "STEREO-B", "Solar Orbiter"]
    spacecraft_data = {}
    for sc in spacecraft_list:
        data = SpacecraftData.get_positions(sc, START_DATE, END_DATE, dt_hours=DT_HOURS)
        if data:
            spacecraft_data[sc] = data
        else:
            print(f"[WARN] No ephemeris data were available for {sc}.")

    if not spacecraft_data:
        print("[ERROR] No spacecraft ephemeris data available; stopping run.")
        return
    
    earth_start = START_DATE - timedelta(days=EARTH_EPHEMERIS_MARGIN_DAYS)
    earth_end   = END_DATE   + timedelta(days=EARTH_EPHEMERIS_MARGIN_DAYS)
    earth_data  = SpacecraftData.get_positions("Earth", earth_start, earth_end, dt_hours=DT_HOURS)

    if earth_data:
        SunEarthLineAnalyzer.set_earth_ephemeris(earth_data["times"], earth_data["positions"])
        spacecraft_data["Earth"] = earth_data
    else:
        print("[WARN] Failed to load Earth ephemeris from Horizons")

    finder = CrossoverFinder(constellation, SUN_EARTH_TOL_AU, DRO_TOL_AU)
    all_events = {}

    # Only treat actual spacecraft as targets, NOT Earth
    for sc in spacecraft_list:
        if sc not in spacecraft_data:
            continue
        sc_data = spacecraft_data[sc]


        ev   = finder.find_events(sc_data, START_DATE)
        uniq = finder.group_events_by_day(ev)
        limit = EVENT_LIMIT.get(sc)
        if isinstance(limit, int) and limit >= 0:
            uniq = uniq[:limit]
        all_events[sc] = uniq
        print(f"[INFO] Identified {len(uniq)} crossover events for {sc}.")

    total_events = sum(len(v) for v in all_events.values())
    if total_events == 0:
        print("\n[INFO] No crossover events found with the current tolerances.")
        return
    
    # LOAD IN-SITU DATA FOR ALL EVENTS (with full ICME matching)
    
    preloaded_data = {}
    
    for sc in all_events.keys():
        if not all_events[sc]:
            continue
            
        print(f"\n[INFO] Loading in-situ data for {sc}.")
        
        if sc == "Solar Orbiter":
            preloaded_data[sc] = SoloDataLoader.preload_all(
                all_events[sc],
                spacecraft_data[sc],
                START_DATE,
                icme_catalogue,
                window_hours=INSITU_WINDOW_HR
            )
        elif sc in ["STEREO-A", "STEREO-B"]:
            preloaded_data[sc] = STEREODataLoader.preload_all(
                sc,
                all_events[sc],
                spacecraft_data[sc],
                START_DATE,
                icme_catalogue,
                window_hours=INSITU_WINDOW_HR
            )
        else:
            print(f"[INFO] Skipping {sc}: no in-situ data loader is implemented for this spacecraft.")
            preloaded_data[sc] = None

    # FILTER EVENTS: Apply ICME filter if toggle is enabled
    
    if PLOT_ONLY_ICME_EVENTS:
        print("\n[INFO] Filtering to crossover events with ICME matches only.")
        events_to_plot = {}
        
        for sc, evs in all_events.items():
            if not evs or preloaded_data.get(sc) is None:
                events_to_plot[sc] = []
                continue
            
            icme_filtered = []
            for e in evs:
                t_ev = e["time"]
                data = preloaded_data[sc].get(t_ev)
                
                if data and data.get("icmes_ballistic") is not None:
                    icme_filtered.append(e)
            
            events_to_plot[sc] = icme_filtered
            print(f"[INFO] {sc}: {len(icme_filtered)} crossover events with ICME matches (out of {len(evs)} total).")
        
        total_events = sum(len(v) for v in events_to_plot.values())
        print(f"\n{'='*80}")
        print(f"TOTAL EVENTS FOR PLOTTING: {total_events}")
        print(f"{'='*80}\n")
    
    if WRITE_EVENTS_CSV:
        print("\n[INFO] Preparing crossover events CSV output.")
        
        # Use events_to_plot if ICME filter is enabled, otherwise use all_events
        events_to_write = events_to_plot if PLOT_ONLY_ICME_EVENTS else all_events
        
        records = []
        for sc, evs in events_to_write.items():
            for e in evs:
                records.append({
                    "Spacecraft":           sc,
                    "Date":                 e["time"].strftime("%Y-%m-%d"),
                    "Time_UTC":             e["time"].strftime("%H:%M:%S"),
                    "DRO_ID":               e["dro_id"],
                    "Perp_to_SE_Line_AU":   round(e["perp_to_se_line_au"], 5),
                    "Along_SE_Line_AU":     round(e["along_se_line_au"], 5),
                    "Distance_to_DRO_AU":   round(e["dist_to_dro_au"], 5),
                    "Between_Sun_Earth":    e["is_between_sun_earth"],
                    "SC_X_AU":              round(e["spacecraft_pos"][0] / AU_KM, 5),
                    "SC_Y_AU":              round(e["spacecraft_pos"][1] / AU_KM, 5),
                    "SC_Z_AU":              round(e["spacecraft_pos"][2] / AU_KM, 5),
                })
        
        if records:
            df = pd.DataFrame(records)
            df.sort_values(["Date", "Spacecraft"], inplace=True)
            
            # Add suffix to filename if ICME filter is enabled
            csv_filename = CSV_FILENAME
            if PLOT_ONLY_ICME_EVENTS:
                csv_filename = CSV_FILENAME.replace('.csv', '_ICME_only.csv')
            
            if output_folder:
                csv_path = os.path.join(output_folder, csv_filename)
                df.to_csv(csv_path, index=False)
                print(f"[INFO] Wrote crossover events CSV: {csv_path} ({len(df)} rows).")
            else:
                csv_path = PROJECT_ROOT / csv_filename
                df.to_csv(csv_path, index=False)
                print(f"[INFO] Wrote crossover events CSV: {csv_path} ({len(df)} rows).")
        else:
            print("[INFO] No crossover events were available to write to CSV.")


    if PLOT_ONLY_ICME_EVENTS and any(len(v) > 0 for v in events_to_plot.values()):
        Plotter.plot_all_icme_orbits_earth_frame(
            events_to_plot, 
            constellation, 
            spacecraft_data, 
            START_DATE,
            context_days=TRACK_CONTEXT_DAYS,
            output_folder=output_folder
        )
    
    # If ICME filter is disabled, set events_to_plot to all_events
    if not PLOT_ONLY_ICME_EVENTS:
        print("\n[INFO] ICME-only filtering is disabled; all crossover events will be plotted.")
        
        events_to_plot = all_events
        
        for sc, evs in all_events.items():
            print(f"[INFO] {sc}: {len(evs)} events selected for plotting.")
        
        total_events = sum(len(v) for v in events_to_plot.values())
        print(f"[INFO] Total events selected for plotting: {total_events}.")

    # PLOTTING
    if PLOT_OVERVIEW:
        suffix = "ICME events only" if PLOT_ONLY_ICME_EVENTS else "all events"
        Plotter.plot_constellation_overview(events_to_plot, constellation, output_folder)

    if PLOT_HELIOCENTRIC and any(len(v) > 0 for v in events_to_plot.values()):
        suffix = "ICME events only" if PLOT_ONLY_ICME_EVENTS else "all events"
        print(f"\n[INFO] Plotting heliocentric tracks (±21 d; {suffix}).")
        for sc, evs in events_to_plot.items():
            for e in evs:
                print(f"Plotting {sc} event on {e['time'].strftime('%Y-%m-%d')}")
                Plotter.plot_event_detail(e, sc, constellation,
                                          spacecraft_data[sc], START_DATE,
                                          context_days=TRACK_CONTEXT_DAYS,
                                          output_folder=output_folder)

    if PLOT_EARTH_FRAME and any(len(v) > 0 for v in events_to_plot.values()):
        suffix = "ICME events only" if PLOT_ONLY_ICME_EVENTS else "all events"
        print(f"\n[INFO] Plotting Earth-centered tracks (±21 d; {suffix}).")
        for sc, evs in events_to_plot.items():
            for e in evs:
                print(f"Plotting {sc} event on {e['time'].strftime('%Y-%m-%d')}")
                Plotter.plot_event_detail_earth_frame(
                    e, sc, constellation, spacecraft_data[sc],
                    START_DATE, context_days=TRACK_CONTEXT_DAYS,
                    output_folder=output_folder
                )

    if PLOT_INSITU_DST:
        suffix = "ICME events only" if PLOT_ONLY_ICME_EVENTS else "all events"
        print(f"\n[INFO] Generating in-situ stack plots ({suffix}).")
        
        # Collect figures for PDF compilation if enabled
        pdf_figures = [] if SAVE_TO_PDF else None

        def handle_insitu_plot(spacecraft_label, event, preloaded_map, sc_ephem):
            """
            Common handler for Solar Orbiter / STEREO-A / STEREO-B in-situ plotting.
            Returns plot_result from Plotter.plot_solo_insitu_with_dst().
            """
            print(f"Plotting {spacecraft_label} event on {event['time'].strftime('%Y-%m-%d')}")

            # Save CSV if enabled
            if SAVE_DATA_CSV and output_folder:
                write_event_csv(
                    event,
                    preloaded_map,
                    sc_ephem,
                    START_DATE,
                    spacecraft_label,
                    output_folder
                )

            plot_output_folder = output_folder if not SAVE_DATA_CSV else os.path.join(output_folder, "plots")


            plot_result = Plotter.plot_solo_insitu_with_dst(
                event,
                preloaded_map,
                sc_ephem,
                START_DATE,
                spacecraft_name=spacecraft_label,
                output_folder=plot_output_folder,
                return_figure=SAVE_TO_PDF
            )

            # True failure
            if plot_result is None:
                print(f"[WARN] Plot generation failed for event at {event['time']}.")
                return None

            # Success
            print(f"[INFO] Plot created successfully for {spacecraft_label} on {event['time'].strftime('%Y-%m-%d')}.")
            # If compiling PDF, keep fig1 and close fig2
            if SAVE_TO_PDF:
                try:
                    fig1, fig2 = plot_result
                    if fig1 is not None:
                        pdf_figures.append((
                            spacecraft_label,
                            event["perp_to_se_line_au"],
                            event["time"],
                            fig1
                        ))
                    if fig2 is not None:
                        plt.close(fig2)
                except Exception as e:
                    print(f"[WARN] Plot was created, but PDF handling failed for event at {event['time']}: {e}")

            return plot_result
        
        # Plot Solar Orbiter events
        if events_to_plot.get("Solar Orbiter") and preloaded_data.get("Solar Orbiter"):
            for e in events_to_plot["Solar Orbiter"]:
                handle_insitu_plot(
                    "Solar Orbiter",
                    e,
                    preloaded_data["Solar Orbiter"],
                    spacecraft_data["Solar Orbiter"]
                )
        
        # Plot STEREO-A events
        if events_to_plot.get("STEREO-A") and preloaded_data.get("STEREO-A"):
            for e in events_to_plot["STEREO-A"]:
                handle_insitu_plot(
                    "STEREO-A",
                    e,
                    preloaded_data["STEREO-A"],
                    spacecraft_data["STEREO-A"]
                )

        # Plot STEREO-B events
        if events_to_plot.get("STEREO-B") and preloaded_data.get("STEREO-B"):
            for e in events_to_plot["STEREO-B"]:
                handle_insitu_plot(
                    "STEREO-B",
                    e,
                    preloaded_data["STEREO-B"],
                    spacecraft_data["STEREO-B"]
                )

        if SAVE_TO_PDF and pdf_figures and output_folder:
            print(f"[INFO] Compiling PDF containing {len(pdf_figures)} plots.")
            
            plot_folder = os.path.join(output_folder, "plots") if SAVE_DATA_CSV else output_folder
            
            # Create filename with date range and ICME filter info
            suffix = "_ICME_only" if PLOT_ONLY_ICME_EVENTS else "_all_events"
            pdf_filename = f"compiled_insitu_plots{suffix}.pdf"
            pdf_path = os.path.join(plot_folder, pdf_filename)
            
            # Sort figures by distance from Sun–Earth line (closest first), then by time
            pdf_figures.sort(key=lambda x: (x[1], x[2]))

            # Create multi-page PDF
            with PdfPages(pdf_path) as pdf:
                for spacecraft, perp_au, event_time, fig in pdf_figures:
                    print(
                            f"[INFO] Adding plot to PDF: {spacecraft} | perp={perp_au:.4f} AU | "
                            f"{event_time.strftime('%Y-%m-%d %H:%M')}"
                        )
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
                # Add metadata
                d = pdf.infodict()
                d['Title'] = f'Compiled In-Situ Plots{suffix}'
                d['Author'] = 'Hénon DRO Analysis Pipeline'
                d['Subject'] = f'{len(pdf_figures)} crossover events'
                d['Keywords'] = 'Solar Wind, Hénon DRO, Crossover Events'
                d['CreationDate'] = datetime.now()
            
            print(f"\nPDF compiled successfully: {pdf_path}")
            print(f"  Total pages: {len(pdf_figures)}")
            print(f"{'='*80}\n")
            
    print("\n[INFO] Analysis complete.")
    if not SAVE_PLOTS_TO_FILE:
        plt.show()
    else:
        print(f"\n[INFO] All plots were saved to: {output_folder}")


if __name__ == "__main__":
    main()
