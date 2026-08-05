"""
injection_analysis.py
=====================

Analysis module for comparing CME-driven injection behaviour between
L1 measurements and spacecraft measurements using Burton-model Dst
predictions and related driver quantities.

This script loads event-level CSV files produced by primary.py, computes
minimum predicted Dst, velocity and B_z snapshots at minimum Dst,
integrated injection proxies, and comparison metrics between L1 and
spacecraft measurements. It also generates parity plots, a single
illustrative event timeseries, decomposition plots, and summary
statistics.

Configuration
-------------
Set the file paths in the configuration block at the top of this module before
running:
- EVENTS_DIR should point to the folder containing event_*.csv files generated
  by primary.py.
- OUTPUT_DIR should point to the folder where output figures will be saved.

Notes
-----
This module uses Burton-model Dst predictions and configurable analysis
settings when comparing injection-related quantities between L1 and
spacecraft event measurements.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------ Configurations ------------------------------ #

@dataclass(frozen=True)
class Config:
    # Folder containing event_*.csv files produced by primary.py
    EVENTS_DIR: str = "/Users/henryhodges/Documents/Year 4/Masters/Code/figures/big file/csv"
    EVENT_GLOB: str = "event_*.csv"

    # Optional event-level filters
    ONLY_CLEAR_CME: bool = False
    EXCLUDE_REPEATED: bool = False

    # Initial Dst value for Burton-model integration
    DST0: float = 0.0

    # Minimum data requirements for valid event metrics
    SNAPSHOT_SEARCH_RADIUS: int = 10
    MIN_FINITE: int = 3

    # Bootstrap settings for regression confidence intervals
    BOOTSTRAP_N: int = 10_000
    BOOTSTRAP_CI: float = 95.0
    RNG_SEED: int = 42

    # Plot a single reproducible example timeseries event
    PLOT_RANDOM_TIMESERIES: bool = True

    # Plot display and save controls
    SHOW_PLOTS: bool = False
    SAVE_PLOTS: bool = True

    # Folder where analysis figures will be saved
    OUTPUT_DIR: str = "/Users/henryhodges/Documents/Year 4/Masters/Code/INJECTION TEST (TEMP)"
    DPI:        int  = 100

REQUIRED_COLUMNS_COMMON = [
    "cme_start_utc",
    "cme_end_utc",
    "unix_timestamp",
    "sc_distance_to_earth_au",
    "sc_angle_from_sun_earth_line_deg",
]

REQUIRED_COLUMNS_SC = [
    "B_n",
]

REQUIRED_COLUMNS_L1_XCORR = [
    "l1_B_z_gse_xcorr",
    "l1_V_mag_xcorr",
]

REQUIRED_COLUMNS_L1_BALLISTIC = [
    "l1_B_z_gse_ballistic",
    "l1_V_mag_ballistic",
]

OPTIONAL_FILTER_COLUMNS = [
    "clear_cme",
    "repeated_measurements",
]


# --------------------------------- Utilities -------------------------------- #

def _num(s):
    """Convert a pandas Series-like object to a float NumPy array, coercing invalid values to NaN."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

def _parse_utc(s):
    """Parse a UTC timestamp-like value into a timezone-aware pandas Timestamp."""
    ts = pd.to_datetime(str(s).strip(), utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse UTC: {s!r}")
    return ts

def _finite_mask(*arrs):
    """Return a boolean mask identifying elements that are finite across all input arrays."""
    m = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def _linear_fit_r2(x, y):
    """Fit a straight line to finite paired samples and return slope, intercept, and $$R^2$$."""
    m = _finite_mask(x, y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    xx, yy = x[m], y[m]
    a, b   = np.polyfit(xx, yy, 1)
    yhat   = a*xx + b
    ss_res = float(np.sum((yy-yhat)**2))
    ss_tot = float(np.sum((yy-yy.mean())**2))
    r2     = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), float(r2)

def _bootstrap_regression(x, y, n_boot, ci, rng):
    """Estimate a linear regression and bootstrap confidence interval band for finite paired samples."""
    m = _finite_mask(x, y)
    xx, yy = x[m], y[m]
    n = int(xx.size)
    if n < 3:
        return None
    slope, intercept, r2 = _linear_fit_r2(xx, yy)
    x_fit = np.linspace(xx.min(), xx.max(), 300)
    boot_slopes = np.empty(n_boot)
    boot_ints   = np.empty(n_boot)
    boot_lines  = np.empty((n_boot, 300))
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(xx[idx]) == 0:
            boot_slopes[i] = boot_ints[i] = np.nan
            boot_lines[i]  = np.nan
            continue
        a, b = np.polyfit(xx[idx], yy[idx], 1)
        boot_slopes[i] = a
        boot_ints[i]   = b
        boot_lines[i]  = a*x_fit + b
    alpha = (100.0 - ci) / 2.0
    return {
        "x": xx, "y": yy, "n": n,
        "slope": slope, "intercept": intercept, "r2": r2,
        "x_fit": x_fit,
        "y_fit": slope*x_fit + intercept,
        "ci_lower":    np.nanpercentile(boot_lines, alpha,       axis=0),
        "ci_upper":    np.nanpercentile(boot_lines, 100.0-alpha, axis=0),
        "boot_slopes": boot_slopes,
        "boot_ints":   boot_ints,
    }

def _nearest_finite(arr, idx, radius):
    """Return the nearest finite value to a target index within a specified search radius."""
    if np.isfinite(arr[idx]):
        return float(arr[idx]), 0
    n = len(arr)
    for delta in range(1, radius+1):
        for i in [idx-delta, idx+delta]:
            if 0 <= i < n and np.isfinite(arr[i]):
                return float(arr[i]), delta
    return np.nan, None

def _validate_sc_velocity_columns(df: pd.DataFrame, filepath):
    """Check that the CSV contains either scalar spacecraft speed or all three spacecraft velocity components."""
    has_vmag = "swa_V_mag" in df.columns
    has_components = all(col in df.columns for col in ["swa_V_r", "swa_V_t", "swa_V_n"])

    if has_vmag or has_components:
        return True

    print(
        f"[WARN] Skipping {Path(filepath).name}: missing required spacecraft velocity columns. "
        "Expected either 'swa_V_mag' or all of 'swa_V_r', 'swa_V_t', and 'swa_V_n'."
    )
    return False

def _validate_columns(df: pd.DataFrame, filepath, required_columns):
    """Check that all required CSV columns are present before processing an event."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(
            f"[WARN] Skipping {Path(filepath).name}: missing required columns: "
            + ", ".join(missing)
        )
        return False
    return True


# --------------------------------- Dst model -------------------------------- #

def predict_dst_obrien(t_unix, bz_nT, v_kms, dst0=0.0):
    """Predict Dst using the O'Brien and McPherron ring-current formulation."""
    n   = len(t_unix)
    dst = np.full(n, np.nan)
    dst[0] = dst0
    for i in range(1, n):
        dt_s = float(t_unix[i]-t_unix[i-1])
        if dt_s <= 0 or not np.isfinite(dt_s):
            dst[i] = dst[i-1]; continue
        dt_hr = dt_s/3600.0
        bz = float(bz_nT[i-1])
        v  = float(v_kms[i-1])
        if not np.isfinite(bz) or not np.isfinite(v) or v <= 0:
            dst[i] = dst[i-1]*np.exp(-dt_hr/7.7); continue
        Bs  = max(-bz, 0.0)
        E   = v*Bs/1000.0
        Q   = -4.4*(E-0.5) if E > 0.5 else 0.0
        tau = 2.4*np.exp(9.74/(4.69+E))
        dst[i] = dst[i-1]+(Q-dst[i-1]/tau)*dt_hr
    return dst

def compute_E_series(bz_nT, v_kms):
    """Compute the solar-wind electric-field proxy $$E = V B_s / 1000$$ in mV/m."""
    Bs = np.maximum(-bz_nT, 0.0)
    E  = v_kms*Bs/1000.0
    E[~np.isfinite(E)] = 0.0
    return E

def integrate_E(t_unix, E_series, end_idx):
    """Integrate the electric-field proxy from CME onset to a specified endpoint index."""
    if end_idx < 1:
        return 0.0
    t_hr = (t_unix[:end_idx+1]-t_unix[0])/3600.0
    E_w  = np.where(np.isfinite(E_series[:end_idx+1]), E_series[:end_idx+1], 0.0)
    return float(np.trapezoid(E_w, t_hr))

def _save_show(fig, cfg, fname):
    """Save a figure to the configured output directory and optionally display it."""
    if cfg.SAVE_PLOTS:
        out = Path(cfg.OUTPUT_DIR).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out/fname, dpi=cfg.DPI, bbox_inches="tight")
        print(f"[INFO] Saved: {out / fname}")
    if cfg.SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ----------------------------- Event processing ----------------------------- #

class EventProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events_dir = Path(cfg.EVENTS_DIR).expanduser().resolve()

    def _sc_v(self, df):
        """Return spacecraft speed magnitude from either a scalar speed column or RTN velocity components."""
        if "swa_V_mag" in df.columns:
            v = _num(df["swa_V_mag"])
            if np.isfinite(v).sum() > 0:
                return v

        comps = ["swa_V_r", "swa_V_t", "swa_V_n"]
        if all(c in df.columns for c in comps):
            vr = _num(df["swa_V_r"])
            vt = _num(df["swa_V_t"])
            vn = _num(df["swa_V_n"])
            return np.sqrt(vr**2 + vt**2 + vn**2)

        return np.full(len(df), np.nan)

    def _l1_bz_v(self, df):
        """Return L1 $$B_z$$ and speed series, preferring cross-correlated data and falling back to ballistic data."""
        if "l1_B_z_gse_xcorr" in df.columns and "l1_V_mag_xcorr" in df.columns:
            bz = _num(df["l1_B_z_gse_xcorr"])
            v = _num(df["l1_V_mag_xcorr"])
            if np.isfinite(bz).sum() > 0 and np.isfinite(v).sum() > 0:
                return bz, v

        if "l1_B_z_gse_ballistic" in df.columns and "l1_V_mag_ballistic" in df.columns:
            bz = _num(df["l1_B_z_gse_ballistic"])
            v = _num(df["l1_V_mag_ballistic"])
            if np.isfinite(bz).sum() > 0 and np.isfinite(v).sum() > 0:
                return bz, v

        return np.full(len(df), np.nan), np.full(len(df), np.nan)
    
    def _distances(self, df):
        """Compute perpendicular, downstream, and absolute spacecraft distances in AU."""
        abs_au = float(pd.to_numeric(df["sc_distance_to_earth_au"], errors="coerce").iloc[0]) if "sc_distance_to_earth_au" in df.columns else np.nan
        ang_deg = float(pd.to_numeric(df["sc_angle_from_sun_earth_line_deg"], errors="coerce").iloc[0]) if "sc_angle_from_sun_earth_line_deg" in df.columns else np.nan

        if not (np.isfinite(abs_au) and np.isfinite(ang_deg)):
            return np.nan, np.nan, np.nan

        ang = np.deg2rad(ang_deg)
        return float(abs_au * np.sin(ang)), float(abs_au * np.cos(ang)), float(abs_au)
    
    def process_event(self, path):
        """Process a single event CSV into injection-analysis metrics and supporting time series."""
        cfg = self.cfg
        fname = path.name

        df = pd.read_csv(path)

        required_columns = REQUIRED_COLUMNS_COMMON + REQUIRED_COLUMNS_SC
        if not _validate_columns(df, path, required_columns):
            return None

        if not _validate_sc_velocity_columns(df, path):
            return None

        if cfg.ONLY_CLEAR_CME and "clear_cme" in df.columns:
            val = str(df["clear_cme"].iloc[0]).strip().upper()
            if val != "TRUE":
                return None, "event excluded by ONLY_CLEAR_CME filter"

        if cfg.EXCLUDE_REPEATED and "repeated_measurements" in df.columns:
            val = str(df["repeated_measurements"].iloc[0]).strip().upper()
            if val == "TRUE":
                return None, "event excluded by EXCLUDE_REPEATED filter"

        t0 = _parse_utc(df["cme_start_utc"].iloc[0])
        t1 = _parse_utc(df["cme_end_utc"].iloc[0])
        if t1 <= t0:
            return None, "invalid CME interval (end time is not after start time)"

        t_unix = _num(df["unix_timestamp"])
        t0s = float(t0.value/1e9)
        t1s = float(t1.value/1e9)
        win = (t_unix>=t0s)&(t_unix<=t1s)&np.isfinite(t_unix)
        n_win = int(win.sum())
        if n_win < 5:
            return None, f"insufficient CME-window samples (n={n_win})"

        tw    = t_unix[win]
        order = np.argsort(tw)
        tw    = tw[order]

        if "B_n" in df.columns:
            sc_bz_full = _num(df["B_n"])
            sc_bz = sc_bz_full[win][order]
        else:
            sc_bz = np.full(tw.shape, np.nan)

        sc_v_full = self._sc_v(df)
        sc_v = sc_v_full[win][order]

        l1_bz_full, l1_v_full = self._l1_bz_v(df)
        l1_bz = l1_bz_full[win][order]
        l1_v  = l1_v_full[win][order]

        sc_dst_min = np.nan
        sc_idx_min = None
        sc_dst     = np.full(tw.shape, np.nan)
        sc_E       = np.zeros(tw.shape)
        sc_int_E   = np.nan

        n_sc_bz = int(np.isfinite(sc_bz).sum())
        n_sc_v  = int(np.isfinite(sc_v).sum())
        if n_sc_bz >= cfg.MIN_FINITE and n_sc_v >= cfg.MIN_FINITE:
            sc_dst     = predict_dst_obrien(tw, sc_bz, sc_v, dst0=cfg.DST0)
            sc_dst_min = float(np.nanmin(sc_dst))
            sc_idx_min = int(np.nanargmin(sc_dst))
            sc_E     = compute_E_series(sc_bz, sc_v)
            sc_int_E = integrate_E(tw, sc_E, sc_idx_min)

        l1_dst_min = np.nan
        l1_idx_min = None
        l1_dst     = np.full(tw.shape, np.nan)
        l1_E       = np.zeros(tw.shape)
        l1_int_E   = np.nan

        n_l1_bz = int(np.isfinite(l1_bz).sum())
        n_l1_v  = int(np.isfinite(l1_v).sum())
        if n_l1_bz >= cfg.MIN_FINITE and n_l1_v >= cfg.MIN_FINITE:
            l1_dst     = predict_dst_obrien(tw, l1_bz, l1_v, dst0=cfg.DST0)
            l1_dst_min = float(np.nanmin(l1_dst))
            l1_idx_min = int(np.nanargmin(l1_dst))
            l1_E     = compute_E_series(l1_bz, l1_v)
            l1_int_E = integrate_E(tw, l1_E, l1_idx_min)

        sc_v_at_min  = np.nan
        l1_v_at_min  = np.nan
        sc_bz_at_min = np.nan
        l1_bz_at_min = np.nan

        if sc_idx_min is not None:
            sc_v_at_min, _  = _nearest_finite(sc_v,  sc_idx_min, cfg.SNAPSHOT_SEARCH_RADIUS)
            sc_bz_at_min, _ = _nearest_finite(sc_bz, sc_idx_min, cfg.SNAPSHOT_SEARCH_RADIUS)

        if l1_idx_min is not None:
            l1_v_at_min, _  = _nearest_finite(l1_v,  l1_idx_min, cfg.SNAPSHOT_SEARCH_RADIUS)
            l1_bz_at_min, _ = _nearest_finite(l1_bz, l1_idx_min, cfg.SNAPSHOT_SEARCH_RADIUS)

        if not any([
            np.isfinite(sc_dst_min), np.isfinite(l1_dst_min),
            np.isfinite(sc_v_at_min), np.isfinite(l1_v_at_min),
            np.isfinite(sc_bz_at_min), np.isfinite(l1_bz_at_min),
            np.isfinite(sc_int_E), np.isfinite(l1_int_E)
        ]):
            return None, "no valid injection-analysis quantities could be computed"

        vswap_dst_min  = np.nan
        bzswap_dst_min = np.nan

        if n_sc_bz >= cfg.MIN_FINITE and n_l1_v >= cfg.MIN_FINITE:
            vswap_dst     = predict_dst_obrien(tw, sc_bz, l1_v, dst0=cfg.DST0)
            vswap_dst_min = float(np.nanmin(vswap_dst))

        if n_l1_bz >= cfg.MIN_FINITE and n_sc_v >= cfg.MIN_FINITE:
            bzswap_dst     = predict_dst_obrien(tw, l1_bz, sc_v, dst0=cfg.DST0)
            bzswap_dst_min = float(np.nanmin(bzswap_dst))

        perp, downstream, abs_au = self._distances(df)

        return {
            "file":           fname,
            "sc_dst_min":     sc_dst_min,     "l1_dst_min":     l1_dst_min,
            "sc_v_at_min":    sc_v_at_min,    "l1_v_at_min":    l1_v_at_min,
            "sc_bz_at_min":   sc_bz_at_min,   "l1_bz_at_min":   l1_bz_at_min,
            "sc_int_E":       sc_int_E,        "l1_int_E":       l1_int_E,
            "vswap_dst_min":  vswap_dst_min,
            "bzswap_dst_min": bzswap_dst_min,
            "_tw":            tw,
            "_sc_bz":         sc_bz,           "_l1_bz":         l1_bz,
            "_sc_v":          sc_v,            "_l1_v":          l1_v,
            "_sc_E":          sc_E,            "_l1_E":          l1_E,
            "_sc_dst":        sc_dst,          "_l1_dst":        l1_dst,
            "_sc_idx_min":    sc_idx_min,      "_l1_idx_min":    l1_idx_min,
            "perp_au":        perp,            "downstream_au":  downstream,
            "abs_au":         abs_au,
        }, None

    def run(self):
        """Process all matching event CSV files in the configured events directory."""
        files = sorted(self.events_dir.glob(self.cfg.EVENT_GLOB))
        print(f"\n[INFO] Processing {len(files)} event files...")
        rows = []

        for p in files:
            try:
                res, reason = self.process_event(p)
                if res is not None:
                    rows.append(res)
                elif reason is not None:
                    print(f"[WARN] Skipping {p.name}: {reason}")
            except Exception as exc:
                print(f"[ERROR] {p.name}: {exc}")

        df = pd.DataFrame(rows)
        print(f"[INFO] Valid events: {len(df)} / {len(files)}\n")
        return df


#  --------------------------------- Plotting --------------------------------- #

C_DST  = "#1f77b4"
C_V    = "#2ca02c"
C_BZ   = "#9467bd"
C_INTE = "#d62728"
C_SC   = "#1f77b4"
C_L1   = "#d62728"


def _parity_panel(ax, x, y, colour, xlabel, ylabel, title, cfg, rng):
    """Draw a parity plot with a bootstrap regression line, confidence band, and summary statistics."""
    mask = _finite_mask(x, y)
    n_valid = mask.sum()
    if n_valid < 3:
        ax.set_title(f"{title}\n(insufficient data: n={n_valid})", fontsize=10)
        return
    all_v = np.concatenate([x[mask], y[mask]])
    pad = 0.07*(all_v.max()-all_v.min())
    lo, hi = float(all_v.min())-pad, float(all_v.max())+pad

    ax.plot([lo, hi], [lo, hi], color="grey", lw=1.2, ls="--",
            label="y = x", zorder=1)
    ax.scatter(x[mask], y[mask], s=55, color=colour, alpha=0.85, zorder=4,
               edgecolors="white", linewidths=0.4)

    reg = _bootstrap_regression(x, y, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
    if reg is not None:
        ax.plot(reg["x_fit"], reg["y_fit"], color=colour, lw=1.8, zorder=3)
        ax.fill_between(reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                        color=colour, alpha=0.18, lw=0, zorder=2)
        sign   = "+" if reg["intercept"] >= 0 else "-"
        eq_str = (f"y = {reg['slope']:.3f}x {sign} {abs(reg['intercept']):.2f}\n"
                  f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}")
        ax.text(0.05, 0.97, eq_str, transform=ax.transAxes, fontsize=8.5,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))

    rmse = float(np.sqrt(np.nanmean((y[mask]-x[mask])**2)))
    mae  = float(np.nanmean(np.abs(y[mask]-x[mask])))
    bias = float(np.nanmean(y[mask]-x[mask]))
    ax.text(0.97, 0.05,
            f"RMSE = {rmse:.2f}\nMAE  = {mae:.2f}\nBias = {bias:+.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

def plot_2x2_parity(results, cfg):
    """Plot a 2×2 parity summary comparing L1 and spacecraft injection-related quantities."""
    rng = np.random.default_rng(cfg.RNG_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)

    def n_valid(sc_col, l1_col):
        m = np.isfinite(results[sc_col].values) & np.isfinite(results[l1_col].values)
        return int(m.sum())

    panels = [
        (axes[0, 0],
         results["l1_dst_min"].values,   results["sc_dst_min"].values,
         C_DST,  "L1 predicted min Dst [nT]",   "SC predicted min Dst [nT]",
         f"1. Min Dst  (n={n_valid('sc_dst_min','l1_dst_min')})"),
        (axes[0, 1],
         results["l1_v_at_min"].values,  results["sc_v_at_min"].values,
         C_V,    "L1 V at min Dst [km/s]",       "SC V at min Dst [km/s]",
         f"2. V at min Dst  (n={n_valid('sc_v_at_min','l1_v_at_min')})"),
        (axes[1, 0],
         results["l1_bz_at_min"].values, results["sc_bz_at_min"].values,
         C_BZ,   "L1 Bz at min Dst [nT]",        "SC Bz at min Dst [nT]",
         f"3. Bz at min Dst  (n={n_valid('sc_bz_at_min','l1_bz_at_min')})"),
        (axes[1, 1],
         results["l1_int_E"].values,     results["sc_int_E"].values,
         C_INTE, "L1 integrated E [mV/m*hr]",    "SC integrated E [mV/m*hr]",
         f"4. Integrated E  (n={n_valid('sc_int_E','l1_int_E')})"),
    ]

    for ax, x, y, colour, xlabel, ylabel, title in panels:
        _parity_panel(ax, x, y, colour, xlabel, ylabel, title, cfg, rng)

    fig.suptitle(
        "SC vs L1 injection analysis — each panel uses maximum available events\n"
        "Panels 2 & 3: instantaneous snapshot. Panel 4: integrated injection path.",
        fontsize=11,
    )
    _save_show(fig, cfg, "injection_2x2_parity.png")


# --------------------------- Time series analysis --------------------------- #

def _pick_example_event(results, cfg):
    """Select a single reproducible example event for the timeseries illustration."""
    df = results.dropna(subset=["sc_dst_min", "l1_dst_min", "sc_int_E", "l1_int_E"]).copy()
    if df.empty:
        return None

    rng = np.random.default_rng(cfg.RNG_SEED)
    idx = int(rng.integers(0, len(df)))
    return df.iloc[idx]

def plot_timeseries(results, cfg):
    """Plot a single example event showing electric-field injection and predicted Dst time series."""
    row = _pick_example_event(results, cfg)
    if row is None:
        print("[WARN] No valid event is available for the example timeseries plot.")
        return

    tw         = row["_tw"]
    sc_E       = row["_sc_E"]
    l1_E       = row["_l1_E"]
    sc_dst     = row["_sc_dst"]
    l1_dst     = row["_l1_dst"]
    sc_idx_min = row["_sc_idx_min"]
    l1_idx_min = row["_l1_idx_min"]

    if sc_idx_min is None or l1_idx_min is None:
        print("[WARN] Example event is missing valid minimum-Dst indices.")
        return

    sc_idx_min = int(sc_idx_min)
    l1_idx_min = int(l1_idx_min)

    fname = (
        str(row["file"])
        .replace("event_STEREOA_", "")
        .replace("event_Solar_Orbiter_", "SO_")
        .replace(".csv", "")
    )

    t_hr = (tw - tw[0]) / 3600.0

    fig, (ax_e, ax_d) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )

    ax_e.plot(t_hr, sc_E, color=C_SC, lw=1.5, alpha=0.9,
              label=f"SC  (intE = {row['sc_int_E']:.2f} mV/m*hr)")
    ax_e.plot(t_hr, l1_E, color=C_L1, lw=1.5, alpha=0.9, ls="--",
              label=f"L1  (intE = {row['l1_int_E']:.2f} mV/m*hr)")
    ax_e.fill_between(t_hr[:sc_idx_min+1], sc_E[:sc_idx_min+1], 0,
                      color=C_SC, alpha=0.15)
    ax_e.fill_between(t_hr[:l1_idx_min+1], l1_E[:l1_idx_min+1], 0,
                      color=C_L1, alpha=0.15)
    ax_e.axvline(t_hr[sc_idx_min], color=C_SC, lw=1.0, ls=":", alpha=0.8)
    ax_e.axvline(t_hr[l1_idx_min], color=C_L1, lw=1.0, ls=":", alpha=0.8)
    ax_e.axhline(0, color="grey", lw=0.6)
    ax_e.set_ylabel("E = V*Bs/1000 [mV/m]", fontsize=10)
    ax_e.set_title(
        f"Example event: {fname}\n"
        f"Shaded = main-phase injection.  "
        f"SC intE = {row['sc_int_E']:.2f}  |  L1 intE = {row['l1_int_E']:.2f} mV/m*hr",
        fontsize=10,
    )
    ax_e.legend(fontsize=9)
    ax_e.grid(True, alpha=0.3)

    ax_d.plot(t_hr, sc_dst, color=C_SC, lw=1.5, alpha=0.9,
              label=f"SC  (min Dst = {row['sc_dst_min']:.1f} nT)")
    ax_d.plot(t_hr, l1_dst, color=C_L1, lw=1.5, alpha=0.9, ls="--",
              label=f"L1  (min Dst = {row['l1_dst_min']:.1f} nT)")
    ax_d.scatter(t_hr[sc_idx_min], row["sc_dst_min"], color=C_SC, s=70, zorder=5)
    ax_d.scatter(t_hr[l1_idx_min], row["l1_dst_min"], color=C_L1, s=70, zorder=5)
    ax_d.axvline(t_hr[sc_idx_min], color=C_SC, lw=1.0, ls=":", alpha=0.8)
    ax_d.axvline(t_hr[l1_idx_min], color=C_L1, lw=1.0, ls=":", alpha=0.8)
    ax_d.axhline(0, color="grey", lw=0.6)
    ax_d.set_xlabel("Hours from CME onset", fontsize=10)
    ax_d.set_ylabel("Predicted Dst [nT]", fontsize=10)
    ax_d.legend(fontsize=9)
    ax_d.grid(True, alpha=0.3)

    _save_show(fig, cfg, f"timeseries_{fname}.png")


# -------------------------------- Statistics -------------------------------- #

def _p_slope_equals_1(x, y, n_boot, rng):
    """Bootstrap p-value for H0: slope = 1 (i.e. data follows y=x)."""
    reg = _bootstrap_regression(x, y, n_boot, 95.0, rng)
    if reg is None:
        return np.nan, np.nan, np.nan
    boot_slopes = reg["boot_slopes"][np.isfinite(reg["boot_slopes"])]
    ci_lo = float(np.percentile(boot_slopes, 2.5))
    ci_hi = float(np.percentile(boot_slopes, 97.5))
    p_gt  = float(np.mean(boot_slopes > 1.0))
    p_val = 2.0 * min(p_gt, 1.0 - p_gt)
    return p_val, ci_lo, ci_hi

def _stat_block(label, x, y, units, cfg, rng, pi_vals=None):
    """Print regression and error statistics for a paired SC–L1 metric comparison."""
    mask = _finite_mask(x, y)
    n_valid = int(mask.sum())
    print(f"  {label}  (n = {n_valid} events with both SC and L1 finite)")
    if n_valid < 3:
        print("    Insufficient data.\n")
        return

    reg = _bootstrap_regression(x, y, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
    if reg is None:
        print("    Bootstrap regression failed.\n")
        return

    boot_slopes = reg["boot_slopes"][np.isfinite(reg["boot_slopes"])]
    ci_lo = float(np.percentile(boot_slopes, 2.5))
    ci_hi = float(np.percentile(boot_slopes, 97.5))
    p_gt  = float(np.mean(boot_slopes > 1.0))
    p_val = 2.0*min(p_gt, 1.0-p_gt)

    slope, intercept, r2, n = reg["slope"], reg["intercept"], reg["r2"], reg["n"]
    rmse      = float(np.sqrt(np.mean((y[mask]-x[mask])**2)))
    mae       = float(np.mean(np.abs(y[mask]-x[mask])))
    bias      = float(np.mean(y[mask]-x[mask]))
    resid_std = float(np.std(y[mask]-(slope*x[mask]+intercept), ddof=1))
    sign      = "+" if intercept >= 0 else "-"

    print(f"    Fit:  y = {slope:.4f}x {sign} {abs(intercept):.4f}   R2 = {r2:.4f}")
    print(f"    95% CI on slope: [{ci_lo:.4f}, {ci_hi:.4f}]")
    if ci_lo > 1.0:
        print(f"    -> Slope SIGNIFICANTLY > 1.0")
    elif ci_hi < 1.0:
        print(f"    -> Slope SIGNIFICANTLY < 1.0")
    else:
        print(f"    -> Slope NOT significantly different from 1.0")
    p_tag = "[p<0.01]" if p_val < 0.01 else ("[p<0.05]" if p_val < 0.05 else "[n.s.]")
    print(f"    p-value (H0: slope=1, i.e. data follows y=x):  p = {p_val:.4f}  {p_tag}")
    print(f"    RMSE={rmse:.3f}  MAE={mae:.3f}  Bias={bias:+.3f}  [{units}]")
    print(f"    Mean SC = {float(np.mean(y[mask])):.3f}  |  Mean L1 = {float(np.mean(x[mask])):.3f}  [{units}]")
    if pi_vals:
        print(f"    95% PIs  (resid std = {resid_std:.3f} {units}):")
        for v in pi_vals:
            pred = slope*v+intercept
            pi   = 1.96*resid_std
            print(f"      L1 = {v:8.2f}  ->  SC = {pred:.2f} +/- {pi:.2f}  [{units}]")
    print()

def print_stats_summary(results, cfg):
    """Print the main statistical summary for the injection-analysis parity metrics."""
    rng = np.random.default_rng(cfg.RNG_SEED)
    print("\n"+"="*70)
    print("INJECTION ANALYSIS — STATISTICAL SUMMARY")
    print(f"Total events loaded: {len(results)}")
    print(f"(Each metric uses the maximum available events for that pair)")
    print("="*70+"\n")

    _stat_block("1. Min Dst [nT]",
                results["l1_dst_min"].values, results["sc_dst_min"].values,
                "nT", cfg, rng, pi_vals=[-50,-100,-150,-200])

    _stat_block("2. V at min Dst [km/s]",
                results["l1_v_at_min"].values, results["sc_v_at_min"].values,
                "km/s", cfg, rng, pi_vals=[300,400,500,600,700])

    _stat_block("3. Bz at min Dst [nT]",
                results["l1_bz_at_min"].values, results["sc_bz_at_min"].values,
                "nT", cfg, rng, pi_vals=[-5,-10,-15,-20,-30])

    _stat_block("4. Integrated E [mV/m*hr]",
                results["l1_int_E"].values, results["sc_int_E"].values,
                "mV/m*hr", cfg, rng, pi_vals=[5,10,20,40,80])

    dst_gap  = results["sc_dst_min"].values - results["l1_dst_min"].values
    intE_gap = results["sc_int_E"].values   - results["l1_int_E"].values
    mask = _finite_mask(dst_gap, intE_gap)
    if mask.sum() >= 3:
        _, _, r2_cross = _linear_fit_r2(intE_gap[mask], dst_gap[mask])
        print(f"  Cross-check: R2(integrated-E gap vs Dst gap) = {r2_cross:.4f}")
        print(f"  -> This fraction of the Dst discrepancy variance is directly")
        print(f"     explained by differences in time-integrated injection.")
    print("\n"+"="*70+"\n")


# -------------------------- Decomposition analysis -------------------------- #

def plot_decomposition_strip(results, cfg):
    """Plot a three-panel Dst decomposition comparing pure, velocity-swapped, and Bz-swapped cases."""
    rng = np.random.default_rng(cfg.RNG_SEED)

    m_raw    = _finite_mask(results["l1_dst_min"].values,  results["sc_dst_min"].values)
    m_vswap  = _finite_mask(results["l1_dst_min"].values,  results["vswap_dst_min"].values)
    m_bzswap = _finite_mask(results["l1_dst_min"].values,  results["bzswap_dst_min"].values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    n1 = int(m_raw.sum())
    n2 = int(m_vswap.sum())
    n3 = int(m_bzswap.sum())
    panels = [
        (axes[0],
         results["l1_dst_min"].values,   results["sc_dst_min"].values,    m_raw,
         C_DST,
         "L1 predicted min Dst [nT]",    "SC predicted min Dst [nT]",
         f"Panel 1: SC (pure) vs L1  (n={n1})"),
        (axes[1],
         results["l1_dst_min"].values,   results["vswap_dst_min"].values,  m_vswap,
         "#e07b00",
         "L1 predicted min Dst [nT]",    "V-swap hybrid min Dst [nT]  [SC Bz + L1 V]",
         f"Panel 2: V-swap hybrid vs L1  (n={n2})"),
        (axes[2],
         results["l1_dst_min"].values,   results["bzswap_dst_min"].values, m_bzswap,
         "#7b22c4",
         "L1 predicted min Dst [nT]",    "Bz-swap hybrid min Dst [nT]  [L1 Bz + SC V]",
         f"Panel 3: Bz-swap hybrid vs L1  (n={n3})"),
    ]

    for ax, x_full, y_full, mask, colour, xlabel, ylabel, title in panels:
        x = x_full[mask]
        y = y_full[mask]
        if len(x) < 3:
            ax.set_title(title + "\n(insufficient data)", fontsize=10)
            continue

        all_v = np.concatenate([x, y])
        pad   = 0.07 * (all_v.max() - all_v.min())
        lo, hi = float(all_v.min()) - pad, float(all_v.max()) + pad

        ax.plot([lo, hi], [lo, hi], color="grey", lw=1.2, ls="--",
                label="y = x", zorder=1)
        ax.scatter(x, y, s=60, color=colour, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.4)

        reg = _bootstrap_regression(x_full, y_full, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
        if reg is not None:
            ax.plot(reg["x_fit"], reg["y_fit"], colour, lw=1.8, zorder=3)
            ax.fill_between(reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                            color=colour, alpha=0.18, lw=0, zorder=2)

            boot_slopes = reg["boot_slopes"][np.isfinite(reg["boot_slopes"])]
            ci_lo = float(np.percentile(boot_slopes, 2.5))
            ci_hi = float(np.percentile(boot_slopes, 97.5))
            p_gt  = float(np.mean(boot_slopes > 1.0))
            p_val = 2.0 * min(p_gt, 1.0 - p_gt)
            p_tag = "***" if p_val < 0.001 else ("**" if p_val < 0.01
                    else ("*" if p_val < 0.05 else "n.s."))

            sign   = "+" if reg["intercept"] >= 0 else "-"
            eq_str = (f"y = {reg['slope']:.3f}x {sign} {abs(reg['intercept']):.2f}\n"
                      f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}\n"
                      f"p(slope=1) = {p_val:.3f} {p_tag}")
            ax.text(0.05, 0.97, eq_str, transform=ax.transAxes, fontsize=8.5,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))

        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        mae  = float(np.mean(np.abs(y - x)))
        bias = float(np.mean(y - x))
        ax.text(0.97, 0.05,
                f"RMSE = {rmse:.2f}\nMAE  = {mae:.2f}\nBias = {bias:+.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Burton model decomposition: how much of the SC–L1 Dst gap is explained by V vs Bz?\n",
        fontsize=10,
    )
    _save_show(fig, cfg, "decomposition_1x3_parity.png")

def _compare_panel_slopes(label_a, label_b, x_a, y_a, x_b, y_b, cfg, rng):
    """Bootstrap comparison of slopes between two parity panels."""
    mask_a = _finite_mask(x_a, y_a)
    mask_b = _finite_mask(x_b, y_b)
    if mask_a.sum() < 3 or mask_b.sum() < 3:
        print(f"  Comparison {label_a} vs {label_b}: insufficient data\n")
        return

    reg_a = _bootstrap_regression(x_a, y_a, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
    reg_b = _bootstrap_regression(x_b, y_b, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
    if reg_a is None or reg_b is None:
        return

    slopes_a = reg_a["boot_slopes"][np.isfinite(reg_a["boot_slopes"])]
    slopes_b = reg_b["boot_slopes"][np.isfinite(reg_b["boot_slopes"])]

    n = min(len(slopes_a), len(slopes_b))
    diff = slopes_a[:n] - slopes_b[:n]

    mean_diff = float(np.mean(diff))
    ci_lo = float(np.percentile(diff, 2.5))
    ci_hi = float(np.percentile(diff, 97.5))
    p_val = float(2.0 * min(np.mean(diff > 0), np.mean(diff < 0)))
    p_tag = "[p<0.01]" if p_val < 0.01 else ("[p<0.05]" if p_val < 0.05 else "[n.s.]")

    rmse_a = float(np.sqrt(np.mean((y_a[mask_a] - x_a[mask_a])**2)))
    rmse_b = float(np.sqrt(np.mean((y_b[mask_b] - x_b[mask_b])**2)))

    print(f"  STATISTICAL COMPARISON: {label_a} vs {label_b}")
    print(f"    Slope difference (A - B): {mean_diff:+.3f}")
    print(f"    95% CI on difference: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    p-value (H0: slopes equal between panels): p = {p_val:.4f}  {p_tag}")
    if ci_lo <= 0.0 <= ci_hi:
        print("    Slopes are not significantly different (CI includes 0).")
    else:
        print("    Slopes are significantly different (CI excludes 0).")
    print(f"    RMSE comparison:  {label_a} = {rmse_a:.1f} nT  |  {label_b} = {rmse_b:.1f} nT")
    print(f"    RMSE difference: {rmse_b - rmse_a:+.1f} nT\n")

def print_decomposition_stats(results, cfg):
    """Print regression statistics for the Dst decomposition panels and compare panel slopes."""
    rng = np.random.default_rng(cfg.RNG_SEED)

    print("\n" + "=" * 70)
    print("DECOMPOSITION ANALYSIS — Panel Statistics")
    print("=" * 70 + "\n")

    sc  = results["sc_dst_min"].values
    l1  = results["l1_dst_min"].values
    vsw = results["vswap_dst_min"].values
    bzw = results["bzswap_dst_min"].values

    for label, x_arr, y_arr in [
        ("Panel 1 — SC (pure) vs L1",      l1,  sc),
        ("Panel 2 — V-swap hybrid vs L1",  l1,  vsw),
        ("Panel 3 — Bz-swap hybrid vs L1", l1,  bzw),
    ]:
        mask = _finite_mask(x_arr, y_arr)
        if mask.sum() < 3:
            print(f"  {label}: insufficient data\n"); continue

        reg = _bootstrap_regression(x_arr, y_arr, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
        if reg is None:
            continue

        boot_slopes = reg["boot_slopes"][np.isfinite(reg["boot_slopes"])]
        ci_lo = float(np.percentile(boot_slopes, 2.5))
        ci_hi = float(np.percentile(boot_slopes, 97.5))
        p_gt  = float(np.mean(boot_slopes > 1.0))
        p_val = 2.0 * min(p_gt, 1.0 - p_gt)
        p_tag = "[p<0.01]" if p_val < 0.01 else ("[p<0.05]" if p_val < 0.05 else "[n.s.]")
        rmse  = float(np.sqrt(np.mean((y_arr[mask] - x_arr[mask]) ** 2)))
        bias  = float(np.mean(y_arr[mask] - x_arr[mask]))
        sign  = "+" if reg["intercept"] >= 0 else "-"

        print(f"  {label}  (n={reg['n']})")
        print(f"    Fit: y = {reg['slope']:.3f}x {sign} {abs(reg['intercept']):.1f}   R² = {reg['r2']:.3f}")
        print(f"    95% CI on slope: [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"    p-value (H0: slope=1, i.e. data follows y=x): p = {p_val:.4f}  {p_tag}")
        print(f"    RMSE = {rmse:.1f} nT,  Bias = {bias:+.1f} nT")

        if ci_lo <= 1.0 <= ci_hi:
            print("    Slope is consistent with 1.0; the data are consistent with y = x.")
        elif ci_hi < 1.0:
            print("    Slope is less than 1.0; the data are not consistent with y = x.")
        print()

    # --- cross-panel slope comparisons ---
    print("  --- Cross-panel slope comparisons (are two panels statistically different?) ---\n")
    _compare_panel_slopes(
        "Panel 1 (SC pure)", "Panel 3 (Bz-swap)",
        l1, sc, l1, bzw,
        cfg, rng
    )
    _compare_panel_slopes(
        "Panel 1 (SC pure)", "Panel 2 (V-swap)",
        l1, sc, l1, vsw,
        cfg, rng
    )

    print("=" * 70 + "\n")


# -------------------------- Velocity shift analysis ------------------------- #

def _xcorr_shift(sc_v, l1_v, t_unix, max_shift_hr=12.0):
    """Estimate the time lag that best aligns spacecraft and L1 velocity series using cross-correlation."""
    mask = np.isfinite(sc_v) & np.isfinite(l1_v)
    if mask.sum() < 10:
        return 0.0

    x = sc_v.copy(); x[~np.isfinite(x)] = 0.0
    y = l1_v.copy(); y[~np.isfinite(y)] = 0.0

    dt_s = float(np.median(np.diff(t_unix[np.isfinite(t_unix)])))
    if dt_s <= 0:
        return 0.0

    max_lag_samples = int(round(max_shift_hr * 3600.0 / dt_s))

    n     = len(x)
    corr  = np.correlate(y - y.mean(), x - x.mean(), mode="full")
    lags  = np.arange(-(n - 1), n)

    valid = np.abs(lags) <= max_lag_samples
    best_lag = int(lags[valid][np.argmax(corr[valid])])

    return float(best_lag * dt_s)

def _apply_shift(series, t_unix, shift_s):
    """Apply a time shift to a sampled series by interpolation onto the original time grid."""
    if shift_s == 0.0:
        return series.copy()
    t_shifted = t_unix - shift_s
    finite    = np.isfinite(series) & np.isfinite(t_unix)
    if finite.sum() < 3:
        return np.full_like(series, np.nan)
    return np.interp(t_shifted,
                     t_unix[finite], series[finite],
                     left=np.nan, right=np.nan)

def velocity_shift_dst_analysis(results, cfg):
    """Evaluate how velocity time-shifting affects Burton-model Dst predictions relative to L1."""
    rng = np.random.default_rng(cfg.RNG_SEED)

    print("\n" + "=" * 70)
    print("VELOCITY SHIFT ANALYSIS")
    print("  DRO V shifted by xcorr lag to align with L1 V,")
    print("  then Burton run with shifted V + original DRO Bz")
    print("=" * 70)

    shift_records = []

    for _, row in results.iterrows():
        tw    = row["_tw"]
        sc_bz = row["_sc_bz"]
        sc_v  = row["_sc_v"]
        l1_v  = row["_l1_v"]
        l1_dst_min = row["l1_dst_min"]

        if not np.isfinite(l1_dst_min):
            continue
        if not (np.isfinite(sc_bz).sum() >= cfg.MIN_FINITE and
                np.isfinite(sc_v).sum()  >= cfg.MIN_FINITE and
                np.isfinite(l1_v).sum()  >= cfg.MIN_FINITE):
            continue

        shift_s = _xcorr_shift(sc_v, l1_v, tw, max_shift_hr=12.0)
        sc_v_shifted = _apply_shift(sc_v, tw, shift_s)

        if np.isfinite(sc_v_shifted).sum() < cfg.MIN_FINITE:
            continue

        dst_shifted     = predict_dst_obrien(tw, sc_bz, sc_v_shifted, dst0=cfg.DST0)
        dst_shifted_min = float(np.nanmin(dst_shifted))

        shift_records.append({
            "file":             row["file"],
            "l1_dst_min":       l1_dst_min,
            "sc_dst_min":       row["sc_dst_min"],
            "shifted_dst_min":  dst_shifted_min,
            "shift_hr":         shift_s / 3600.0,
        })

    if not shift_records:
        print("  No valid events for velocity shift analysis.")
        return

    df = pd.DataFrame(shift_records)

    print(f"\n  Valid events: {len(df)}")
    print(f"  Mean shift applied: {df['shift_hr'].mean():+.2f} hr  "
          f"(std {df['shift_hr'].std():.2f} hr)")
    print(f"  Range: [{df['shift_hr'].min():.2f}, {df['shift_hr'].max():.2f}] hr\n")

    x = df["l1_dst_min"].values
    y = df["shifted_dst_min"].values
    _stat_block_simple("Shifted-V hybrid vs L1", x, y, "nT", cfg, rng)

    x2 = df["l1_dst_min"].values
    y2 = df["sc_dst_min"].values
    _stat_block_simple("Pure SC vs L1 (same events)", x2, y2, "nT", cfg, rng)

    print("=" * 70 + "\n")

    _plot_vshift_parity(df, cfg, rng)

def _stat_block_simple(label, x, y, units, cfg, rng):
    """Print a compact regression summary for a paired SC–L1 Dst comparison."""
    mask = _finite_mask(x, y)
    n    = int(mask.sum())
    print(f"  {label}  (n={n})")
    if n < 3:
        print("    Insufficient data.\n")
        return
    reg = _bootstrap_regression(x, y, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_CI, rng)
    if reg is None:
        print("    Bootstrap regression failed.\n")
        return
    boot_slopes = reg["boot_slopes"][np.isfinite(reg["boot_slopes"])]
    ci_lo = float(np.percentile(boot_slopes, 2.5))
    ci_hi = float(np.percentile(boot_slopes, 97.5))
    p_gt  = float(np.mean(boot_slopes > 1.0))
    p_val = 2.0 * min(p_gt, 1.0 - p_gt)
    p_tag = "[p<0.01]" if p_val < 0.01 else ("[p<0.05]" if p_val < 0.05 else "[n.s.]")
    rmse  = float(np.sqrt(np.mean((y[mask] - x[mask]) ** 2)))
    bias  = float(np.mean(y[mask] - x[mask]))
    sign  = "+" if reg["intercept"] >= 0 else "-"
    print(f"    Fit: y = {reg['slope']:.3f}x {sign} {abs(reg['intercept']):.2f}  "
          f"R²={reg['r2']:.3f}")
    print(f"    95% CI on slope: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    p-value (H0: slope=1, i.e. data follows y=x): p = {p_val:.4f}  {p_tag}")
    print(f"    RMSE={rmse:.2f} {units}  Bias={bias:+.2f} {units}")
    if ci_lo <= 1.0 <= ci_hi:
        print(f"    ✓ Slope consistent with 1.0 — data follows y=x")
    elif ci_hi < 1.0:
        print(f"    ✗ Slope < 1.0 — data does NOT follow y=x")
    else:
        print(f"    ✗ Slope > 1.0 — data does NOT follow y=x")
    print()

def _plot_vshift_parity(df, cfg, rng):
    """Plot parity comparisons for the pure spacecraft case and the velocity-shifted hybrid case."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    panels = [
        (axes[0], df["l1_dst_min"].values, df["sc_dst_min"].values,
         C_DST, "SC predicted min Dst [nT]",
         f"Panel 1: Pure SC vs L1  (n={len(df)})"),
        (axes[1], df["l1_dst_min"].values, df["shifted_dst_min"].values,
         "#e07b00", "Shifted-V hybrid min Dst [nT]\n[DRO Bz + xcorr-shifted DRO V]",
         f"Panel 2: Shifted-V hybrid vs L1  (n={len(df)})"),
    ]

    all_vals = np.concatenate([
        df["l1_dst_min"].values,
        df["sc_dst_min"].values,
        df["shifted_dst_min"].values,
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    pad  = 0.07 * (all_vals.max() - all_vals.min())
    lo   = float(all_vals.min()) - pad
    hi   = float(all_vals.max()) + pad

    for ax, x_full, y_full, colour, ylabel, title in panels:
        mask = _finite_mask(x_full, y_full)
        x = x_full[mask]; y = y_full[mask]

        ax.plot([lo, hi], [lo, hi], color="grey", lw=1.2, ls="--",
                label="y = x", zorder=1)
        ax.scatter(x, y, s=60, color=colour, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.4)

        reg = _bootstrap_regression(x_full, y_full, cfg.BOOTSTRAP_N,
                                    cfg.BOOTSTRAP_CI, rng)
        if reg is not None:
            ax.plot(reg["x_fit"], reg["y_fit"], color=colour, lw=1.8, zorder=3)
            ax.fill_between(reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                            color=colour, alpha=0.18, lw=0, zorder=2)

            boot_slopes = reg["boot_slopes"][np.isfinite(reg["boot_slopes"])]
            ci_lo = float(np.percentile(boot_slopes, 2.5))
            ci_hi = float(np.percentile(boot_slopes, 97.5))
            p_gt  = float(np.mean(boot_slopes > 1.0))
            p_val = 2.0 * min(p_gt, 1.0 - p_gt)
            p_tag = "***" if p_val < 0.001 else ("**" if p_val < 0.01
                    else ("*" if p_val < 0.05 else "n.s."))

            sign   = "+" if reg["intercept"] >= 0 else "-"
            eq_str = (f"y = {reg['slope']:.3f}x {sign} {abs(reg['intercept']):.2f}\n"
                      f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}\n"
                      f"p(slope=1) = {p_val:.3f} {p_tag}")
            ax.text(0.05, 0.97, eq_str, transform=ax.transAxes, fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                              alpha=0.85))

        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))
        ax.text(0.97, 0.05,
                f"RMSE = {rmse:.2f} nT\nBias = {bias:+.2f} nT",
                transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("L1 predicted min Dst [nT]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Effect of xcorr velocity shift on Dst prediction\n"
        "Panel 2 uses DRO Bz with DRO V time-shifted to best align with L1 V",
        fontsize=11,
    )
    _save_show(fig, cfg, "vshift_dst_parity.png")


# ----------------------------------------------------------------------------- #

def main():
    """Run the full injection-analysis workflow and generate configured outputs."""
    cfg       = Config()
    processor = EventProcessor(cfg)
    results   = processor.run()

    if results.empty:
        print("[ERROR] No valid events were available for injection analysis.")
        return

    public = [c for c in results.columns if not c.startswith("_")]
    print(f"[INFO] Computed results for {len(results)} valid events.")

    print_stats_summary(results, cfg)
    plot_2x2_parity(results, cfg)
    plot_timeseries(results, cfg)

    plot_decomposition_strip(results, cfg)
    print_decomposition_stats(results, cfg)

    velocity_shift_dst_analysis(results, cfg)

if __name__ == "__main__":
    main()