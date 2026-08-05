"""
bmag_powerlaw.py
================

Analysis module for power-law normalisation of magnetic-field magnitude
and geometric decomposition of spacecraft-L1 RMSE across CME events.

Loads event-level CSV files produced by primary.py, sweeps over a range of
radial power-law exponents (B ~ r^alpha) to find the normalisation that best
removes the distance-dependent trend in RMSE, and generates summary figures
showing the alpha sweep and normalised RMSE versus upstream, perpendicular,
and total spacecraft-Earth separation distance.

Required CSV Schema
-------------------
Each event_*.csv file must contain the following columns:

timestamp                        : ISO 8601 datetime string
xcorr_valid                      : float, 1.0 if cross-correlation is valid
cme_start_utc                    : ISO 8601 datetime string, CME interval start
cme_end_utc                      : ISO 8601 datetime string, CME interval end
B_mag                            : float, spacecraft |B| [nT]
l1_B_mag_xcorr                   : float, L1 |B| after cross-correlation shift [nT]
sc_distance_to_earth_au          : float, spacecraft distance to Earth [AU]
sc_angle_from_sun_earth_line_deg : float, angle from Sun-Earth line [deg]

Optional columns:

sc_heliocentric_distance_au      : float, spacecraft heliocentric distance [AU]
                                   If absent, this is estimated from geometry.

These files are produced by primary.py with SAVE_DATA_CSV = True.

Configuration
-------------
Set EVENTS_DIR and SAVE_FOLDER in the configuration block at the top of this
module to point to your local CSV folder and desired output folder respectively
before running.

Outputs
-------
Two PDF figures are produced if the corresponding save toggles are enabled:

    bmag_alpha_sweep.pdf        : RMSE slope improvement (%) vs power-law
                                  exponent alpha, for upstream, perpendicular,
                                  and total distance metrics.
    bmag_rmse_normalised.pdf    : RMSE vs distance for unnormalised data and
                                  three normalisation choices (alpha = -1, -2,
                                  and the optimal alpha per distance metric).

References
----------
Parker, E. N. (1958). Dynamics of the interplanetary gas and magnetic fields.
    The Astrophysical Journal, 128, 664.
    (Radial dependence of solar wind magnetic field magnitude.)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Folder containing event_*.csv files produced by primary.py
EVENTS_DIR  = "/Users/henryhodges/Documents/Year 4/Masters/Code/figures/big file/csv"

# Folder where analysis figures will be saved
SAVE_FOLDER = "/Users/henryhodges/Documents/Year 4/Masters/Report figures"

# Toggle PDF saving for all output figures
SAVE_PLOTS_PDF = False

# Reference heliocentric distance [AU] for power-law normalisation.
# Set to ~0.99 AU to match the approximate heliocentric distance of L1,
# so spacecraft fields are normalised to the same radial distance as the
# L1 measurement before comparison.
R_REF = 0.99


# ----------------------------- Helper functions ----------------------------- #

class Utilities:
    """
    Static utility methods for file I/O, datetime handling, statistics,
    geometry, and event processing used by the power-law normalisation
    analysis pipeline.
    """

    @staticmethod
    def save(fig, filename):
        save_dir = Path(SAVE_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")

    # --------------------------------- Datetime --------------------------------- #

    @staticmethod
    def strip_tz(series: pd.Series) -> pd.Series:
        if series.dt.tz is not None:
            return series.dt.tz_convert(None)
        return series

    @staticmethod
    def naive_ts(value) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isnull(ts):
            return ts
        if ts.tzinfo is not None:
            return ts.tz_convert(None)
        return ts

    # -------------------------------- Statistics -------------------------------- #

    @staticmethod
    def pearson_corr(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            return np.nan
        return pearsonr(x[mask], y[mask])[0]

    @staticmethod
    def rmse_calc(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            return np.nan
        return float(np.sqrt(np.mean((x[mask] - y[mask])**2)))

    @staticmethod
    def linear_fit(x: np.ndarray, y: np.ndarray):
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            return None
        xx, yy = x[mask], y[mask]
        slope, intercept = np.polyfit(xx, yy, 1)
        y_pred = slope * xx + intercept
        ss_res = np.sum((yy - y_pred)**2)
        ss_tot = np.sum((yy - np.mean(yy))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {"slope": slope, "intercept": intercept, "r2": r2, "n": len(xx)}

    # --------------------------------- Geometry --------------------------------- #

    @staticmethod
    def geometry(df: pd.DataFrame):
        total     = df["sc_distance_to_earth_au"].iloc[0]
        angle_rad = np.deg2rad(df["sc_angle_from_sun_earth_line_deg"].iloc[0])
        upstream  = total * np.cos(angle_rad)
        perp      = total * np.sin(angle_rad)
        return float(upstream), float(perp), float(total)

    @staticmethod
    def calculate_helio_angle(r_sc, r_earth, d_sep):
        """
        Compute the heliocentric angle between spacecraft and Earth using
        the law of cosines.

        Parameters
        ----------
        r_sc : float
            Spacecraft heliocentric distance [AU].
        r_earth : float
            Earth heliocentric distance [AU].
        d_sep : float
            Spacecraft-Earth separation distance [AU].

        Returns
        -------
        float
            Heliocentric angle [degrees].
        """
        cos_theta = (r_sc**2 + r_earth**2 - d_sep**2) / (2 * r_sc * r_earth)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_theta))

    @staticmethod
    def normalize_by_power_law(b_data: np.ndarray, r: float, alpha: float, r_ref: float = 1.0) -> np.ndarray:
        return b_data * (r / r_ref)**(-alpha)

    # ----------------------------- Event processing ----------------------------- #

    REQUIRED_COLUMNS = {
        "timestamp",
        "xcorr_valid",
        "cme_start_utc",
        "cme_end_utc",
        "B_mag",
        "l1_B_mag_xcorr",
        "sc_distance_to_earth_au",
        "sc_angle_from_sun_earth_line_deg",
    }

    @staticmethod
    def process_event(filepath: str):
        """
        Load and process a single event CSV file.

        Returns a dict of computed metrics, or None if the file does not
        contain sufficient valid data or is missing required columns.

        Parameters
        ----------
        filepath : str or Path
            Path to an event_*.csv file produced by primary.py.

        Returns
        -------
        dict or None
        """
        df = pd.read_csv(filepath)

        missing = Utilities.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"[WARN] Skipping {Path(filepath).name}: "
                  f"missing columns: {sorted(missing)}")
            return None

        df = df[df["xcorr_valid"].astype(float) > 0].copy()
        if len(df) < 20:
            return None

        df["timestamp"] = Utilities.strip_tz(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
        cme_start = Utilities.naive_ts(df["cme_start_utc"].iloc[0])
        cme_end   = Utilities.naive_ts(df["cme_end_utc"].iloc[0])
        cme = df[(df["timestamp"] >= cme_start) & (df["timestamp"] <= cme_end)]

        upstream, perp, total = Utilities.geometry(df)

        if "sc_heliocentric_distance_au" in df.columns:
            r_sc = df["sc_heliocentric_distance_au"].iloc[0]
        else:
            angle_rad = np.deg2rad(df["sc_angle_from_sun_earth_line_deg"].iloc[0])
            r_sc = np.sqrt(1.0**2 + total**2 - 2*1.0*total*np.cos(angle_rad))

        angle_helio_deg    = Utilities.calculate_helio_angle(r_sc, 1.0, total)
        angle_from_se_line = df["sc_angle_from_sun_earth_line_deg"].iloc[0]

        return {
            "upstream":           upstream,
            "perp":               perp,
            "total":              total,
            "r_sc":               r_sc,
            "angle_helio_deg":    angle_helio_deg,
            "angle_from_se_line": angle_from_se_line,
            "cme_r":              Utilities.pearson_corr(cme["B_mag"].values, cme["l1_B_mag_xcorr"].values),
            "cme_rmse":           Utilities.rmse_calc(cme["B_mag"].values, cme["l1_B_mag_xcorr"].values),
            "sc_b":               cme["B_mag"].values,
            "l1_b":               cme["l1_B_mag_xcorr"].values,
            "B_SC_mean":          np.nanmean(cme["B_mag"].values),
            "B_L1_mean":          np.nanmean(cme["l1_B_mag_xcorr"].values),
        }

# ------------------------------- Data loading ------------------------------- #

class EventLoader:
    """
    Loads and validates event CSV files produced by primary.py.

    Parameters
    ----------
    data_directory : str
        Path to the folder containing event_*.csv files.
    """

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.files = list(Path(data_directory).glob("event_*.csv"))
        self.events: list = []

    def load(self) -> list:
        """
        Load all event files from data_directory.

        Returns
        -------
        list of dict
            Each entry is the result dict returned by Utilities.process_event().
        """

        for f in self.files:
            try:
                result = Utilities.process_event(f)
                if result is not None:
                    self.events.append(result)
            except Exception as exc:
                print(f"[WARN] {f.name:40s} | {exc}")

        print(f"\nProcessed {len(self.events)} events successfully\n")

        if len(self.events) == 0:
            raise ValueError(
                f"No valid events found in: {self.data_directory}\n"
                "Check that the directory contains event_*.csv files "
                "produced by primary.py with SAVE_DATA_CSV = True."
            )

        return self.events

# ---------------------------- Geometric analysis ---------------------------- #

class GeometricAnalysis:
    """
    Computes and prints summary statistics for the geometric distribution
    of spacecraft positions across loaded events.

    Parameters
    ----------
    events : list of dict
        Event list returned by EventLoader.load().
    """

    def __init__(self, events: list):
        self.events          = events
        self.baseline_slopes = {}

        self.upstream_vals    = np.array([e['upstream']           for e in events])
        self.perp_vals        = np.array([e['perp']               for e in events])
        self.total_vals       = np.array([e['total']              for e in events])
        self.angle_helio_vals = np.array([e['angle_helio_deg']    for e in events])
        self.angle_se_vals    = np.array([e['angle_from_se_line'] for e in events])

    def run(self):
        """
        Run the full geometric analysis and print results to stdout.

        Computes distance ranges, inter-distance correlations, baseline
        RMSE slopes, and a perpendicular-vs-upstream hypothesis test.
        Populates self.baseline_slopes for use by downstream analysis.
        """


        mask = (np.isfinite(self.upstream_vals) &
                np.isfinite(self.perp_vals) &
                np.isfinite(self.total_vals))



        for dist_type in ['upstream', 'perp', 'total']:
            distances = np.array([e[dist_type]  for e in self.events])
            values    = np.array([e['cme_rmse'] for e in self.events])
            fit = Utilities.linear_fit(distances, values)
            if fit:
                self.baseline_slopes[dist_type] = fit['slope']

# ------------------------------ Alpha Analysis ------------------------------ #

class AlphaSweep:
    """
    Sweeps over a range of radial power-law exponents to find the normalisation
    that best removes the distance-dependent trend in RMSE.

    Parameters
    ----------
    events : list of dict
        Event list returned by EventLoader.load().
    baseline_slopes : dict
        Baseline RMSE slopes per distance type, from GeometricAnalysis.baseline_slopes.
    alpha_range : np.ndarray, optional
        Array of alpha values to sweep over. Defaults to linspace(-3.0, -0.1, 100).
    """

    def __init__(self, events: list, baseline_slopes: dict,
                 alpha_range: np.ndarray = None):
        self.events          = events
        self.baseline_slopes = baseline_slopes
        self.alpha_range     = alpha_range if alpha_range is not None else np.linspace(-3.0, -0.1, 100)

        self.sweep_results   = {'upstream': [], 'perp': [], 'total': []}
        self.optimal_alphas  = {}
        self.max_improvements = {}

    def run(self):
        """
        Run the alpha sweep and print optimal values to stdout.

        Populates self.optimal_alphas and self.max_improvements for use
        by downstream analysis and plotting.
        """

        for alpha_test in self.alpha_range:
            test_data = {'upstream': [], 'perp': [], 'total': []}

            for event in self.events:
                sc_b_norm = Utilities.normalize_by_power_law(
                    event['sc_b'], event['r_sc'], alpha_test, R_REF
                )
                rmse_norm = Utilities.rmse_calc(sc_b_norm, event['l1_b'])
                for dist_type in ['upstream', 'perp', 'total']:
                    test_data[dist_type].append({'distance': event[dist_type], 'rmse': rmse_norm})

            for dist_type in ['upstream', 'perp', 'total']:
                distances = np.array([d['distance'] for d in test_data[dist_type]])
                values    = np.array([d['rmse']     for d in test_data[dist_type]])
                fit = Utilities.linear_fit(distances, values)
                if fit and dist_type in self.baseline_slopes:
                    improvement = ((self.baseline_slopes[dist_type] - fit['slope']) /
                                   self.baseline_slopes[dist_type]) * 100
                    self.sweep_results[dist_type].append(improvement)
                else:
                    self.sweep_results[dist_type].append(np.nan)

        for dist_type in ['upstream', 'perp', 'total']:
            improvements = np.array(self.sweep_results[dist_type])
            if np.isfinite(improvements).any():
                max_idx = np.nanargmax(improvements)
                self.optimal_alphas[dist_type]   = self.alpha_range[max_idx]
                self.max_improvements[dist_type] = improvements[max_idx]


                
# ------------------------------ Normalised RMSE ----------------------------- #

class NormalisedRMSE:
    """
    Computes RMSE under several power-law normalisations and prints a
    slope comparison table.

    Parameters
    ----------
    events : list of dict
        Event list returned by EventLoader.load().
    optimal_alphas : dict
        Optimal alpha per distance type, from AlphaSweep.optimal_alphas.
    """

    def __init__(self, events: list, optimal_alphas: dict):
        self.events         = events
        self.optimal_alphas = optimal_alphas
        self.results        = {
            'unnormalized':       {'upstream': [], 'perp': [], 'total': []},
            'alpha_1r':           {'upstream': [], 'perp': [], 'total': []},
            'alpha_1r2':          {'upstream': [], 'perp': [], 'total': []},
            'alpha_opt_upstream': {'upstream': [], 'perp': [], 'total': []},
            'alpha_opt_perp':     {'upstream': [], 'perp': [], 'total': []},
            'alpha_opt_total':    {'upstream': [], 'perp': [], 'total': []},
        }

    def run(self):
        """
        Compute normalised RMSE for all events and normalisations, then
        print a slope comparison table to stdout.

        Populates self.results for use by the plotting class.
        """
        for event in self.events:
            sc_b, l1_b, r_sc = event['sc_b'], event['l1_b'], event['r_sc']

            for dist_type in ['upstream', 'perp', 'total']:
                self.results['unnormalized'][dist_type].append(
                    {'distance': event[dist_type], 'rmse': event['cme_rmse']})

            for key, alpha in [('alpha_1r', -1.0), ('alpha_1r2', -2.0)]:
                sc_b_norm = Utilities.normalize_by_power_law(sc_b, r_sc, alpha, R_REF)
                rmse_norm = Utilities.rmse_calc(sc_b_norm, l1_b)
                for dist_type in ['upstream', 'perp', 'total']:
                    self.results[key][dist_type].append(
                        {'distance': event[dist_type], 'rmse': rmse_norm})

            for dist_type in ['upstream', 'perp', 'total']:
                if dist_type in self.optimal_alphas:
                    sc_b_norm = Utilities.normalize_by_power_law(
                        sc_b, r_sc, self.optimal_alphas[dist_type], R_REF
                    )
                    rmse_norm = Utilities.rmse_calc(sc_b_norm, l1_b)
                    self.results[f'alpha_opt_{dist_type}'][dist_type].append(
                        {'distance': event[dist_type], 'rmse': rmse_norm})

        for dist_type, dist_label in [
            ('upstream', 'UPSTREAM'),
            ('perp',     'PERPENDICULAR'),
            ('total',    'TOTAL'),
        ]:
            baseline_fit = None
            for key, label in [
                ('unnormalized',           'Unnormalized'),
                ('alpha_1r',               'α = -1.0 (1/r)'),
                ('alpha_1r2',              'α = -2.0 (1/r²)'),
                (f'alpha_opt_{dist_type}', f'α = {self.optimal_alphas.get(dist_type, 0):.3f} (optimal)'),
            ]:
                data      = self.results[key][dist_type]
                distances = np.array([d['distance'] for d in data])
                values    = np.array([d['rmse']     for d in data])
                fit = Utilities.linear_fit(distances, values)
                if fit:
                    if key == 'unnormalized':
                        baseline_fit = fit
                        improvement  = "(baseline)"
                    elif baseline_fit:
                        pct         = ((baseline_fit['slope'] - fit['slope']) / baseline_fit['slope']) * 100
                        improvement = f"({baseline_fit['slope'] - fit['slope']:+.2f} nT/AU, {pct:+.1f}%)"
                    else:
                        improvement = ""


# --------------------------------- Plotting --------------------------------- #

class Plotter:
    """
    Generates and saves summary figures for the power-law normalisation
    analysis.

    Parameters
    ----------
    sweep : AlphaSweep
        Completed AlphaSweep instance (after .run() has been called).
    normalised : NormalisedRMSE
        Completed NormalisedRMSE instance (after .run() has been called).
    """

    DIST_TYPES  = ['upstream', 'perp', 'total']
    DIST_LABELS = [
        'Upstream Distance (AU)',
        'Perpendicular Distance (AU)',
        'Total Distance (AU)',
    ]
    COLORS = {
        'unnormalized': '#34495E',
        'alpha_1r':     '#E74C3C',
        'alpha_1r2':    '#3498DB',
        'optimal':      '#F39C12',
    }

    def __init__(self, sweep: AlphaSweep, normalised: NormalisedRMSE):
        self.sweep      = sweep
        self.normalised = normalised

    def plot_alpha_sweep(self):
        """Plot RMSE slope improvement (%) vs power-law exponent alpha."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

        for ax, dist_type, dist_label in zip(axes, self.DIST_TYPES, self.DIST_LABELS):
            improvements = np.array(self.sweep.sweep_results[dist_type])

            ax.plot(self.sweep.alpha_range, improvements, color='#34495E', linewidth=1.5,
                    label='RMSE slope improvement', zorder=3)
            ax.axhline(0, color='#95A5A6', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

            for alpha_val, color, label in [
                (-1.0, '#E74C3C', r'$\alpha = -1.0$'),
                (-2.0, '#3498DB', r'$\alpha = -2.0$'),
            ]:
                idx = np.argmin(np.abs(self.sweep.alpha_range - alpha_val))
                ax.scatter([alpha_val], [improvements[idx]], c=color, s=50,
                           zorder=5, edgecolors='k', linewidths=0.4, label=label)

            if dist_type in self.sweep.optimal_alphas:
                opt_alpha = self.sweep.optimal_alphas[dist_type]
                ax.scatter([opt_alpha], [self.sweep.max_improvements[dist_type]],
                           c='#F39C12', s=80, marker='*', zorder=6,
                           edgecolors='k', linewidths=0.4,
                           label=f'Optimal: α = {opt_alpha:.2f}')

            ax.set_xlabel(dist_label, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=9, loc='best')

        axes[0].set_ylabel('RMSE slope improvement (%)', fontsize=11)
        plt.tight_layout()

        if SAVE_PLOTS_PDF:
            Utilities.save(fig, "bmag_alpha_sweep.pdf")
        plt.show()
        plt.close(fig)

    def plot_rmse_normalised(self):
        """Plot RMSE vs distance for unnormalised and normalised data."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

        for ax, dist_type, dist_label in zip(axes, self.DIST_TYPES, self.DIST_LABELS):
            plot_specs = [
                ('unnormalized',           self.COLORS['unnormalized'], 'Unnormalized'),
                ('alpha_1r',               self.COLORS['alpha_1r'],     r'$\alpha = -1.0$'),
                ('alpha_1r2',              self.COLORS['alpha_1r2'],    r'$\alpha = -2.0$'),
                (f'alpha_opt_{dist_type}', self.COLORS['optimal'],
                 f'$\\alpha = {self.sweep.optimal_alphas.get(dist_type, 0):.2f}$ (optimal)'),
            ]

            for key, color, label in plot_specs:
                data      = self.normalised.results[key][dist_type]
                distances = np.array([d['distance'] for d in data])
                values    = np.array([d['rmse']     for d in data])

                mask = np.isfinite(distances) & np.isfinite(values)
                if mask.sum() < 3:
                    continue

                ax.scatter(distances[mask], values[mask], c=color, label=label,
                           alpha=0.7, edgecolors='k', linewidths=0.4, s=50, zorder=3)

                fit = Utilities.linear_fit(distances, values)
                if fit is not None:
                    x_line = np.linspace(distances[mask].min(), distances[mask].max(), 100)
                    ax.plot(x_line, fit['slope'] * x_line + fit['intercept'],
                            c=color, linewidth=1.5, alpha=0.8, zorder=4)

            ax.set_xlabel(dist_label, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=9, loc='best')

        axes[0].set_ylabel('RMSE [nT]', fontsize=11)
        plt.tight_layout()

        if SAVE_PLOTS_PDF:
            Utilities.save(fig, "bmag_rmse_normalised.pdf")
        plt.show()
        plt.close(fig)

    def run(self):
        """Generate all figures."""
        self.plot_alpha_sweep()
        self.plot_rmse_normalised()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    loader= EventLoader(EVENTS_DIR)
    events= loader.load()

    geo= GeometricAnalysis(events)
    geo.run()

    sweep= AlphaSweep(events, geo.baseline_slopes)
    sweep.run()

    normalised= NormalisedRMSE(events, sweep.optimal_alphas)
    normalised.run()

    plotter= Plotter(sweep, normalised)
    plotter.run()

    print("Analysis complete")