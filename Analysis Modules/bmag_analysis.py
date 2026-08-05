"""
bmag_analysis.py
================

Analysis module for comparing magnetic-field magnitude behaviour between
L1 measurements and spacecraft measurements during identified CME periods.

Loads event-level CSV files produced by primary.py, computes correlation
and error metrics for |B| during CME windows, and generates summary figures
showing peak-field parity, Pearson correlation versus distance, and RMSE
versus distance.

Required CSV Schema
-------------------
Each event_*.csv file must contain the following columns:

timestamp:                          ISO 8601 datetime string
xcorr_valid:                        float, 1.0 if cross-correlation is valid
cme_start_utc:                      ISO 8601 datetime string, CME interval start
cme_end_utc:                        ISO 8601 datetime string, CME interval end
B_mag:                              float, spacecraft |B| [nT]
l1_B_mag_xcorr:                     float, L1 |B| after cross-correlation shift [nT]
swa_V_mag:                          float, spacecraft solar wind speed [km/s]
sc_distance_to_earth_au:            float, spacecraft distance to Earth [AU]
sc_angle_from_sun_earth_line_deg:   float, angle from Sun-Earth line [deg]

These files are produced by primary.py with SAVE_DATA_CSV = True.

Configuration
-------------
Set DATA_PATH and SAVE_FOLDER in the configuration block below before running,
or pass DATA_PATH directly to AbsoluteBCorrelationAnalysis().

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, t
from sklearn.linear_model import LinearRegression


# ------------------------------ Configurations ------------------------------ #

# Folder containing event_*.csv files produced by primary.py
DATA_PATH   = "/Users/henryhodges/Documents/Year 4/Masters/Code/figures/big file/csv"

# Folder where analysis figures will be saved
SAVE_FOLDER = "/Users/henryhodges/Documents/Year 4/Masters/Report figures"

# Toggle PDF saving for each figure
SAVE_PLOT_PDF = False

# --------------------------------- Analysis --------------------------------- #

class AbsoluteBCorrelationAnalysis:

    # Columns that must be present in each event CSV
    REQUIRED_COLUMNS = {
        "timestamp",
        "xcorr_valid",
        "cme_start_utc",
        "cme_end_utc",
        "B_mag",
        "l1_B_mag_xcorr",
        "swa_V_mag",
        "sc_distance_to_earth_au",
        "sc_angle_from_sun_earth_line_deg",
    }

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.files = list(Path(data_directory).glob("event_*.csv"))
        self.results: list = []

    # --------------------------------- Utilities -------------------------------- #

    @staticmethod
    def _strip_tz(series: pd.Series) -> pd.Series:
        if series.dt.tz is not None:
            return series.dt.tz_convert(None)
        return series

    @staticmethod
    def _naive_ts(value) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isnull(ts):
            return ts
        if ts.tzinfo is not None:
            return ts.tz_convert(None)
        return ts

    # -------------------------------- Statistics -------------------------------- #

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            return np.nan
        return pearsonr(x[mask], y[mask])[0]

    @staticmethod
    def _rmse(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            return np.nan
        return float(np.sqrt(np.mean((x[mask] - y[mask])**2)))

    @staticmethod
    def _percentile(x: np.ndarray, p: float, min_n: int = 20) -> float:
        mask = np.isfinite(x)
        if mask.sum() < min_n:
            return np.nan
        return float(np.percentile(x[mask], p))

    @staticmethod
    def _mean_velocity(df: pd.DataFrame) -> float:
        v = df["swa_V_mag"].values
        mask = np.isfinite(v)
        if mask.sum() < 5:
            return np.nan
        return float(np.mean(v[mask]))

    # --------------------------------- Geometry --------------------------------- #

    @staticmethod
    def _geometry(df: pd.DataFrame):
        total     = df["sc_distance_to_earth_au"].iloc[0]
        angle_rad = np.deg2rad(df["sc_angle_from_sun_earth_line_deg"].iloc[0])
        upstream  = total * np.cos(angle_rad)
        perp      = total * np.sin(angle_rad)
        return float(upstream), float(perp), float(total)

    # ----------------------------- Event processing ----------------------------- #

    def _process_event(self, filepath: str):
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

        # Validate required columns before any processing
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"  [WARN] Skipping {Path(filepath).name}: "
                f"missing columns: {sorted(missing)}")
            return None

        df = df[df["xcorr_valid"].astype(float) > 0].copy()
        if len(df) < 20:
            return None

        df["timestamp"] = self._strip_tz(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
        cme_start = self._naive_ts(df["cme_start_utc"].iloc[0])
        cme_end   = self._naive_ts(df["cme_end_utc"].iloc[0])

        cme = df[(df["timestamp"] >= cme_start) & (df["timestamp"] <= cme_end)]

        upstream, perp, total = self._geometry(df)

        cme_v_mean = self._mean_velocity(cme)
        sc_peak_b  = self._percentile(cme["B_mag"].values, 95)
        l1_peak_b  = self._percentile(cme["l1_B_mag_xcorr"].values, 95)

        return {
            "upstream":   upstream,
            "perp":       perp,
            "total":      total,
            "cme_v_mean": cme_v_mean,
            "cme_r":      self._pearson(cme["B_mag"].values, cme["l1_B_mag_xcorr"].values),
            "cme_rmse":   self._rmse(cme["B_mag"].values,   cme["l1_B_mag_xcorr"].values),
            "sc_peak_b":  sc_peak_b,
            "l1_peak_b":  l1_peak_b,
        }
    
    # --------------------------- Bootstrap regression --------------------------- #

    @staticmethod
    def _bootstrap_regression(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, ci: float = 95.0):
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            return None

        model     = LinearRegression().fit(x.reshape(-1, 1), y)
        slope     = model.coef_[0]
        intercept = model.intercept_
        r2        = model.score(x.reshape(-1, 1), y)
        _, _, _, p_value, std_err = linregress(x, y)

        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = model.predict(x_fit.reshape(-1, 1))

        rng        = np.random.default_rng(42)
        boot_lines = np.empty((n_boot, len(x_fit)))
        for i in range(n_boot):
            idx = rng.integers(0, len(x), len(x))
            m   = LinearRegression().fit(x[idx].reshape(-1, 1), y[idx])
            boot_lines[i] = m.predict(x_fit.reshape(-1, 1))

        lo = (100 - ci) / 2
        return {
            "x_fit":     x_fit,
            "y_fit":     y_fit,
            "ci_lower":  np.percentile(boot_lines, lo,       axis=0),
            "ci_upper":  np.percentile(boot_lines, 100 - lo, axis=0),
            "slope":     slope,
            "intercept": intercept,
            "r2":        r2,
            "p_value":   p_value,
            "std_err":   std_err,
            "n":         len(x),
        }

    @staticmethod
    def _save(fig, filename):
        save_dir = Path(SAVE_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")

    # ---------------------- FIGURE 1: Peak |B| parity plot ---------------------- #
    
    def _plot_peak_b_conservation(self):
        x = self.results["l1_peak_b"].values
        y = self.results["sc_peak_b"].values

        reg = self._bootstrap_regression(x, y)
        if reg is None:
            print(" [WARN] Insufficient data for peak |B| parity plot")
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(x, y, color="#1f77b4", alpha=0.7, s=60,
                   edgecolors="k", linewidths=0.5, zorder=3)

        lim_min = min(x[np.isfinite(x)].min(), y[np.isfinite(y)].min())
        lim_max = max(x[np.isfinite(x)].max(), y[np.isfinite(y)].max())
        margin  = 0.1 * (lim_max - lim_min)
        ax.plot([lim_min - margin, lim_max + margin],
                [lim_min - margin, lim_max + margin],
                'k--', lw=1.2, alpha=0.5, label='y = x', zorder=1)

        ax.plot(reg["x_fit"], reg["y_fit"], color="#1f77b4", linewidth=2, zorder=4)
        ax.fill_between(
            reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
            color="#1f77b4", alpha=0.2, zorder=2, label="95% CI"
        )

        t_stat    = (reg['slope'] - 1.0) / reg['std_err']
        p_slope_1 = 2 * (1 - t.cdf(abs(t_stat), reg['n'] - 2))

        sign   = "+" if reg["intercept"] >= 0 else ""
        eq_str = (
            f"y = {reg['slope']:+.3f}x {sign}{abs(reg['intercept']):.3f}\n"
            f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}\n"
            f"p(slope=1) = {p_slope_1:.3f}"
        )
        ax.text(0.05, 0.95, eq_str, transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="grey", alpha=0.9))

        ax.set_xlabel("L1 peak |B| [nT]", fontsize=12)
        ax.set_ylabel("DRO peak |B| [nT]", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")

        plt.tight_layout()
        if SAVE_PLOT_PDF:
            self._save(fig, "bmag_parity_peak_b.pdf")
        plt.show()
        plt.close(fig)

    # ----------- FIGURE 2: CME Pearson vs distance (no fit line / CI) ----------- #
    
    def _plot_cme_pearson(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        DISTANCE_KEYS   = ["upstream", "perp", "total"]
        DISTANCE_LABELS = [
                "Upstream Distance (AU)",
                "Perpendicular Distance (AU)",
                "Total Distance (AU)",
            ]



        for ax, dist_key, dist_label in zip(axes, DISTANCE_KEYS, DISTANCE_LABELS):
            x = self.results[dist_key].values
            y = self.results["cme_r"].values

            ax.scatter(x, y, color='orange', alpha=0.7,
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_xlabel(dist_label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_ylabel("Pearson Correlation r", fontsize=11)
        plt.tight_layout()
        if SAVE_PLOT_PDF:
            self._save(fig, "bmag_cme_pearson_vs_distance.pdf")
        plt.show()
        plt.close(fig)

    # ---------- FIGURE 3: CME RMSE vs distance (fit line + CI retained) --------- #
    
    def _plot_cme_rmse(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        CME_COLOUR = "#d62728"
        DISTANCE_KEYS   = ["upstream", "perp", "total"]
        DISTANCE_LABELS = [
            "Upstream Distance (AU)",
            "Perpendicular Distance (AU)",
            "Total Distance (AU)",
        ]

        for ax, dist_key, dist_label in zip(axes, DISTANCE_KEYS, DISTANCE_LABELS):
            x = self.results[dist_key].values
            y = self.results["cme_rmse"].values

            ax.scatter(x, y, color=CME_COLOUR, alpha=0.7,
                       edgecolors="k", linewidths=0.4, zorder=3)

            reg = self._bootstrap_regression(x, y)
            if reg is not None:
                ax.plot(reg["x_fit"], reg["y_fit"], color=CME_COLOUR,
                        linewidth=1.5, zorder=4)
                ax.fill_between(
                    reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                    color=CME_COLOUR, alpha=0.2, zorder=2, label="95% CI"
                )
                sign  = "+" if reg["intercept"] >= 0 else ""
                eq_str = (
                    f"y = {reg['slope']:+.3f}x {sign}{abs(reg['intercept']):.3f}\n"
                    f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}\n"
                )
                ax.text(0.05, 0.95, eq_str, transform=ax.transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="grey", alpha=0.8))

            ax.set_xlabel(dist_label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_ylabel("RMSE [nT]", fontsize=11)
        plt.tight_layout()
        if SAVE_PLOT_PDF:
            self._save(fig, "bmag_cme_rmse_vs_distance.pdf")
        plt.show()
        plt.close(fig)

    # ------------------------------- Run pipeline ------------------------------- #

    def run(self):
        for f in self.files:
            try:
                result = self._process_event(f)
                if result is not None:
                    self.results.append(result)
            except Exception as exc:
                print(f" [WARN] Error processing {Path(f).name}: {exc}")

        self.results = pd.DataFrame(self.results)
        print(f"\nProcessed {len(self.results)} events successfully.")

    # -------------------------------- Entry point ------------------------------- #

    def analyze_all(self):
        if self.results is None or len(self.results) == 0:
            print(" [WARN] No valid results to analyze.")
            return

        self._plot_peak_b_conservation()
        self._plot_cme_pearson()
        self._plot_cme_rmse()

if __name__ == "__main__":
    analysis = AbsoluteBCorrelationAnalysis(DATA_PATH)
    analysis.run()
    analysis.analyze_all()