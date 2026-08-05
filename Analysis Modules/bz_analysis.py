"""
bz_analysis.py
==============

Analysis module for comparing B_z behaviour between L1 measurements and
spacecraft measurements during the same identified CME period.

This script loads event-level CSV files produced by primary.py, transforms
spacecraft magnetic-field measurements from RTN to GSE coordinates, computes
correlation and error metrics for B_z during CME windows, and generates
summary figures showing minimum-B_z parity, minimum-B_z difference versus
distance, Pearson correlation versus distance, and RMSE versus distance.

Configuration
-------------
Set the file paths in the configuration block at the top of this module before
running:
- DATA_PATH should point to the folder containing event_*.csv files generated
  by primary.py.
- SAVE_FOLDER should point to the folder where output figures will be saved.
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
SAVE_PLOTS_PDF      = False

DISTANCE_KEYS   = ["upstream", "perp", "total"]
DISTANCE_LABELS = ["Upstream Distance (AU)", "Perpendicular Distance (AU)", "Total Distance (AU)",]
REQUIRED_COLUMNS = [
    "xcorr_valid",
    "timestamp",
    "cme_start_utc",
    "cme_end_utc",
    "B_r",
    "B_t",
    "B_n",
    "sc_x_hci",
    "sc_y_hci",
    "sc_z_hci",
    "earth_x_hci",
    "earth_y_hci",
    "earth_z_hci",
    "sc_distance_to_earth_au",
    "sc_angle_from_sun_earth_line_deg",
    "l1_B_z_gse_xcorr",
]


# ------------------------- Coordinate transformation ------------------------ #

def rtn_to_gse(B_r, B_t, B_n, sc_x_hci, sc_y_hci, sc_z_hci,
               earth_x_hci, earth_y_hci, earth_z_hci):
    B_r = np.asarray(B_r); B_t = np.asarray(B_t); B_n = np.asarray(B_n)
    sc_x_hci = np.asarray(sc_x_hci); sc_y_hci = np.asarray(sc_y_hci); sc_z_hci = np.asarray(sc_z_hci)
    earth_x_hci = np.asarray(earth_x_hci); earth_y_hci = np.asarray(earth_y_hci); earth_z_hci = np.asarray(earth_z_hci)

    r_sc   = np.column_stack([sc_x_hci, sc_y_hci, sc_z_hci])
    r_norm = np.linalg.norm(r_sc, axis=1, keepdims=True)
    r_hat  = r_sc / np.where(r_norm > 0, r_norm, 1.0)

    ecl_tang = np.column_stack([-sc_y_hci, sc_x_hci, np.zeros_like(sc_z_hci)])
    t_norm   = np.linalg.norm(ecl_tang, axis=1, keepdims=True)
    t_hat    = ecl_tang / np.where(t_norm > 0, t_norm, 1.0)

    n_hat  = np.cross(r_hat, t_hat)
    n_norm = np.linalg.norm(n_hat, axis=1, keepdims=True)
    n_hat  = n_hat / np.where(n_norm > 0, n_norm, 1.0)

    B_hci = (r_hat * B_r[:, np.newaxis] +
             t_hat * B_t[:, np.newaxis] +
             n_hat * B_n[:, np.newaxis])

    earth_pos  = np.column_stack([earth_x_hci, earth_y_hci, earth_z_hci])
    earth_norm = np.linalg.norm(earth_pos, axis=1, keepdims=True)

    x_gse_hat = -earth_pos / np.where(earth_norm > 0, earth_norm, 1.0)
    z_gse_hat = np.tile([0.0, 0.0, 1.0], (len(B_r), 1))
    y_gse_hat = np.cross(z_gse_hat, x_gse_hat)
    y_norm    = np.linalg.norm(y_gse_hat, axis=1, keepdims=True)
    y_gse_hat = y_gse_hat / np.where(y_norm > 0, y_norm, 1.0)
    z_gse_hat = np.cross(x_gse_hat, y_gse_hat)

    B_x_gse = np.sum(x_gse_hat * B_hci, axis=1)
    B_y_gse = np.sum(y_gse_hat * B_hci, axis=1)
    B_z_gse = np.sum(z_gse_hat * B_hci, axis=1)

    invalid = (~np.isfinite(B_r) | ~np.isfinite(B_t) | ~np.isfinite(B_n) |
               ~np.isfinite(sc_x_hci) | ~np.isfinite(sc_y_hci) | ~np.isfinite(sc_z_hci) |
               ~np.isfinite(earth_x_hci) | ~np.isfinite(earth_y_hci) | ~np.isfinite(earth_z_hci) |
               (r_norm.flatten() == 0) | (earth_norm.flatten() == 0))
    B_x_gse[invalid] = np.nan
    B_y_gse[invalid] = np.nan
    B_z_gse[invalid] = np.nan

    return B_x_gse, B_y_gse, B_z_gse

# --------------------------------- Analysis --------------------------------- #

class BzCorrelationAnalysis:

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.files = list(Path(data_directory).glob("event_*.csv"))
        self.results = []

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

    @staticmethod
    def _pearson(x, y, min_n=20):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            return np.nan
        return pearsonr(x[mask], y[mask])[0]

    @staticmethod
    def _rmse(x, y, min_n=20):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            return np.nan
        return float(np.sqrt(np.mean((x[mask] - y[mask])**2)))

    @staticmethod
    def _percentile(x, p, min_n=20):
        mask = np.isfinite(x)
        if mask.sum() < min_n:
            return np.nan
        return float(np.percentile(x[mask], p))

    @staticmethod
    def _geometry(df):
        total     = df["sc_distance_to_earth_au"].iloc[0]
        angle_rad = np.deg2rad(df["sc_angle_from_sun_earth_line_deg"].iloc[0])
        return float(total * np.cos(angle_rad)), float(total * np.sin(angle_rad)), float(total)

    @staticmethod
    def _save(fig, filename):
        save_dir = Path(SAVE_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")

    @staticmethod
    def _validate_columns(df: pd.DataFrame, filepath) -> bool:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            print(
                f"[WARN] Skipping {Path(filepath).name}: missing required columns: "
                + ", ".join(missing)
            )
            return False
        return True

    # ----------------------------- Event processing ----------------------------- #

    def _process_event(self, filepath):
        df = pd.read_csv(filepath)

        if not self._validate_columns(df, filepath):
            return None

        df = df[df["xcorr_valid"].astype(float) > 0].copy()
        if len(df) < 20:
            return None

        B_x_gse, B_y_gse, B_z_gse = rtn_to_gse(
            df["B_r"].values, df["B_t"].values, df["B_n"].values,
            df["sc_x_hci"].values, df["sc_y_hci"].values, df["sc_z_hci"].values,
            df["earth_x_hci"].values, df["earth_y_hci"].values, df["earth_z_hci"].values,
        )
        df["Bz_gse"] = B_z_gse

        df["timestamp"] = self._strip_tz(pd.to_datetime(df["timestamp"], errors="coerce"))
        cme_start = self._naive_ts(df["cme_start_utc"].iloc[0])
        cme_end   = self._naive_ts(df["cme_end_utc"].iloc[0])
        cme = df[(df["timestamp"] >= cme_start) & (df["timestamp"] <= cme_end)]

        upstream, perp, total = self._geometry(df)

        return {
            "upstream":   upstream,
            "perp":       perp,
            "total":      total,
            "cme_r":      self._pearson(cme["Bz_gse"].values,  cme["l1_B_z_gse_xcorr"].values),
            "cme_rmse":   self._rmse(cme["Bz_gse"].values,     cme["l1_B_z_gse_xcorr"].values),
            "sc_min_bz":  self._percentile(cme["Bz_gse"].values,           5),
            "l1_min_bz":  self._percentile(cme["l1_B_z_gse_xcorr"].values, 5),
        }


    def run(self):
        for f in self.files:
            try:
                result = self._process_event(f)
                if result is not None:
                    self.results.append(result)
            except Exception as exc:
                print(f"[WARN] Skipping {Path(f).name}: {exc}")
        self.results = pd.DataFrame(self.results)
        print(f"\n[INFO] Processed {len(self.results)} events successfully.")

    @staticmethod
    def _bootstrap_regression(x, y, n_boot=1000, ci=95.0):
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
            "x_fit": x_fit, "y_fit": y_fit,
            "ci_lower":  np.percentile(boot_lines, lo,       axis=0),
            "ci_upper":  np.percentile(boot_lines, 100 - lo, axis=0),
            "slope": slope, "intercept": intercept,
            "r2": r2, "p_value": p_value, "std_err": std_err, "n": len(x),
        }

    # ------------------------ Fig. 1 - Min Bz parity plot ----------------------- #

    def _plot_min_bz_conservation(self):
        x = self.results["l1_min_bz"].values
        y = self.results["sc_min_bz"].values

        reg = self._bootstrap_regression(x, y)
        if reg is None:
            print("[WARN] Insufficient data for the minimum-Bz parity plot.")
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(x, y, color='red', alpha=0.7, s=60,
                   edgecolors="k", linewidths=0.5, zorder=3)

        lim_min = min(x[np.isfinite(x)].min(), y[np.isfinite(y)].min())
        lim_max = max(x[np.isfinite(x)].max(), y[np.isfinite(y)].max())
        margin  = 0.1 * (lim_max - lim_min)
        ax.plot([lim_min - margin, lim_max + margin],
                [lim_min - margin, lim_max + margin],
                "k--", lw=1.2, alpha=0.5, label="y = x", zorder=1)

        ax.plot(reg["x_fit"], reg["y_fit"], color='red', linewidth=2, zorder=4)
        ax.fill_between(reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                        color='red', alpha=0.2, zorder=2, label="95% CI")

        t_stat    = (reg["slope"] - 1.0) / reg["std_err"]
        p_slope_1 = 2 * (1 - t.cdf(abs(t_stat), reg["n"] - 2))

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

        ax.set_xlabel("L1 5th Percentile Bz [nT]", fontsize=12)
        ax.set_ylabel("DRO 5th Percentile Bz [nT]", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")

        plt.tight_layout()
        if SAVE_PLOTS_PDF:
            self._save(fig, "bz_parity_min_bz.pdf")
        plt.show()
        plt.close(fig)

    # ------------------ Fig. 2 - Min Bz difference vs distance ------------------ #

    def _plot_min_bz_distance(self):
        diff = self.results["sc_min_bz"].values - self.results["l1_min_bz"].values

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

        for ax, dist_key, dist_label in zip(axes, DISTANCE_KEYS, DISTANCE_LABELS):
            x = self.results[dist_key].values

            ax.scatter(x, diff, color="#ff7f0e", alpha=0.7, s=50,
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.axhline(0, color="grey", linewidth=1.0, linestyle="--", zorder=1)

            reg = self._bootstrap_regression(x, diff)
            if reg is not None:
                ax.plot(reg["x_fit"], reg["y_fit"], color="#ff7f0e",
                        linewidth=1.5, zorder=4)
                ax.fill_between(reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                                color="#ff7f0e", alpha=0.15, zorder=2)

                sign  = "+" if reg["intercept"] >= 0 else ""
                p_sig = ("***" if reg["p_value"] < 0.001
                          else "**" if reg["p_value"] < 0.01
                          else "*"  if reg["p_value"] < 0.05
                          else "n.s.")
                eq_str = (
                    f"y = {reg['slope']:+.3f}x {sign}{abs(reg['intercept']):.3f}\n"
                    f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}\n"
                    f"p = {reg['p_value']:.4f} {p_sig}"
                )
                ax.text(0.05, 0.95, eq_str, transform=ax.transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="grey", alpha=0.8))

            ax.set_xlabel(dist_label, fontsize=11)

            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_ylabel("Difference  SC − L1  [nT]", fontsize=11)
        plt.tight_layout()
        if SAVE_PLOTS_PDF:
            self._save(fig, "bz_min_bz_diff_vs_distance.pdf")
        plt.show()
        plt.close(fig)

    # --------------------- Fig. 3 - CME Pearson vs distance --------------------- #

    def _plot_cme_pearson(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

        for ax, dist_key, dist_label in zip(axes, DISTANCE_KEYS, DISTANCE_LABELS):
            x = self.results[dist_key].values
            y = self.results["cme_r"].values

            ax.scatter(x, y, color='red', alpha=0.7,
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_xlabel(dist_label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_ylabel("Pearson Correlation r", fontsize=11)
        plt.tight_layout()
        if SAVE_PLOTS_PDF:
            self._save(fig, "bz_cme_pearson_vs_distance.pdf")
        plt.show()
        plt.close(fig)

    # ----------------------- Fig. 4 - CME RMSE vs distance ---------------------- #

    def _plot_cme_rmse(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

        for ax, dist_key, dist_label in zip(axes, DISTANCE_KEYS, DISTANCE_LABELS):
            x = self.results[dist_key].values
            y = self.results["cme_rmse"].values

            ax.scatter(x, y, color='red', alpha=0.7,
                       edgecolors="k", linewidths=0.4, zorder=3)

            reg = self._bootstrap_regression(x, y)
            if reg is not None:
                ax.plot(reg["x_fit"], reg["y_fit"], color='red',
                        linewidth=1.5, zorder=4)
                ax.fill_between(reg["x_fit"], reg["ci_lower"], reg["ci_upper"],
                                color='red', alpha=0.2, zorder=2)

                sign  = "+" if reg["intercept"] >= 0 else ""
                p_sig = ("***" if reg["p_value"] < 0.001
                          else "**" if reg["p_value"] < 0.01
                          else "*"  if reg["p_value"] < 0.05
                          else "n.s.")
                eq_str = (
                    f"y = {reg['slope']:+.3f}x {sign}{abs(reg['intercept']):.3f}\n"
                    f"$R^2$ = {reg['r2']:.3f},  n = {reg['n']}\n"
                    f"p = {reg['p_value']:.4f} {p_sig}"
                )
                ax.text(0.05, 0.95, eq_str, transform=ax.transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="grey", alpha=0.8))

            ax.set_xlabel(dist_label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_ylabel("RMSE [nT]", fontsize=11)
        plt.tight_layout()
        if SAVE_PLOTS_PDF:
            self._save(fig, "bz_cme_rmse_vs_distance.pdf")
        plt.show()
        plt.close(fig)

    # -------------------------------- Entry point ------------------------------- #

    def analyze_all(self):
        if self.results is None or len(self.results) == 0:
            print("[WARN] No valid results to analyze.")
            return

        self._plot_min_bz_conservation()
        self._plot_min_bz_distance()
        self._plot_cme_pearson()
        self._plot_cme_rmse()


if __name__ == "__main__":
    analysis = BzCorrelationAnalysis(DATA_PATH)
    analysis.run()
    analysis.analyze_all()