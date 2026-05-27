from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks


INPUT_FILE = Path("../outputs/ssa/ssa_reconstructed_series.csv.gz")
OUT_DIR = Path("../outputs/ssa_component_spectral_analysis")

COMPONENT_COLUMNS = [
    "original",
    "trend",
    "interannual",
    "annual",
    "semiannual",
    "seasonal_subannual",
    "subseasonal",
    "high_frequency",
    "noise",
    "reconstructed_signal",
    "reconstruction_residual",
]

PERIOD_BANDS = {
    "high_frequency_2_7_days": (2, 7),
    "subseasonal_7_30_days": (7, 30),
    "seasonal_30_150_days": (30, 150),
    "semiannual_150_300_days": (150, 300),
    "annual_300_430_days": (300, 430),
    "interannual_1_3_years": (365, 365 * 3),
    "low_frequency_gt_3_years": (365 * 3, np.inf),
}


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_spectrum(x: np.ndarray, nperseg: int = 2048) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()

    if np.nanstd(x) == 0:
        return pd.DataFrame()

    x = x - np.nanmean(x)
    x = x / np.nanstd(x)

    nperseg = min(nperseg, len(x))

    freqs, power = welch(
        x,
        fs=1.0,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    spectrum = pd.DataFrame({
        "frequency_cycles_per_day": freqs,
        "power": power,
    })

    spectrum = spectrum[spectrum["frequency_cycles_per_day"] > 0].copy()
    spectrum["period_days"] = 1 / spectrum["frequency_cycles_per_day"]

    return spectrum


def dominant_periods(spectrum: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if spectrum.empty or len(spectrum) < 3:
        return pd.DataFrame()

    power = spectrum["power"].to_numpy()

    peaks, _ = find_peaks(
        power,
        prominence=np.nanmax(power) * 0.02,
    )

    if len(peaks) == 0:
        return pd.DataFrame()

    out = spectrum.iloc[peaks].copy()
    out = out.sort_values("power", ascending=False).head(top_n)
    out["rank"] = range(1, len(out) + 1)

    return out[[
        "rank",
        "period_days",
        "frequency_cycles_per_day",
        "power",
    ]]


def band_power_summary(spectrum: pd.DataFrame) -> pd.DataFrame:
    if spectrum.empty:
        return pd.DataFrame()

    total_power = spectrum["power"].sum()
    rows = []

    for band, (min_period, max_period) in PERIOD_BANDS.items():
        mask = (
            (spectrum["period_days"] >= min_period)
            & (spectrum["period_days"] < max_period)
        )

        band_power = spectrum.loc[mask, "power"].sum()

        rows.append({
            "band": band,
            "min_period_days": min_period,
            "max_period_days": max_period,
            "band_power": band_power,
            "total_power": total_power,
            "band_power_fraction": band_power / total_power if total_power > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def main() -> None:
    ensure_out_dir(OUT_DIR)

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    component_cols = [c for c in COMPONENT_COLUMNS if c in df.columns]

    all_dominant = []
    all_band_power = []

    grouped = df.groupby(["prefecture_code", "variable"])

    for (prefecture_code, variable), g in grouped:
        g = g.sort_values("date")

        print(f"Processing {prefecture_code} - {variable}")

        for component in component_cols:
            x = g[component].to_numpy(dtype=float)

            spectrum = compute_spectrum(x)

            dom = dominant_periods(spectrum)
            if not dom.empty:
                dom["prefecture_code"] = prefecture_code
                dom["variable"] = variable
                dom["component"] = component
                all_dominant.append(dom)

            bands = band_power_summary(spectrum)
            if not bands.empty:
                bands["prefecture_code"] = prefecture_code
                bands["variable"] = variable
                bands["component"] = component
                all_band_power.append(bands)

    dominant_df = pd.concat(all_dominant, ignore_index=True)
    band_power_df = pd.concat(all_band_power, ignore_index=True)

    dominant_file = OUT_DIR / "ssa_component_dominant_periods.csv"
    band_file = OUT_DIR / "ssa_component_band_power.csv"

    dominant_df.to_csv(dominant_file, index=False)
    band_power_df.to_csv(band_file, index=False)

    print(f"Saved: {dominant_file}")
    print(f"Saved: {band_file}")


if __name__ == "__main__":
    main()