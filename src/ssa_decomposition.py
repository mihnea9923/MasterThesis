from pathlib import Path

import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis


INPUT_FILE = Path("../outputs/intermediate_series/model_ready_analysis_dataset.csv")
OUT_DIR = Path("../outputs/ssa")

WINDOW_SIZE = 365
SIGNAL_VARIANCE_CUTOFF = 0.95

VARIABLES = [
    "cases",
    "incidence_per_100k",
    "t2m",
    "t2m_std_anom",
    "c2w_event",
    "w2c_event",
    "c2w_transition_intensity",
    "w2c_transition_intensity",
    "ah",
    "rh",
    "tp",
    "nw_wind",
]


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_daily_series(df: pd.DataFrame, variable: str) -> pd.Series:
    s = df[["date", variable]].copy()
    s["date"] = pd.to_datetime(s["date"])

    s = s.groupby("date")[variable].mean().sort_index()

    full_index = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_index)

    zero_fill_variables = {
        "cases",
        "c2w_event",
        "w2c_event",
        "c2w_transition_intensity",
        "w2c_transition_intensity",
    }

    if variable in zero_fill_variables:
        s = s.fillna(0)
    else:
        s = s.interpolate(limit_direction="both")

    return s


def dominant_period_days(component: np.ndarray) -> tuple[float, float]:
    x = np.asarray(component, dtype=float)
    x = x - np.nanmean(x)

    if not np.isfinite(x).all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()

    if np.nanstd(x) == 0:
        return np.inf, 0.0

    freqs = np.fft.rfftfreq(len(x), d=1.0)
    power = np.abs(np.fft.rfft(x)) ** 2

    if len(freqs) <= 1:
        return np.inf, 0.0

    freqs = freqs[1:]
    power = power[1:]

    idx = int(np.argmax(power))
    dominant_freq = freqs[idx]

    if dominant_freq <= 0:
        return np.inf, float(power[idx])

    return float(1 / dominant_freq), float(power[idx])


def classify_period(period_days: float) -> str:
    if not np.isfinite(period_days):
        return "trend"

    if period_days >= 365 * 3:
        return "trend"
    if 365 * 1.2 <= period_days < 365 * 3:
        return "interannual"
    if 300 <= period_days < 430:
        return "annual"
    if 150 <= period_days < 300:
        return "semiannual"
    if 30 <= period_days < 150:
        return "seasonal_subannual"
    if 7 <= period_days < 30:
        return "subseasonal"
    if 2 <= period_days < 7:
        return "high_frequency"

    return "noise"


def ssa_decompose_series(series: pd.Series, window_size: int = WINDOW_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = series.to_numpy(dtype=float)

    if not np.isfinite(x).all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()

    if np.nanstd(x) == 0:
        reconstructed = pd.DataFrame({
            "date": series.index,
            "original": x,
            "trend": 0.0,
            "interannual": 0.0,
            "annual": 0.0,
            "semiannual": 0.0,
            "seasonal_subannual": 0.0,
            "subseasonal": 0.0,
            "high_frequency": 0.0,
            "noise": 0.0,
        })
        metadata = pd.DataFrame()
        return reconstructed, metadata

    X = x.reshape(1, -1)

    this_window_size = min(window_size, len(x) // 2)

    ssa = SingularSpectrumAnalysis(
        window_size=this_window_size,
        groups=None
    )

    components = ssa.fit_transform(X)[0]

    component_variance = np.var(components, axis=1)
    total_component_variance = component_variance.sum()

    if total_component_variance == 0:
        variance_fraction = np.zeros_like(component_variance)
    else:
        variance_fraction = component_variance / total_component_variance

    cumulative_variance = np.cumsum(variance_fraction)

    component_rows = []

    reconstructed_parts = {
        "trend": np.zeros_like(x),
        "interannual": np.zeros_like(x),
        "annual": np.zeros_like(x),
        "semiannual": np.zeros_like(x),
        "seasonal_subannual": np.zeros_like(x),
        "subseasonal": np.zeros_like(x),
        "high_frequency": np.zeros_like(x),
        "noise": np.zeros_like(x),
    }

    for component_id, component in enumerate(components):
        period, spectral_power = dominant_period_days(component)

        band = classify_period(period)

        if variance_fraction[component_id] < 1e-5:
            band = "noise"

        reconstructed_parts[band] += component

        component_rows.append({
            "component_id": component_id,
            "dominant_period_days": period,
            "dominant_spectral_power": spectral_power,
            "variance_fraction": variance_fraction[component_id],
            "cumulative_variance_fraction": cumulative_variance[component_id],
            "assigned_band": band,
        })

    reconstructed = pd.DataFrame({
        "date": series.index,
        "original": x,
        **reconstructed_parts,
    })

    reconstructed["reconstructed_signal"] = (
        reconstructed["trend"]
        + reconstructed["interannual"]
        + reconstructed["annual"]
        + reconstructed["semiannual"]
        + reconstructed["seasonal_subannual"]
        + reconstructed["subseasonal"]
        + reconstructed["high_frequency"]
    )

    reconstructed["reconstruction_residual"] = (
        reconstructed["original"] - reconstructed["reconstructed_signal"]
    )

    metadata = pd.DataFrame(component_rows)

    return reconstructed, metadata


def main() -> None:
    ensure_out_dir(OUT_DIR)

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    variables = [v for v in VARIABLES if v in df.columns]

    all_reconstructed = []
    all_metadata = []

    for prefecture_code, pref_df in df.groupby("prefecture_code"):
        pref_df = pref_df.sort_values("date")
        print(f"Processing prefecture {prefecture_code}")

        for variable in variables:
            print(f"  SSA: {variable}")

            series = prepare_daily_series(pref_df, variable)

            reconstructed, metadata = ssa_decompose_series(
                series=series,
                window_size=WINDOW_SIZE,
            )

            reconstructed["prefecture_code"] = prefecture_code
            reconstructed["variable"] = variable

            if not metadata.empty:
                metadata["prefecture_code"] = prefecture_code
                metadata["variable"] = variable
                all_metadata.append(metadata)

            all_reconstructed.append(reconstructed)

    reconstructed_df = pd.concat(all_reconstructed, ignore_index=True)
    metadata_df = pd.concat(all_metadata, ignore_index=True)

    reconstructed_file = OUT_DIR / "ssa_reconstructed_series.csv.gz"
    metadata_file = OUT_DIR / "ssa_component_metadata.csv"

    reconstructed_df.to_csv(reconstructed_file, index=False, compression="gzip")
    metadata_df.to_csv(metadata_file, index=False)

    print(f"Saved: {reconstructed_file}")
    print(f"Saved: {metadata_file}")

    band_by_prefecture = (
    metadata_df
    .groupby(["prefecture_code", "variable", "assigned_band"], as_index=False)
    .agg(
        band_variance_fraction=("variance_fraction", "sum"),
        n_components=("component_id", "count"),
    )
)

    band_by_prefecture_file = OUT_DIR / "ssa_band_by_prefecture.csv"
    band_by_prefecture.to_csv(band_by_prefecture_file, index=False)

    summary = (
        band_by_prefecture
        .groupby(["variable", "assigned_band"], as_index=False)
        .agg(
            mean_band_variance_fraction=("band_variance_fraction", "mean"),
            median_band_variance_fraction=("band_variance_fraction", "median"),
            min_band_variance_fraction=("band_variance_fraction", "min"),
            max_band_variance_fraction=("band_variance_fraction", "max"),
            mean_n_components=("n_components", "mean"),
        )
    )

    summary_file = OUT_DIR / "ssa_band_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()