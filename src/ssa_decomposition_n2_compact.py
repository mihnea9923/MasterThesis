from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, periodogram
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "intermediate_series" / "model_ready_analysis_dataset.csv"
OUT_DIR = ROOT / "outputs" / "ssa_n2"

WINDOW_FRACTION = 0.5
N_COMPONENTS = 400
RANDOM_STATE = 42

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

BANDS = [
    "trend",
    "interannual",
    "annual",
    "semiannual",
    "seasonal_subannual",
    "subseasonal",
    "high_frequency",
    "unresolved_retained",
]


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_daily_series(df: pd.DataFrame, variable: str) -> pd.Series:
    series = df[["date", variable]].copy()
    series["date"] = pd.to_datetime(series["date"])

    series = series.groupby("date")[variable].mean().sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index)

    zero_fill_variables = {
        "cases",
        "c2w_event",
        "w2c_event",
        "c2w_transition_intensity",
        "w2c_transition_intensity",
    }

    if variable in zero_fill_variables:
        series = series.fillna(0.0)
    else:
        series = series.interpolate(limit_direction="both")

    return series.astype(float)


def build_trajectory_matrix(values: np.ndarray, window_length: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=window_length,
        ).T
    )


def diagonal_average_components(
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
    right_vectors_transposed: np.ndarray,
    window_length: int,
    number_of_windows: int,
) -> np.ndarray:
    weights = fftconvolve(
        np.ones(window_length),
        np.ones(number_of_windows),
        mode="full",
    )

    components = np.empty(
        (len(singular_values), len(weights)),
        dtype=float,
    )

    for index in range(len(singular_values)):
        anti_diagonal_sum = fftconvolve(
            left_vectors[:, index],
            right_vectors_transposed[index],
            mode="full",
        )
        components[index] = (
            singular_values[index]
            * anti_diagonal_sum
            / weights
        )

    return components


def dominant_period_days(component: np.ndarray) -> tuple[float, float]:
    values = np.asarray(component, dtype=float)

    if not np.isfinite(values).all():
        values = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .to_numpy()
        )

    if np.std(values) < 1e-12:
        return np.inf, 0.0

    frequencies, power = periodogram(
        values,
        fs=1.0,
        window="hann",
        detrend="constant",
        scaling="spectrum",
    )

    positive = frequencies > 0
    frequencies = frequencies[positive]
    power = power[positive]

    if len(frequencies) == 0:
        return np.inf, 0.0

    peak_index = int(np.argmax(power))
    frequency = float(frequencies[peak_index])
    peak_power = float(power[peak_index])

    if frequency <= 0:
        return np.inf, peak_power

    return float(1.0 / frequency), peak_power


def classify_period(period_days: float, n_observations: int) -> str:
    if not np.isfinite(period_days):
        return "trend"

    if period_days >= 0.75 * n_observations:
        return "trend"

    if period_days > 430:
        return "interannual"
    if 300 <= period_days <= 430:
        return "annual"
    if 150 <= period_days < 300:
        return "semiannual"
    if 30 <= period_days < 150:
        return "seasonal_subannual"
    if 7 <= period_days < 30:
        return "subseasonal"
    if 2 <= period_days < 7:
        return "high_frequency"

    return "unresolved_retained"


def constant_result(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = series.to_numpy(dtype=float)

    reconstructed_parts = {
        band: np.zeros_like(values)
        for band in BANDS
    }
    reconstructed_parts["trend"] = values.copy()

    reconstructed = pd.DataFrame(
        {
            "date": series.index,
            "original": values,
            **reconstructed_parts,
            "retained_reconstruction": values,
            "truncated_svd_residual": np.zeros_like(values),
            "reconstructed_signal": values,
            "reconstruction_residual": np.zeros_like(values),
        }
    )

    return reconstructed, pd.DataFrame()


def ssa_decompose_series(
    series: pd.Series,
    n_components: int = N_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = series.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        values = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .to_numpy()
        )

    if np.std(values) < 1e-12:
        return constant_result(series)

    n_observations = len(values)
    window_length = int(n_observations * WINDOW_FRACTION)
    number_of_windows = n_observations - window_length + 1

    trajectory_matrix = build_trajectory_matrix(
        values=values,
        window_length=window_length,
    )

    actual_components = min(
        n_components,
        min(trajectory_matrix.shape) - 1,
    )

    left_vectors, singular_values, right_vectors_transposed = randomized_svd(
        trajectory_matrix,
        n_components=actual_components,
        n_iter=5,
        random_state=RANDOM_STATE,
    )

    components = diagonal_average_components(
        left_vectors=left_vectors,
        singular_values=singular_values,
        right_vectors_transposed=right_vectors_transposed,
        window_length=window_length,
        number_of_windows=number_of_windows,
    )

    total_trajectory_energy = float(
        np.einsum("ij,ij->", trajectory_matrix, trajectory_matrix)
    )
    energy_fraction = singular_values**2 / total_trajectory_energy
    cumulative_energy_fraction = np.cumsum(energy_fraction)

    reconstructed_parts = {
        band: np.zeros_like(values)
        for band in BANDS
    }

    metadata_rows: list[dict] = []

    for index, component in enumerate(components):
        period_days, spectral_power = dominant_period_days(component)
        band = classify_period(period_days, n_observations)

        reconstructed_parts[band] += component

        metadata_rows.append(
            {
                "component_id": index + 1,
                "singular_value": singular_values[index],
                "dominant_period_days": period_days,
                "dominant_spectral_power": spectral_power,
                "trajectory_energy_fraction": energy_fraction[index],
                "cumulative_trajectory_energy_fraction":
                    cumulative_energy_fraction[index],
                "assigned_band": band,
                "window_length": window_length,
            }
        )

    retained_reconstruction = components.sum(axis=0)
    truncated_svd_residual = values - retained_reconstruction

    reconstructed_signal = sum(
        reconstructed_parts[band]
        for band in BANDS
        if band != "unresolved_retained"
    )

    reconstruction_residual = values - reconstructed_signal

    reconstructed = pd.DataFrame(
        {
            "date": series.index,
            "original": values,
            **reconstructed_parts,
            "retained_reconstruction": retained_reconstruction,
            "truncated_svd_residual": truncated_svd_residual,
            "reconstructed_signal": reconstructed_signal,
            "reconstruction_residual": reconstruction_residual,
        }
    )

    metadata = pd.DataFrame(metadata_rows)
    return reconstructed, metadata


def main() -> None:
    ensure_out_dir(OUT_DIR)

    dataframe = pd.read_csv(INPUT_FILE)
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    variables = [
        variable
        for variable in VARIABLES
        if variable in dataframe.columns
    ]

    reconstructed_file = OUT_DIR / "ssa_reconstructed_series.csv.gz"
    metadata_file = OUT_DIR / "ssa_component_metadata.csv.gz"

    metadata_parts: list[pd.DataFrame] = []
    reconstructed_header_written = False
    metadata_header_written = False

    with gzip.open(
        reconstructed_file,
        "wt",
        newline="",
    ) as reconstructed_handle, gzip.open(
        metadata_file,
        "wt",
        newline="",
    ) as metadata_handle:

        for prefecture_code, prefecture_df in dataframe.groupby(
            "prefecture_code"
        ):
            prefecture_df = prefecture_df.sort_values("date")
            print(f"Processing prefecture {prefecture_code}")

            for variable in variables:
                print(f"  SSA N/2: {variable}")

                series = prepare_daily_series(
                    prefecture_df,
                    variable,
                )

                reconstructed, metadata = ssa_decompose_series(series)

                reconstructed["prefecture_code"] = prefecture_code
                reconstructed["variable"] = variable

                reconstructed.to_csv(
                    reconstructed_handle,
                    index=False,
                    header=not reconstructed_header_written,
                )
                reconstructed_header_written = True

                if not metadata.empty:
                    metadata["prefecture_code"] = prefecture_code
                    metadata["variable"] = variable

                    metadata.to_csv(
                        metadata_handle,
                        index=False,
                        header=not metadata_header_written,
                    )
                    metadata_header_written = True
                    metadata_parts.append(metadata)

    metadata_df = pd.concat(metadata_parts, ignore_index=True)

    band_by_prefecture = (
        metadata_df
        .groupby(
            ["prefecture_code", "variable", "assigned_band"],
            as_index=False,
        )
        .agg(
            band_trajectory_energy_fraction=(
                "trajectory_energy_fraction",
                "sum",
            ),
            n_components=("component_id", "count"),
        )
    )

    band_by_prefecture_file = OUT_DIR / "ssa_band_by_prefecture.csv"
    band_by_prefecture.to_csv(band_by_prefecture_file, index=False)

    summary = (
        band_by_prefecture
        .groupby(["variable", "assigned_band"], as_index=False)
        .agg(
            mean_band_trajectory_energy_fraction=(
                "band_trajectory_energy_fraction",
                "mean",
            ),
            median_band_trajectory_energy_fraction=(
                "band_trajectory_energy_fraction",
                "median",
            ),
            min_band_trajectory_energy_fraction=(
                "band_trajectory_energy_fraction",
                "min",
            ),
            max_band_trajectory_energy_fraction=(
                "band_trajectory_energy_fraction",
                "max",
            ),
            mean_n_components=("n_components", "mean"),
        )
    )

    summary_file = OUT_DIR / "ssa_band_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("\nSaved:")
    print(f"  {reconstructed_file}")
    print(f"  {metadata_file}")
    print(f"  {band_by_prefecture_file}")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()
