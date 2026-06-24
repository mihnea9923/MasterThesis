from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, periodogram
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "intermediate_series"
    / "model_ready_analysis_dataset.csv"
)

OUT_DIR = ROOT / "outputs" / "ssa_n2_pilot_k150"

PILOT_SERIES = [
    ("JP-13", "cases"),
    ("JP-13", "t2m"),
]

MAX_COMPONENTS = 150

WCOR_COMPONENTS = 40

EIGENVECTOR_PAIRS = 12

RANDOM_STATE = 42


def prepare_daily_series(
    df: pd.DataFrame,
    variable: str,
) -> pd.Series:
    series = df[["date", variable]].copy()
    series["date"] = pd.to_datetime(series["date"])

    series = (
        series
        .groupby("date")[variable]
        .mean()
        .sort_index()
    )

    full_index = pd.date_range(
        series.index.min(),
        series.index.max(),
        freq="D",
    )

    series = series.reindex(full_index)

    zero_fill_variables = {
        "cases",
        "c2w_event",
        "w2c_event",
        "c2w_transition_intensity",
        "w2c_transition_intensity",
    }

    if variable in zero_fill_variables:
        series = series.fillna(0)
    else:
        series = series.interpolate(limit_direction="both")

    return series.astype(float)


def build_trajectory_matrix(
    x: np.ndarray,
    window_length: int,
) -> np.ndarray:
    matrix = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=window_length,
    ).T

    return np.ascontiguousarray(matrix)


def diagonal_averaging_from_svd(
    u: np.ndarray,
    singular_value: float,
    vt: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    anti_diagonal_sum = fftconvolve(u, vt, mode="full")

    return singular_value * anti_diagonal_sum / weights


def reconstruct_elementary_components(
    U: np.ndarray,
    singular_values: np.ndarray,
    Vt: np.ndarray,
    window_length: int,
    number_of_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = fftconvolve(
        np.ones(window_length),
        np.ones(number_of_windows),
        mode="full",
    )

    components = np.empty(
        (len(singular_values), len(weights)),
        dtype=float,
    )

    for component_index in range(len(singular_values)):
        components[component_index] = diagonal_averaging_from_svd(
            u=U[:, component_index],
            singular_value=singular_values[component_index],
            vt=Vt[component_index],
            weights=weights,
        )

    return components, weights

def save_elementary_components(
    series: pd.Series,
    components: np.ndarray,
    output_file: Path,
) -> None:
    if components.ndim != 2:
        raise ValueError(
            "components must be a 2D array with shape "
            "(number_of_components, number_of_observations)"
        )

    if components.shape[1] != len(series):
        raise ValueError(
            "The reconstructed component length does not match "
            "the original series length."
        )

    output_data = {
        "date": series.index,
        "original": series.to_numpy(dtype=float),
    }

    for component_index, component in enumerate(
        components,
        start=1,
    ):
        column_name = f"RC_{component_index:03d}"
        output_data[column_name] = component

    components_df = pd.DataFrame(output_data)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    components_df.to_csv(
        output_file,
        index=False,
        compression="gzip",
    )

    print(
        "Saved elementary reconstructed components: "
        f"{output_file}"
    )

def dominant_period_days(
    component: np.ndarray,
) -> tuple[float, float]:
    x = np.asarray(component, dtype=float)

    if not np.isfinite(x).all():
        x = (
            pd.Series(x)
            .interpolate(limit_direction="both")
            .to_numpy()
        )

    if np.std(x) < 1e-12:
        return np.inf, 0.0

    frequencies, power = periodogram(
        x,
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
    frequency = frequencies[peak_index]

    if frequency <= 0:
        return np.inf, float(power[peak_index])

    return (
        float(1.0 / frequency),
        float(power[peak_index]),
    )


def provisional_period_label(period_days: float) -> str:
    if not np.isfinite(period_days):
        return "trend_or_constant"

    if period_days >= 3 * 365:
        return "trend_or_low_frequency"

    if 1.2 * 365 <= period_days < 3 * 365:
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

    return "noise_or_ultra_high_frequency"


def calculate_w_correlation(
    components: np.ndarray,
    weights: np.ndarray,
    max_components: int,
) -> np.ndarray:
    number_to_use = min(max_components, components.shape[0])

    selected = components[:number_to_use]

    weighted_components = selected * np.sqrt(weights)[None, :]

    gram_matrix = weighted_components @ weighted_components.T
    norms = np.sqrt(np.maximum(np.diag(gram_matrix), 0))

    denominator = np.outer(norms, norms)

    w_correlation = np.divide(
        gram_matrix,
        denominator,
        out=np.zeros_like(gram_matrix),
        where=denominator > 0,
    )

    return np.clip(w_correlation, -1, 1)


def plot_component_norms(
    singular_values: np.ndarray,
    total_trajectory_energy: float,
    output_file: Path,
) -> None:
    component_numbers = np.arange(1, len(singular_values) + 1)

    energy_fraction = (
        singular_values ** 2
        / total_trajectory_energy
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        component_numbers,
        energy_fraction,
        marker="o",
        markersize=3,
        linewidth=1,
    )

    ax.set_yscale("log")
    ax.set_xlabel("SSA component")
    ax.set_ylabel("Fraction of trajectory-matrix energy")
    ax.set_title("SSA component norms / singular-value energy")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_eigenvector_pairs(
    U: np.ndarray,
    output_file: Path,
) -> None:
    number_of_pairs = min(
        EIGENVECTOR_PAIRS,
        U.shape[1] - 1,
    )

    number_of_columns = 4
    number_of_rows = int(
        np.ceil(number_of_pairs / number_of_columns)
    )

    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(16, 3.5 * number_of_rows),
    )

    axes = np.atleast_1d(axes).ravel()

    for pair_index in range(number_of_pairs):
        ax = axes[pair_index]

        first_component = pair_index
        second_component = pair_index + 1

        ax.plot(
            U[:, first_component],
            U[:, second_component],
            linewidth=0.9,
        )

        ax.set_xlabel(f"Eigenvector {first_component + 1}")
        ax.set_ylabel(f"Eigenvector {second_component + 1}")
        ax.set_title(
            f"{first_component + 1} vs "
            f"{second_component + 1}"
        )
        ax.grid(alpha=0.25)

    for ax in axes[number_of_pairs:]:
        ax.axis("off")

    fig.suptitle("Adjacent SSA eigenvector pairs", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_w_correlation(
    w_correlation: np.ndarray,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))

    image = ax.imshow(
        np.abs(w_correlation),
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    ax.set_xlabel("SSA component")
    ax.set_ylabel("SSA component")
    ax.set_title("Absolute W-correlation matrix")

    ticks = np.arange(0, len(w_correlation), 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)

    fig.colorbar(
        image,
        ax=ax,
        label="Absolute weighted correlation",
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_elementary_components(
    dates: pd.DatetimeIndex,
    components: np.ndarray,
    output_file: Path,
    number_to_plot: int = 12,
) -> None:
    number_to_plot = min(number_to_plot, components.shape[0])

    fig, axes = plt.subplots(
        number_to_plot,
        1,
        figsize=(15, 2.1 * number_to_plot),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    for index in range(number_to_plot):
        axes[index].plot(
            dates,
            components[index],
            linewidth=0.8,
        )

        axes[index].set_ylabel(f"RC {index + 1}")
        axes[index].grid(alpha=0.2)

    axes[-1].set_xlabel("Date")

    fig.suptitle(
        "First elementary reconstructed SSA components",
        fontsize=15,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_retained_reconstruction(
    dates: pd.DatetimeIndex,
    original: np.ndarray,
    reconstruction: np.ndarray,
    output_file: Path,
) -> None:
    residual = original - reconstruction

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
    )

    axes[0].plot(
        dates,
        original,
        label="Original",
        alpha=0.55,
        linewidth=0.8,
    )

    axes[0].plot(
        dates,
        reconstruction,
        label="Reconstruction from retained components",
        linewidth=1,
    )

    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        dates,
        residual,
        linewidth=0.8,
    )

    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)

    fig.suptitle("N/2 SSA retained reconstruction", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def run_pilot(
    series: pd.Series,
    prefecture_code: str,
    variable: str,
) -> dict:
    output_directory = (
        OUT_DIR
        / f"{prefecture_code}_{variable}"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    x = series.to_numpy(dtype=float)

    number_of_observations = len(x)
    window_length = number_of_observations // 2
    number_of_windows = (
        number_of_observations
        - window_length
        + 1
    )

    print()
    print(f"Processing {prefecture_code} - {variable}")
    print(f"N = {number_of_observations}")
    print(f"L = {window_length}")
    print(f"K = {number_of_windows}")
    print(f"L / N = {window_length / number_of_observations:.4f}")

    trajectory_matrix = build_trajectory_matrix(
        x=x,
        window_length=window_length,
    )

    number_of_components = min(
        MAX_COMPONENTS,
        min(trajectory_matrix.shape) - 1,
    )

    U, singular_values, Vt = randomized_svd(
        trajectory_matrix,
        n_components=number_of_components,
        n_iter=5,
        random_state=RANDOM_STATE,
    )

    total_trajectory_energy = float(
        np.einsum(
            "ij,ij->",
            trajectory_matrix,
            trajectory_matrix,
        )
    )

    elementary_components, weights = (
        reconstruct_elementary_components(
            U=U,
            singular_values=singular_values,
            Vt=Vt,
            window_length=window_length,
            number_of_windows=number_of_windows,
        )
    )
    
    save_elementary_components(
    series=series,
    components=elementary_components,
    output_file=(
        output_directory
        / "elementary_components.csv.gz"
    ),
)

    trajectory_energy_fraction = (
        singular_values ** 2
        / total_trajectory_energy
    )

    cumulative_energy_fraction = np.cumsum(
        trajectory_energy_fraction
    )

    period_rows = []

    for component_index, component in enumerate(
        elementary_components
    ):
        period_days, spectral_power = dominant_period_days(
            component
        )

        period_rows.append({
            "component": component_index + 1,
            "singular_value": singular_values[component_index],
            "trajectory_energy_fraction":
                trajectory_energy_fraction[component_index],
            "cumulative_trajectory_energy_fraction":
                cumulative_energy_fraction[component_index],
            "dominant_period_days": period_days,
            "dominant_spectral_power": spectral_power,
            "provisional_period_label":
                provisional_period_label(period_days),
        })

    period_table = pd.DataFrame(period_rows)

    period_table.to_csv(
        output_directory
        / "component_period_estimates.csv",
        index=False,
    )

    w_correlation = calculate_w_correlation(
        components=elementary_components,
        weights=weights,
        max_components=WCOR_COMPONENTS,
    )

    component_labels = np.arange(
        1,
        len(w_correlation) + 1,
    )

    pd.DataFrame(
        w_correlation,
        index=component_labels,
        columns=component_labels,
    ).to_csv(
        output_directory / "w_correlation_matrix.csv"
    )

    plot_component_norms(
        singular_values=singular_values,
        total_trajectory_energy=total_trajectory_energy,
        output_file=(
            output_directory
            / "component_norms.png"
        ),
    )

    plot_eigenvector_pairs(
        U=U,
        output_file=(
            output_directory
            / "eigenvector_pairs.png"
        ),
    )

    plot_w_correlation(
        w_correlation=w_correlation,
        output_file=(
            output_directory
            / "w_correlation_matrix.png"
        ),
    )

    plot_elementary_components(
        dates=series.index,
        components=elementary_components,
        output_file=(
            output_directory
            / "elementary_reconstructed_components.png"
        ),
    )

    retained_reconstruction = elementary_components.sum(axis=0)
    residual = x - retained_reconstruction

    plot_retained_reconstruction(
        dates=series.index,
        original=x,
        reconstruction=retained_reconstruction,
        output_file=(
            output_directory
            / "retained_reconstruction.png"
        ),
    )

    reconstruction_rmse = float(
        np.sqrt(np.mean(residual ** 2))
    )

    if np.std(retained_reconstruction) > 0:
        reconstruction_correlation = float(
            np.corrcoef(
                x,
                retained_reconstruction,
            )[0, 1]
        )
    else:
        reconstruction_correlation = np.nan

    retained_energy_fraction = float(
        cumulative_energy_fraction[-1]
    )

    print(
        "Retained trajectory energy: "
        f"{retained_energy_fraction:.4f}"
    )
    print(
        "Reconstruction correlation: "
        f"{reconstruction_correlation:.4f}"
    )

    return {
        "prefecture_code": prefecture_code,
        "variable": variable,
        "n_observations": number_of_observations,
        "window_length": window_length,
        "number_of_windows": number_of_windows,
        "window_fraction":
            window_length / number_of_observations,
        "number_of_retained_components":
            number_of_components,
        "retained_trajectory_energy_fraction":
            retained_energy_fraction,
        "reconstruction_rmse": reconstruction_rmse,
        "reconstruction_correlation":
            reconstruction_correlation,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(INPUT_FILE)
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    summary_rows = []

    for prefecture_code, variable in PILOT_SERIES:
        if variable not in dataframe.columns:
            print(f"Missing variable: {variable}")
            continue

        prefecture_data = dataframe[
            dataframe["prefecture_code"]
            == prefecture_code
        ].copy()

        if prefecture_data.empty:
            print(f"Missing prefecture: {prefecture_code}")
            continue

        series = prepare_daily_series(
            df=prefecture_data,
            variable=variable,
        )

        summary_rows.append(
            run_pilot(
                series=series,
                prefecture_code=prefecture_code,
                variable=variable,
            )
        )

    summary = pd.DataFrame(summary_rows)

    summary_file = OUT_DIR / "pilot_summary.csv"
    summary.to_csv(summary_file, index=False)

    print()
    print(summary)
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()