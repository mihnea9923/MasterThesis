from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from ssa_n2_pilot import (
    build_trajectory_matrix,
    prepare_daily_series,
    reconstruct_elementary_components,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "intermediate_series"
    / "model_ready_analysis_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "ssa_n2_component_sensitivity"
)

PREFECTURE_CODE = "JP-13"

SENSITIVITY_VARIABLES = [
    "t2m_std_anom",
    "c2w_event",
    "w2c_event",
    "c2w_transition_intensity",
    "w2c_transition_intensity",
    "rh",
    "tp",
    "nw_wind",
]

COMPONENT_CUTOFFS = [100, 200, 400]
MAX_COMPONENTS = max(COMPONENT_CUTOFFS)

RANDOM_STATE = 42


def calculate_reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, float]:
    residual = original - reconstruction

    rmse = float(
        np.sqrt(
            np.mean(residual**2)
        )
    )

    original_std = float(
        np.std(original, ddof=1)
    )

    if original_std > 0:
        normalized_rmse = rmse / original_std
    else:
        normalized_rmse = np.nan

    original_variance = float(
        np.var(original, ddof=1)
    )

    residual_variance = float(
        np.var(residual, ddof=1)
    )

    if original_variance > 0:
        explained_variance = (
            1.0
            - residual_variance / original_variance
        )
    else:
        explained_variance = np.nan

    reconstruction_std = float(
        np.std(reconstruction, ddof=1)
    )

    if original_std > 0 and reconstruction_std > 0:
        reconstruction_correlation = float(
            np.corrcoef(
                original,
                reconstruction,
            )[0, 1]
        )
    else:
        reconstruction_correlation = np.nan

    return {
        "reconstruction_rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "time_domain_explained_variance":
            explained_variance,
        "reconstruction_correlation":
            reconstruction_correlation,
        "residual_std": float(
            np.std(residual, ddof=1)
        ),
    }


def analyse_variable(
    dataframe: pd.DataFrame,
    variable: str,
) -> list[dict]:
    variable_start = time.perf_counter()

    prefecture_data = dataframe[
        dataframe["prefecture_code"]
        == PREFECTURE_CODE
    ].copy()

    series = prepare_daily_series(
        df=prefecture_data,
        variable=variable,
    )

    original = series.to_numpy(dtype=float)

    n_observations = len(original)
    window_length = n_observations // 2
    number_of_windows = (
        n_observations
        - window_length
        + 1
    )

    print()
    print("=" * 70)
    print(
        f"Processing {PREFECTURE_CODE} - {variable}"
    )
    print(f"N = {n_observations}")
    print(f"L = {window_length}")
    print(f"K = {number_of_windows}")
    print(
        f"Calculating the first "
        f"{MAX_COMPONENTS} components..."
    )

    trajectory_matrix = build_trajectory_matrix(
        x=original,
        window_length=window_length,
    )

    available_components = min(
        trajectory_matrix.shape
    )

    number_of_components = min(
        MAX_COMPONENTS,
        available_components - 1,
    )

    decomposition_start = time.perf_counter()

    U, singular_values, Vt = randomized_svd(
        trajectory_matrix,
        n_components=number_of_components,
        n_iter=5,
        random_state=RANDOM_STATE,
    )

    decomposition_seconds = (
        time.perf_counter()
        - decomposition_start
    )

    print(
        "Randomized SVD completed in "
        f"{decomposition_seconds:.1f} seconds."
    )

    total_trajectory_energy = float(
        np.einsum(
            "ij,ij->",
            trajectory_matrix,
            trajectory_matrix,
        )
    )

    elementary_components, _ = (
        reconstruct_elementary_components(
            U=U,
            singular_values=singular_values,
            Vt=Vt,
            window_length=window_length,
            number_of_windows=number_of_windows,
        )
    )

    singular_value_energy = (
        singular_values**2
    )

    cumulative_energy = np.cumsum(
        singular_value_energy
    )

    rows: list[dict] = []

    for cutoff in COMPONENT_CUTOFFS:
        actual_cutoff = min(
            cutoff,
            elementary_components.shape[0],
        )

        reconstruction = elementary_components[
            :actual_cutoff
        ].sum(axis=0)

        metrics = calculate_reconstruction_metrics(
            original=original,
            reconstruction=reconstruction,
        )

        retained_energy_fraction = float(
            cumulative_energy[actual_cutoff - 1]
            / total_trajectory_energy
        )

        row = {
            "prefecture_code":
                PREFECTURE_CODE,
            "variable":
                variable,
            "n_observations":
                n_observations,
            "window_length":
                window_length,
            "window_fraction":
                window_length / n_observations,
            "requested_components":
                cutoff,
            "retained_components":
                actual_cutoff,
            "retained_trajectory_energy_fraction":
                retained_energy_fraction,
            **metrics,
            "svd_runtime_seconds":
                decomposition_seconds,
        }

        rows.append(row)

        print(
            f"k={actual_cutoff:3d} | "
            f"energy={retained_energy_fraction:.4f} | "
            f"corr="
            f"{metrics['reconstruction_correlation']:.4f} | "
            f"explained_var="
            f"{metrics['time_domain_explained_variance']:.4f} | "
            f"NRMSE="
            f"{metrics['normalized_rmse']:.4f}"
        )

    variable_seconds = (
        time.perf_counter()
        - variable_start
    )

    print(
        f"Completed {variable} in "
        f"{variable_seconds:.1f} seconds."
    )

    # Release the large arrays before processing the next variable.
    del trajectory_matrix
    del elementary_components
    del U
    del singular_values
    del Vt

    gc.collect()

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(INPUT_FILE)
    dataframe["date"] = pd.to_datetime(
        dataframe["date"]
    )

    all_rows: list[dict] = []

    for variable in SENSITIVITY_VARIABLES:
        if variable not in dataframe.columns:
            print(
                f"Skipping missing variable: {variable}"
            )
            continue

        variable_rows = analyse_variable(
            dataframe=dataframe,
            variable=variable,
        )

        all_rows.extend(variable_rows)

        # Save after every variable so progress is not lost if a
        # later decomposition fails.
        partial_summary = pd.DataFrame(all_rows)

        partial_summary.to_csv(
            OUTPUT_DIR
            / "sensitivity_summary.csv",
            index=False,
        )

    summary = pd.DataFrame(all_rows)

    summary_file = (
        OUTPUT_DIR
        / "sensitivity_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("Sensitivity analysis completed.")
    print(f"Saved: {summary_file}")

    display_columns = [
        "variable",
        "retained_components",
        "retained_trajectory_energy_fraction",
        "reconstruction_correlation",
        "time_domain_explained_variance",
        "normalized_rmse",
    ]

    print()
    print(
        summary[display_columns]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()