from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "ssa_n2_pilot"
    / "JP-13_t2m"
    / "elementary_components.csv.gz"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "ssa_n2_grouped_pilot"
    / "JP-13_t2m"
)


CANDIDATE_GROUPS = {
    "baseline": [1],
    "annual_primary": [2, 3],
    "semiannual": [4, 5],
    "seasonal_121d": [6, 7],
    "annual_secondary_322d": [8, 9],
    "seasonal_71d": [10, 11],
    "seasonal_100d": [12, 13],
}


def component_column(component_number: int) -> str:
    return f"RC_{component_number:03d}"


def validate_groups(
    dataframe: pd.DataFrame,
    groups: dict[str, list[int]],
) -> None:
    used_components: list[int] = []

    for group_name, component_numbers in groups.items():
        if not component_numbers:
            raise ValueError(
                f"Group '{group_name}' contains no components."
            )

        for component_number in component_numbers:
            column = component_column(component_number)

            if column not in dataframe.columns:
                raise ValueError(
                    f"Missing component column '{column}' "
                    f"for group '{group_name}'."
                )

            used_components.append(component_number)

    duplicates = sorted({
        component
        for component in used_components
        if used_components.count(component) > 1
    })

    if duplicates:
        raise ValueError(
            f"Components assigned to multiple groups: {duplicates}"
        )


def reconstruct_group(
    dataframe: pd.DataFrame,
    component_numbers: list[int],
) -> np.ndarray:
    columns = [
        component_column(number)
        for number in component_numbers
    ]

    return dataframe[columns].sum(axis=1).to_numpy(dtype=float)


def dominant_period_days(
    values: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).all():
        values = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .to_numpy()
        )

    if np.std(values) < 1e-12:
        return np.inf, 0.0, 0.0

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
        return np.inf, 0.0, 0.0

    peak_index = int(np.argmax(power))
    dominant_frequency = float(frequencies[peak_index])
    dominant_power = float(power[peak_index])

    if dominant_frequency <= 0:
        return np.inf, dominant_frequency, dominant_power

    dominant_period = float(1.0 / dominant_frequency)

    return (
        dominant_period,
        dominant_frequency,
        dominant_power,
    )


def classify_period(period_days: float) -> str:
    if not np.isfinite(period_days):
        return "constant_or_trend"

    if period_days >= 3 * 365.25:
        return "trend_or_low_frequency"

    if period_days >= 1.2 * 365.25:
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

    return "ultra_high_frequency"


def create_grouped_series(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = dataframe[["date", "original"]].copy()

    frequency_rows: list[dict] = []

    selected_components: set[int] = set()

    for group_name, component_numbers in CANDIDATE_GROUPS.items():
        selected_components.update(component_numbers)

        reconstructed = reconstruct_group(
            dataframe=dataframe,
            component_numbers=component_numbers,
        )

        output[group_name] = reconstructed

        (
            period_days,
            frequency_per_day,
            dominant_power,
        ) = dominant_period_days(reconstructed)

        frequency_rows.append({
            "group": group_name,
            "components": ",".join(
                str(number)
                for number in component_numbers
            ),
            "number_of_components": len(component_numbers),
            "dominant_period_days": period_days,
            "dominant_frequency_cycles_per_day":
                frequency_per_day,
            "dominant_power": dominant_power,
            "frequency_label": classify_period(period_days),
            "group_variance": float(
                np.var(reconstructed, ddof=1)
            ),
        })

    all_component_numbers = sorted(
        int(column.replace("RC_", ""))
        for column in dataframe.columns
        if column.startswith("RC_")
    )

    unassigned_components = [
        number
        for number in all_component_numbers
        if number not in selected_components
    ]

    output["selected_groups_sum"] = output[
        list(CANDIDATE_GROUPS)
    ].sum(axis=1)

    if unassigned_components:
        output["unassigned_retained_components"] = (
            reconstruct_group(
                dataframe=dataframe,
                component_numbers=unassigned_components,
            )
        )
    else:
        output["unassigned_retained_components"] = 0.0

    retained_columns = [
        component_column(number)
        for number in all_component_numbers
    ]

    output["retained_100_component_reconstruction"] = (
        dataframe[retained_columns]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )

    output["truncated_svd_residual"] = (
        output["original"]
        - output["retained_100_component_reconstruction"]
    )

    output["complete_reconstruction_check"] = (
        output["selected_groups_sum"]
        + output["unassigned_retained_components"]
        + output["truncated_svd_residual"]
    )

    frequency_summary = pd.DataFrame(frequency_rows)

    return output, frequency_summary


def plot_grouped_reconstruction(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    plot_columns = [
        "original",
        "baseline",
        "annual_primary",
        "annual_secondary_322d",
        "semiannual",
        "seasonal_121d",
        "seasonal_100d",
        "seasonal_71d",
        "unassigned_retained_components",
        "truncated_svd_residual",
    ]

    number_of_plots = len(plot_columns)

    fig, axes = plt.subplots(
        number_of_plots,
        1,
        figsize=(16, 2.4 * number_of_plots),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    for axis, column in zip(axes, plot_columns):
        axis.plot(
            dataframe["date"],
            dataframe[column],
            linewidth=0.8,
        )

        axis.set_ylabel(column)
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")

    fig.suptitle(
        "JP-13 t2m: grouped N/2 SSA reconstruction",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def save_grouping_definition(
    output_file: Path,
) -> None:
    grouping_definition = {
        "prefecture_code": "JP-13",
        "variable": "t2m",
        "window_rule": "L = N // 2",
        "number_of_retained_components": 100,
        "candidate_groups": CANDIDATE_GROUPS,
        "note": (
            "These are pilot groups based on adjacent eigenvector "
            "pairs, elementary reconstructed components, "
            "W-correlation diagnostics, and period estimates."
        ),
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            grouping_definition,
            file,
            indent=2,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(INPUT_FILE)
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    validate_groups(
        dataframe=dataframe,
        groups=CANDIDATE_GROUPS,
    )

    grouped_series, frequency_summary = (
        create_grouped_series(dataframe)
    )

    grouped_series_file = (
        OUTPUT_DIR
        / "grouped_reconstructed_series.csv.gz"
    )

    frequency_summary_file = (
        OUTPUT_DIR
        / "group_frequency_summary.csv"
    )

    grouped_series.to_csv(
        grouped_series_file,
        index=False,
        compression="gzip",
    )

    frequency_summary.to_csv(
        frequency_summary_file,
        index=False,
    )

    save_grouping_definition(
        OUTPUT_DIR / "grouping_definition.json"
    )

    plot_grouped_reconstruction(
        dataframe=grouped_series,
        output_file=(
            OUTPUT_DIR
            / "grouped_reconstruction.png"
        ),
    )

    reconstruction_error = (
        grouped_series["original"]
        - grouped_series["complete_reconstruction_check"]
    )

    print()
    print("Candidate group frequency summary:")
    print(
        frequency_summary[
            [
                "group",
                "components",
                "dominant_period_days",
                "frequency_label",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Maximum absolute reconstruction check error: "
        f"{np.max(np.abs(reconstruction_error)):.12f}"
    )

    print()
    print(f"Saved: {grouped_series_file}")
    print(f"Saved: {frequency_summary_file}")
    print(
        f"Saved: "
        f"{OUTPUT_DIR / 'grouping_definition.json'}"
    )
    print(
        f"Saved: "
        f"{OUTPUT_DIR / 'grouped_reconstruction.png'}"
    )


if __name__ == "__main__":
    main()