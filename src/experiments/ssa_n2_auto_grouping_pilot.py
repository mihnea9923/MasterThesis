from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import fftconvolve, periodogram
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "ssa_n2_auto_grouping_pilot"

SERIES_CONFIGS = [
    ("JP-13", "cases", ROOT / "outputs" / "ssa_n2_pilot" / "JP-13_cases"),
    ("JP-13", "t2m", ROOT / "outputs" / "ssa_n2_pilot" / "JP-13_t2m"),
    (
        "JP-13",
        "t2m_std_anom",
        ROOT
        / "outputs"
        / "ssa_n2_pilot_difficult_variables"
        / "JP-13_t2m_std_anom",
    ),
    (
        "JP-13",
        "c2w_event",
        ROOT
        / "outputs"
        / "ssa_n2_pilot_difficult_variables"
        / "JP-13_c2w_event",
    ),
    (
        "JP-13",
        "tp",
        ROOT
        / "outputs"
        / "ssa_n2_pilot_difficult_variables"
        / "JP-13_tp",
    ),
]

# Pilot threshold. Complete-linkage clusters require every within-cluster
# absolute W-correlation to be at least this value.
WCOR_THRESHOLD = 0.70
CLUSTER_DISTANCE_THRESHOLD = 1.0 - WCOR_THRESHOLD

# A period close to the full record length is treated as baseline/trend-like.
BASELINE_PERIOD_FRACTION = 0.75

BROAD_BANDS = [
    "baseline",
    "interannual",
    "annual",
    "semiannual",
    "seasonal_subannual",
    "subseasonal",
    "high_frequency",
]


def classify_period(period_days: float, n_observations: int) -> str:
    if not np.isfinite(period_days):
        return "baseline"
    if period_days >= BASELINE_PERIOD_FRACTION * n_observations:
        return "baseline"
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
    return "unresolved_frequency"


def calculate_w_correlation(
    components: np.ndarray,
    window_length: int,
) -> np.ndarray:
    _, n_observations = components.shape
    number_of_windows = n_observations - window_length + 1

    weights = fftconvolve(
        np.ones(window_length),
        np.ones(number_of_windows),
        mode="full",
    )
    if len(weights) != n_observations:
        raise ValueError("W-correlation weights do not match series length.")

    weighted = components * np.sqrt(weights)[None, :]
    gram = weighted @ weighted.T
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(norms, norms)

    wcor = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0,
    )
    wcor = np.clip(wcor, -1.0, 1.0)
    np.fill_diagonal(wcor, 1.0)
    return wcor


def dominant_period_and_concentration(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
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

    if len(frequencies) == 0 or np.sum(power) <= 0:
        return np.inf, 0.0

    peak = int(np.argmax(power))
    frequency = float(frequencies[peak])
    period = np.inf if frequency <= 0 else float(1.0 / frequency)
    concentration = float(power[peak] / np.sum(power))
    return period, concentration


def pairwise_wcor_summary(
    indices: list[int],
    absolute_wcor: np.ndarray,
) -> tuple[float, float]:
    if len(indices) < 2:
        return np.nan, np.nan

    block = absolute_wcor[np.ix_(indices, indices)]
    values = block[np.triu_indices_from(block, k=1)]
    return float(np.min(values)), float(np.mean(values))


def cluster_within_band(
    indices: list[int],
    absolute_wcor: np.ndarray,
) -> list[list[int]]:
    if not indices:
        return []
    if len(indices) == 1:
        return [indices]

    block = absolute_wcor[np.ix_(indices, indices)]
    distance = np.clip(1.0 - block, 0.0, 1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    hierarchy = linkage(squareform(distance, checks=True), method="complete")
    labels = fcluster(
        hierarchy,
        t=CLUSTER_DISTANCE_THRESHOLD,
        criterion="distance",
    )

    clusters: list[list[int]] = []
    for label in sorted(np.unique(labels)):
        cluster = [
            indices[position]
            for position in range(len(indices))
            if labels[position] == label
        ]
        clusters.append(sorted(cluster))
    return clusters


def load_pilot_outputs(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    components_file = input_dir / "elementary_components.csv.gz"
    periods_file = input_dir / "component_period_estimates.csv"

    if not components_file.exists():
        raise FileNotFoundError(components_file)
    if not periods_file.exists():
        raise FileNotFoundError(periods_file)

    series_df = pd.read_csv(components_file)
    series_df["date"] = pd.to_datetime(series_df["date"])

    component_columns = sorted(
        column for column in series_df.columns if column.startswith("RC_")
    )
    components = series_df[component_columns].to_numpy(dtype=float).T

    metadata = (
        pd.read_csv(periods_file)
        .sort_values("component")
        .reset_index(drop=True)
    )
    if len(metadata) != len(component_columns):
        raise ValueError(
            f"{input_dir}: {len(metadata)} metadata rows but "
            f"{len(component_columns)} reconstructed components."
        )

    return series_df, metadata, components


def build_groups(
    metadata: pd.DataFrame,
    components: np.ndarray,
    absolute_wcor: np.ndarray,
    n_observations: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], set[int]]:
    metadata = metadata.copy()
    metadata["source_band"] = metadata["dominant_period_days"].apply(
        lambda period: classify_period(float(period), n_observations)
    )

    energy_column = next(
        (
            column
            for column in ["trajectory_energy_fraction", "energy_fraction"]
            if column in metadata.columns
        ),
        None,
    )

    rows: list[dict] = []
    accepted_group_series: dict[str, np.ndarray] = {}
    accepted_indices: set[int] = set()
    processed_indices: set[int] = set()
    group_number = 1

    for band in BROAD_BANDS:
        numbers = (
            metadata.loc[metadata["source_band"] == band, "component"]
            .astype(int)
            .tolist()
        )
        indices = [number - 1 for number in numbers]

        for cluster in cluster_within_band(indices, absolute_wcor):
            processed_indices.update(cluster)
            component_numbers = [index + 1 for index in cluster]
            reconstructed = components[cluster].sum(axis=0)
            period, concentration = dominant_period_and_concentration(
                reconstructed
            )
            final_band = classify_period(period, n_observations)
            minimum_wcor, mean_wcor = pairwise_wcor_summary(
                cluster,
                absolute_wcor,
            )

            # Only RC1 is automatically accepted as baseline. Other isolated
            # full-record-length components are kept for manual review because
            # they may be edge effects rather than trend.
            if band == "baseline" and component_numbers == [1]:
                status = "accepted_baseline"
            elif len(cluster) >= 2 and final_band in BROAD_BANDS:
                status = "accepted_cluster"
            elif len(cluster) >= 2:
                status = "unresolved_cluster"
            else:
                status = "singleton_candidate"

            group_id = f"G{group_number:03d}"
            group_number += 1

            if status.startswith("accepted"):
                accepted_indices.update(cluster)
                accepted_group_series[group_id] = reconstructed

            if energy_column:
                energy = float(
                    metadata.loc[
                        metadata["component"].isin(component_numbers),
                        energy_column,
                    ].sum()
                )
            else:
                energy = np.nan

            singular_ratio = np.nan
            if "singular_value" in metadata.columns and len(cluster) >= 2:
                singular_values = metadata.loc[
                    metadata["component"].isin(component_numbers),
                    "singular_value",
                ].astype(float)
                singular_ratio = float(
                    singular_values.min() / singular_values.max()
                )

            rows.append(
                {
                    "group_id": group_id,
                    "source_band": band,
                    "final_band": final_band,
                    "status": status,
                    "components": ",".join(map(str, component_numbers)),
                    "number_of_components": len(cluster),
                    "dominant_period_days": period,
                    "spectral_concentration": concentration,
                    "minimum_absolute_wcor": minimum_wcor,
                    "mean_absolute_wcor": mean_wcor,
                    "singular_value_ratio": singular_ratio,
                    "trajectory_energy_fraction": energy,
                }
            )

    # Components outside the declared frequency bins are explicitly reported.
    all_indices = set(range(components.shape[0]))
    for index in sorted(all_indices - processed_indices):
        component_number = index + 1
        rows.append(
            {
                "group_id": f"G{group_number:03d}",
                "source_band": "unresolved_frequency",
                "final_band": "unresolved_frequency",
                "status": "unresolved_frequency_component",
                "components": str(component_number),
                "number_of_components": 1,
                "dominant_period_days": float(
                    metadata.loc[
                        metadata["component"] == component_number,
                        "dominant_period_days",
                    ].iloc[0]
                ),
                "spectral_concentration": np.nan,
                "minimum_absolute_wcor": np.nan,
                "mean_absolute_wcor": np.nan,
                "singular_value_ratio": np.nan,
                "trajectory_energy_fraction": (
                    float(
                        metadata.loc[
                            metadata["component"] == component_number,
                            energy_column,
                        ].iloc[0]
                    )
                    if energy_column
                    else np.nan
                ),
            }
        )
        group_number += 1

    return pd.DataFrame(rows), accepted_group_series, accepted_indices


def create_broad_reconstruction(
    series_df: pd.DataFrame,
    components: np.ndarray,
    group_summary: pd.DataFrame,
    accepted_group_series: dict[str, np.ndarray],
    accepted_indices: set[int],
) -> pd.DataFrame:
    output = series_df[["date", "original"]].copy()
    for band in BROAD_BANDS:
        output[band] = 0.0

    summary_by_id = group_summary.set_index("group_id")

    for group_id, reconstructed in accepted_group_series.items():
        band = str(summary_by_id.loc[group_id, "final_band"])
        output[band] += reconstructed

    unresolved_indices = [
        index
        for index in range(components.shape[0])
        if index not in accepted_indices
    ]
    output["unresolved_retained_components"] = (
        components[unresolved_indices].sum(axis=0)
        if unresolved_indices
        else 0.0
    )

    retained = components.sum(axis=0)
    output["retained_component_reconstruction"] = retained
    output["truncated_svd_residual"] = (
        output["original"].to_numpy(dtype=float) - retained
    )

    check_columns = BROAD_BANDS + [
        "unresolved_retained_components",
        "truncated_svd_residual",
    ]
    output["complete_reconstruction_check"] = output[check_columns].sum(axis=1)
    return output


def plot_broad_reconstruction(
    dataframe: pd.DataFrame,
    prefecture_code: str,
    variable: str,
    output_file: Path,
) -> None:
    columns = ["original"] + BROAD_BANDS + [
        "unresolved_retained_components",
        "truncated_svd_residual",
    ]

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=(16, 2.2 * len(columns)),
        sharex=True,
    )

    for axis, column in zip(np.atleast_1d(axes), columns):
        axis.plot(dataframe["date"], dataframe[column], linewidth=0.75)
        axis.set_ylabel(column)
        axis.grid(alpha=0.2)

    axes[-1].set_xlabel("Date")
    fig.suptitle(
        f"{prefecture_code} {variable}: automatic N/2 SSA grouping pilot",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def process_series(
    prefecture_code: str,
    variable: str,
    input_dir: Path,
) -> pd.DataFrame:
    output_dir = OUTPUT_ROOT / f"{prefecture_code}_{variable}"
    output_dir.mkdir(parents=True, exist_ok=True)

    series_df, metadata, components = load_pilot_outputs(input_dir)
    n_observations = len(series_df)
    window_length = n_observations // 2

    absolute_wcor = np.abs(
        calculate_w_correlation(
            components=components,
            window_length=window_length,
        )
    )

    group_summary, accepted_series, accepted_indices = build_groups(
        metadata=metadata,
        components=components,
        absolute_wcor=absolute_wcor,
        n_observations=n_observations,
    )

    broad = create_broad_reconstruction(
        series_df=series_df,
        components=components,
        group_summary=group_summary,
        accepted_group_series=accepted_series,
        accepted_indices=accepted_indices,
    )

    group_summary.to_csv(output_dir / "auto_group_summary.csv", index=False)
    broad.to_csv(
        output_dir / "auto_grouped_reconstructed_series.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        absolute_wcor,
        index=np.arange(1, len(absolute_wcor) + 1),
        columns=np.arange(1, len(absolute_wcor) + 1),
    ).to_csv(
        output_dir / "absolute_w_correlation_matrix.csv.gz",
        compression="gzip",
    )

    config = {
        "prefecture_code": prefecture_code,
        "variable": variable,
        "window_rule": "L = N // 2",
        "retained_components": int(components.shape[0]),
        "wcor_threshold": WCOR_THRESHOLD,
        "cluster_linkage": "complete",
        "component_dissimilarity": "1 - absolute W-correlation",
        "baseline_period_fraction": BASELINE_PERIOD_FRACTION,
        "acceptance_rule": (
            "RC1 may be accepted as baseline; other accepted groups require "
            "at least two components. Singletons and unresolved frequencies "
            "remain in unresolved_retained_components."
        ),
    }
    with (output_dir / "auto_grouping_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    plot_broad_reconstruction(
        dataframe=broad,
        prefecture_code=prefecture_code,
        variable=variable,
        output_file=output_dir / "auto_grouped_reconstruction.png",
    )

    error = broad["original"] - broad["complete_reconstruction_check"]
    accepted_count = int(
        group_summary["status"].str.startswith("accepted").sum()
    )
    unresolved_count = int(
        (~group_summary["status"].str.startswith("accepted")).sum()
    )

    print("\n" + "=" * 72)
    print(f"{prefecture_code} - {variable}")
    print(f"Retained components: {components.shape[0]}")
    print(f"Accepted groups: {accepted_count}")
    print(f"Groups/components requiring review: {unresolved_count}")
    print(f"Maximum reconstruction error: {np.max(np.abs(error)):.12f}")
    print(
        group_summary[
            [
                "group_id",
                "final_band",
                "status",
                "components",
                "dominant_period_days",
                "minimum_absolute_wcor",
                "spectral_concentration",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    return group_summary.assign(
        prefecture_code=prefecture_code,
        variable=variable,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = [
        process_series(prefecture_code, variable, Path(input_dir))
        for prefecture_code, variable, input_dir in SERIES_CONFIGS
    ]

    combined = pd.concat(summaries, ignore_index=True)
    combined_file = OUTPUT_ROOT / "combined_auto_group_summary.csv"
    combined.to_csv(combined_file, index=False)
    print(f"\nSaved: {combined_file}")


if __name__ == "__main__":
    main()
