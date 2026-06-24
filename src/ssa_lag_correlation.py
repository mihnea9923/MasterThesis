from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "ssa_n2"
    / "ssa_reconstructed_series.csv.gz"
)

OUT_DIR = ROOT / "outputs" / "ssa_lag_correlation"

TARGET_VARIABLES = [
    "cases",
    "incidence_per_100k",
]

PREDICTOR_VARIABLES = [
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

FREQUENCY_BANDS = [
    "trend",
    "interannual",
    "annual",
    "semiannual",
    "seasonal_subannual",
    "subseasonal",
    "high_frequency",
]

MAX_LAG_BY_BAND = {
    "trend": 0,
    "interannual": 0,

    "annual": 30,
    "semiannual": 30,
    "seasonal_subannual": 30,

    "subseasonal": 21,
    "high_frequency": 14,
}

ANALYSIS_START_DATE = "2005-01-01"
ANALYSIS_END_DATE = "2016-12-31"

MIN_OBSERVATIONS = 365
MIN_STANDARD_DEVIATION = 1e-12


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def lagged_pearson_correlations(
    target: np.ndarray,
    predictor: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Pearson correlations for lags 0..max_lag.

    For lag L, the calculation is:

        corr(target[L:], predictor[:-L])

    Therefore, a positive lag means that the predictor occurs before
    the target by L days.

    The cross-products for every lag are calculated with FFT-based
    cross-correlation. Prefix sums are used to calculate the exact
    lag-specific means and variances over the overlapping observations.
    """
    target = np.asarray(target, dtype=float)
    predictor = np.asarray(predictor, dtype=float)

    if target.ndim != 1 or predictor.ndim != 1:
        raise ValueError("target and predictor must be one-dimensional arrays.")

    if len(target) != len(predictor):
        raise ValueError("target and predictor must have the same length.")

    if not np.isfinite(target).all() or not np.isfinite(predictor).all():
        raise ValueError("target and predictor must not contain NaN or infinity.")

    n_observations = len(target)
    actual_max_lag = min(max_lag, n_observations - 2)

    lags = np.arange(actual_max_lag + 1, dtype=int)
    overlap_counts = n_observations - lags

    full_cross_correlation = correlate(
        target,
        predictor,
        mode="full",
        method="fft",
    )

    # For lag L:
    # sum(target[L:] * predictor[:-L])
    cross_products = full_cross_correlation[
        n_observations - 1 + lags
    ]

    predictor_prefix = np.concatenate(
        ([0.0], np.cumsum(predictor, dtype=float))
    )
    target_prefix = np.concatenate(
        ([0.0], np.cumsum(target, dtype=float))
    )

    predictor_square_prefix = np.concatenate(
        ([0.0], np.cumsum(predictor * predictor, dtype=float))
    )
    target_square_prefix = np.concatenate(
        ([0.0], np.cumsum(target * target, dtype=float))
    )

    predictor_sums = predictor_prefix[n_observations - lags]
    target_sums = (
        target_prefix[n_observations]
        - target_prefix[lags]
    )

    predictor_square_sums = (
        predictor_square_prefix[n_observations - lags]
    )
    target_square_sums = (
        target_square_prefix[n_observations]
        - target_square_prefix[lags]
    )

    covariance_numerators = (
        cross_products
        - predictor_sums * target_sums / overlap_counts
    )

    predictor_variation = (
        predictor_square_sums
        - predictor_sums**2 / overlap_counts
    )
    target_variation = (
        target_square_sums
        - target_sums**2 / overlap_counts
    )

    # Small negative values can occur because of floating-point rounding.
    predictor_variation = np.maximum(predictor_variation, 0.0)
    target_variation = np.maximum(target_variation, 0.0)

    denominators = np.sqrt(
        predictor_variation * target_variation
    )

    correlations = np.divide(
        covariance_numerators,
        denominators,
        out=np.full_like(
            covariance_numerators,
            np.nan,
            dtype=float,
        ),
        where=denominators > 0,
    )

    correlations = np.clip(correlations, -1.0, 1.0)

    return correlations, overlap_counts


def prepare_prefecture_data(
    prefecture_data: pd.DataFrame,
) -> pd.DataFrame:
    result = prefecture_data.copy()
    result["date"] = pd.to_datetime(result["date"])

    if ANALYSIS_START_DATE is not None:
        result = result[
            result["date"] >= pd.Timestamp(ANALYSIS_START_DATE)
        ]

    if ANALYSIS_END_DATE is not None:
        result = result[
            result["date"] <= pd.Timestamp(ANALYSIS_END_DATE)
        ]

    return result.sort_values(["date", "variable"])


def create_band_wide_table(
    prefecture_data: pd.DataFrame,
    band: str,
) -> pd.DataFrame:
    wide = prefecture_data.pivot(
        index="date",
        columns="variable",
        values=band,
    ).sort_index()

    # The SSA input was created on a complete daily index, so missing
    # values are unexpected. Interpolation is only a defensive fallback.
    wide = wide.interpolate(limit_direction="both")

    return wide


def select_best_lag(
    correlations: np.ndarray,
) -> int | None:
    finite = np.isfinite(correlations)

    if not finite.any():
        return None

    valid_indices = np.flatnonzero(finite)
    valid_absolute_correlations = np.abs(
        correlations[valid_indices]
    )

    # np.argmax returns the first maximum, so ties are resolved by
    # selecting the smallest lag.
    best_position = int(
        np.argmax(valid_absolute_correlations)
    )

    return int(valid_indices[best_position])


def main() -> None:
    ensure_out_dir(OUT_DIR)

    required_columns = [
        "date",
        "prefecture_code",
        "variable",
        *FREQUENCY_BANDS,
    ]

    dataframe = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns,
    )

    dataframe["prefecture_code"] = (
        dataframe["prefecture_code"].astype("category")
    )
    dataframe["variable"] = dataframe["variable"].astype("category")

    all_lags_file = (
        OUT_DIR
        / "lag_correlation_all_lags.csv.gz"
    )
    best_lags_file = OUT_DIR / "best_lags.csv"
    skipped_pairs_file = OUT_DIR / "skipped_pairs.csv"

    all_lag_fields = [
        "prefecture_code",
        "target_variable",
        "predictor_variable",
        "frequency_band",
        "lag_days",
        "correlation",
        "absolute_correlation",
        "n_observations",
        "max_lag_searched",
        "lag_interpretation",
    ]

    best_rows: list[dict] = []
    skipped_rows: list[dict] = []

    with gzip.open(
        all_lags_file,
        "wt",
        newline="",
        encoding="utf-8",
    ) as all_lags_handle:
        all_lags_writer = csv.DictWriter(
            all_lags_handle,
            fieldnames=all_lag_fields,
        )
        all_lags_writer.writeheader()

        prefecture_codes = sorted(
            dataframe["prefecture_code"]
            .astype(str)
            .unique()
        )

        for prefecture_code in prefecture_codes:
            print(f"Processing {prefecture_code}")

            prefecture_data = dataframe[
                dataframe["prefecture_code"].astype(str)
                == prefecture_code
            ]

            prefecture_data = prepare_prefecture_data(
                prefecture_data
            )

            for band in FREQUENCY_BANDS:
                max_lag = MAX_LAG_BY_BAND[band]

                wide = create_band_wide_table(
                    prefecture_data=prefecture_data,
                    band=band,
                )

                for target_variable in TARGET_VARIABLES:
                    if target_variable not in wide.columns:
                        skipped_rows.append(
                            {
                                "prefecture_code": prefecture_code,
                                "target_variable": target_variable,
                                "predictor_variable": "",
                                "frequency_band": band,
                                "reason": "missing_target_variable",
                            }
                        )
                        continue

                    target = wide[
                        target_variable
                    ].to_numpy(dtype=float)

                    for predictor_variable in PREDICTOR_VARIABLES:
                        if predictor_variable not in wide.columns:
                            skipped_rows.append(
                                {
                                    "prefecture_code": prefecture_code,
                                    "target_variable": target_variable,
                                    "predictor_variable": predictor_variable,
                                    "frequency_band": band,
                                    "reason": "missing_predictor_variable",
                                }
                            )
                            continue

                        predictor = wide[
                            predictor_variable
                        ].to_numpy(dtype=float)

                        finite_mask = (
                            np.isfinite(target)
                            & np.isfinite(predictor)
                        )

                        target_values = target[finite_mask]
                        predictor_values = predictor[finite_mask]

                        if len(target_values) < MIN_OBSERVATIONS:
                            skipped_rows.append(
                                {
                                    "prefecture_code": prefecture_code,
                                    "target_variable": target_variable,
                                    "predictor_variable": predictor_variable,
                                    "frequency_band": band,
                                    "reason": "too_few_observations",
                                }
                            )
                            continue

                        if (
                            np.std(target_values)
                            < MIN_STANDARD_DEVIATION
                        ):
                            skipped_rows.append(
                                {
                                    "prefecture_code": prefecture_code,
                                    "target_variable": target_variable,
                                    "predictor_variable": predictor_variable,
                                    "frequency_band": band,
                                    "reason": "constant_target_band",
                                }
                            )
                            continue

                        if (
                            np.std(predictor_values)
                            < MIN_STANDARD_DEVIATION
                        ):
                            skipped_rows.append(
                                {
                                    "prefecture_code": prefecture_code,
                                    "target_variable": target_variable,
                                    "predictor_variable": predictor_variable,
                                    "frequency_band": band,
                                    "reason": "constant_predictor_band",
                                }
                            )
                            continue

                        correlations, overlap_counts = (
                            lagged_pearson_correlations(
                                target=target_values,
                                predictor=predictor_values,
                                max_lag=max_lag,
                            )
                        )

                        for lag_days, (
                            correlation,
                            n_observations,
                        ) in enumerate(
                            zip(
                                correlations,
                                overlap_counts,
                            )
                        ):
                            if not np.isfinite(correlation):
                                continue

                            all_lags_writer.writerow(
                                {
                                    "prefecture_code": prefecture_code,
                                    "target_variable": target_variable,
                                    "predictor_variable": predictor_variable,
                                    "frequency_band": band,
                                    "lag_days": lag_days,
                                    "correlation": float(correlation),
                                    "absolute_correlation": float(
                                        abs(correlation)
                                    ),
                                    "n_observations": int(n_observations),
                                    "max_lag_searched": max_lag,
                                    "lag_interpretation": (
                                        "predictor_leads_target"
                                    ),
                                }
                            )

                        best_lag = select_best_lag(correlations)

                        if best_lag is None:
                            skipped_rows.append(
                                {
                                    "prefecture_code": prefecture_code,
                                    "target_variable": target_variable,
                                    "predictor_variable": predictor_variable,
                                    "frequency_band": band,
                                    "reason": "no_finite_correlation",
                                }
                            )
                            continue

                        best_correlation = float(
                            correlations[best_lag]
                        )
                        zero_lag_correlation = float(
                            correlations[0]
                        )

                        best_rows.append(
                            {
                                "prefecture_code": prefecture_code,
                                "target_variable": target_variable,
                                "predictor_variable": predictor_variable,
                                "frequency_band": band,
                                "best_lag_days": best_lag,
                                "best_correlation": best_correlation,
                                "best_absolute_correlation": abs(
                                    best_correlation
                                ),
                                "zero_lag_correlation":
                                    zero_lag_correlation,
                                "absolute_correlation_gain_vs_zero":
                                    abs(best_correlation)
                                    - abs(zero_lag_correlation),
                                "n_observations_at_best_lag": int(
                                    overlap_counts[best_lag]
                                ),
                                "max_lag_searched": max_lag,
                                "lag_interpretation":
                                    "predictor_leads_target",
                            }
                        )

    best_lags = pd.DataFrame(best_rows)

    if not best_lags.empty:
        best_lags = best_lags.sort_values(
            [
                "prefecture_code",
                "target_variable",
                "frequency_band",
                "best_absolute_correlation",
                "predictor_variable",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                True,
            ],
        )

    best_lags.to_csv(
        best_lags_file,
        index=False,
    )

    skipped_pairs = pd.DataFrame(skipped_rows)
    skipped_pairs.to_csv(
        skipped_pairs_file,
        index=False,
    )

    print("\nSaved:")
    print(f"  {all_lags_file}")
    print(f"  {best_lags_file}")
    print(f"  {skipped_pairs_file}")

    print(f"\nBest-lag rows: {len(best_lags):,}")
    print(f"Skipped rows: {len(skipped_pairs):,}")

    if not best_lags.empty:
        print("\nLargest absolute lag correlations:")

        largest_correlations = (
            best_lags
            .sort_values(
                "best_absolute_correlation",
                ascending=False,
            )
            .head(20)
        )

        print(
            largest_correlations[
                [
                    "prefecture_code",
                    "target_variable",
                    "predictor_variable",
                    "frequency_band",
                    "best_lag_days",
                    "best_correlation",
                    "best_absolute_correlation",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
