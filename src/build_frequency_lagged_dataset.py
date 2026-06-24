from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SSA_FILE = (
    ROOT
    / "outputs"
    / "ssa_n2"
    / "ssa_reconstructed_series.csv.gz"
)

BEST_LAGS_FILE = (
    ROOT
    / "outputs"
    / "ssa_lag_correlation"
    / "best_lags.csv"
)

OUT_DIR = ROOT / "outputs" / "modeling"

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

# Protect against accidentally using the 60-day semiannual sensitivity file.
EXPECTED_MAX_LAG_BY_BAND = {
    "trend": 0,
    "interannual": 0,
    "annual": 30,
    "semiannual": 30,
    "seasonal_subannual": 30,
    "subseasonal": 21,
    "high_frequency": 14,
}

ANALYSIS_START_DATE = pd.Timestamp("2005-01-01")
TRAIN_END_DATE = pd.Timestamp("2016-12-31")
VALIDATION_START_DATE = pd.Timestamp("2017-01-01")
VALIDATION_END_DATE = pd.Timestamp("2017-12-31")
TEST_START_DATE = pd.Timestamp("2018-01-01")
TEST_END_DATE = pd.Timestamp("2019-12-31")

MIN_STANDARD_DEVIATION = 1e-12


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_best_lags(best_lags: pd.DataFrame) -> None:
    required_columns = {
        "prefecture_code",
        "target_variable",
        "predictor_variable",
        "frequency_band",
        "best_lag_days",
        "max_lag_searched",
    }

    missing_columns = required_columns - set(best_lags.columns)
    if missing_columns:
        raise ValueError(
            "best_lags.csv is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    duplicated = best_lags.duplicated(
        subset=[
            "prefecture_code",
            "target_variable",
            "predictor_variable",
            "frequency_band",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = best_lags.loc[
            duplicated,
            [
                "prefecture_code",
                "target_variable",
                "predictor_variable",
                "frequency_band",
            ],
        ]
        raise ValueError(
            "best_lags.csv contains duplicated lag definitions:\n"
            + duplicate_rows.head(20).to_string(index=False)
        )

    for band, expected_max_lag in EXPECTED_MAX_LAG_BY_BAND.items():
        observed = (
            best_lags.loc[
                best_lags["frequency_band"] == band,
                "max_lag_searched",
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(observed) == 0:
            continue

        if set(observed) != {expected_max_lag}:
            raise ValueError(
                f"Unexpected lag-search limit for {band!r}. "
                f"Expected {expected_max_lag}, found {sorted(observed)}. "
                "Make sure BEST_LAGS_FILE points to the final 30-day "
                "semiannual run, not the 60-day sensitivity run."
            )


def assign_split(dates: pd.Series) -> pd.Series:
    conditions = [
        dates.between(
            ANALYSIS_START_DATE,
            TRAIN_END_DATE,
            inclusive="both",
        ),
        dates.between(
            VALIDATION_START_DATE,
            VALIDATION_END_DATE,
            inclusive="both",
        ),
        dates.between(
            TEST_START_DATE,
            TEST_END_DATE,
            inclusive="both",
        ),
    ]

    choices = [
        "train",
        "validation",
        "test",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="outside_analysis_period",
        ),
        index=dates.index,
        dtype="object",
    )


def build_lag_lookup(
    best_lags: pd.DataFrame,
) -> dict[tuple[str, str, str, str], int]:
    lookup: dict[tuple[str, str, str, str], int] = {}

    for row in best_lags.itertuples(index=False):
        key = (
            str(row.prefecture_code),
            str(row.target_variable),
            str(row.predictor_variable),
            str(row.frequency_band),
        )
        lookup[key] = int(row.best_lag_days)

    return lookup


def create_band_wide_table(
    prefecture_data: pd.DataFrame,
    band: str,
) -> pd.DataFrame:
    wide = prefecture_data.pivot(
        index="date",
        columns="variable",
        values=band,
    ).sort_index()

    expected_index = pd.date_range(
        wide.index.min(),
        wide.index.max(),
        freq="D",
    )

    wide = wide.reindex(expected_index)
    wide.index.name = "date"

    # Missing values are not expected because SSA was run on complete
    # daily series. This is a defensive fallback only.
    wide = wide.interpolate(limit_direction="both")

    return wide


def main() -> None:
    ensure_out_dir(OUT_DIR)

    output_file = (
        OUT_DIR
        / "frequency_lagged_dataset.csv.gz"
    )
    summary_file = (
        OUT_DIR
        / "frequency_lagged_dataset_summary.csv"
    )
    applied_lags_file = (
        OUT_DIR
        / "applied_lags.csv"
    )

    required_ssa_columns = [
        "date",
        "prefecture_code",
        "variable",
        *FREQUENCY_BANDS,
    ]

    print("Reading SSA reconstructions...")
    ssa_data = pd.read_csv(
        SSA_FILE,
        usecols=required_ssa_columns,
    )

    ssa_data["date"] = pd.to_datetime(ssa_data["date"])
    ssa_data["prefecture_code"] = (
        ssa_data["prefecture_code"].astype(str)
    )
    ssa_data["variable"] = ssa_data["variable"].astype(str)

    ssa_data = ssa_data[
        ssa_data["date"].between(
            ANALYSIS_START_DATE,
            TEST_END_DATE,
            inclusive="both",
        )
    ].copy()

    print("Reading selected lags...")
    best_lags = pd.read_csv(BEST_LAGS_FILE)
    best_lags["prefecture_code"] = (
        best_lags["prefecture_code"].astype(str)
    )
    best_lags["target_variable"] = (
        best_lags["target_variable"].astype(str)
    )
    best_lags["predictor_variable"] = (
        best_lags["predictor_variable"].astype(str)
    )
    best_lags["frequency_band"] = (
        best_lags["frequency_band"].astype(str)
    )

    validate_best_lags(best_lags)
    lag_lookup = build_lag_lookup(best_lags)

    feature_columns = [
        f"{predictor}_lagged"
        for predictor in PREDICTOR_VARIABLES
    ]

    summary_rows: list[dict] = []
    applied_lag_rows: list[dict] = []

    output_header_written = False

    with gzip.open(
        output_file,
        "wt",
        newline="",
        encoding="utf-8",
    ) as output_handle:

        prefecture_codes = sorted(
            ssa_data["prefecture_code"].unique()
        )

        for prefecture_code in prefecture_codes:
            print(f"Processing {prefecture_code}")

            prefecture_data = ssa_data[
                ssa_data["prefecture_code"]
                == prefecture_code
            ]

            for band in FREQUENCY_BANDS:
                wide = create_band_wide_table(
                    prefecture_data=prefecture_data,
                    band=band,
                )

                missing_variables = (
                    set(TARGET_VARIABLES + PREDICTOR_VARIABLES)
                    - set(wide.columns)
                )

                if missing_variables:
                    raise ValueError(
                        f"{prefecture_code}/{band} is missing variables: "
                        + ", ".join(sorted(missing_variables))
                    )

                for target_variable in TARGET_VARIABLES:
                    target_values = wide[
                        target_variable
                    ].to_numpy(dtype=float)

                    if (
                        np.std(target_values)
                        < MIN_STANDARD_DEVIATION
                    ):
                        summary_rows.append(
                            {
                                "prefecture_code": prefecture_code,
                                "target_variable": target_variable,
                                "frequency_band": band,
                                "status": "skipped_constant_target_band",
                                "rows_before_lag_drop": len(wide),
                                "rows_after_lag_drop": 0,
                                "dropped_initial_rows": len(wide),
                                "train_rows": 0,
                                "validation_rows": 0,
                                "test_rows": 0,
                                "maximum_applied_lag_days": np.nan,
                            }
                        )
                        continue

                    model_data = pd.DataFrame(
                        {
                            "date": wide.index,
                            "prefecture_code": prefecture_code,
                            "target_variable": target_variable,
                            "frequency_band": band,
                            "target_component": target_values,
                        }
                    )

                    applied_lags_for_group: list[int] = []

                    for predictor_variable in PREDICTOR_VARIABLES:
                        predictor_values = wide[
                            predictor_variable
                        ].astype(float)

                        lookup_key = (
                            prefecture_code,
                            target_variable,
                            predictor_variable,
                            band,
                        )

                        if lookup_key in lag_lookup:
                            lag_days = lag_lookup[lookup_key]
                            lag_source = "selected_on_training_period"
                        else:
                            # A missing lag is valid only when the
                            # reconstructed predictor band is constant.
                            if (
                                np.std(predictor_values.to_numpy())
                                >= MIN_STANDARD_DEVIATION
                            ):
                                raise ValueError(
                                    "Missing selected lag for a nonconstant "
                                    "predictor band: "
                                    f"{lookup_key}"
                                )

                            lag_days = 0
                            lag_source = "constant_predictor_default_zero"

                        expected_max_lag = (
                            EXPECTED_MAX_LAG_BY_BAND[band]
                        )

                        if not 0 <= lag_days <= expected_max_lag:
                            raise ValueError(
                                f"Invalid lag {lag_days} for {lookup_key}; "
                                f"allowed range is 0..{expected_max_lag}."
                            )

                        feature_name = (
                            f"{predictor_variable}_lagged"
                        )

                        # Positive lag means predictor[t - lag] is used
                        # to explain target[t].
                        model_data[feature_name] = (
                            predictor_values.shift(lag_days).to_numpy()
                        )

                        applied_lags_for_group.append(lag_days)

                        applied_lag_rows.append(
                            {
                                "prefecture_code": prefecture_code,
                                "target_variable": target_variable,
                                "predictor_variable": predictor_variable,
                                "frequency_band": band,
                                "applied_lag_days": lag_days,
                                "lag_source": lag_source,
                            }
                        )

                    model_data["split"] = assign_split(
                        model_data["date"]
                    )

                    rows_before = len(model_data)

                    # NaNs should occur only at the beginning because of
                    # positive predictor lags.
                    model_data = model_data.dropna(
                        subset=feature_columns
                    ).copy()

                    model_data = model_data[
                        model_data["split"]
                        != "outside_analysis_period"
                    ].copy()

                    rows_after = len(model_data)

                    split_counts = (
                        model_data["split"]
                        .value_counts()
                        .to_dict()
                    )

                    summary_rows.append(
                        {
                            "prefecture_code": prefecture_code,
                            "target_variable": target_variable,
                            "frequency_band": band,
                            "status": "created",
                            "rows_before_lag_drop": rows_before,
                            "rows_after_lag_drop": rows_after,
                            "dropped_initial_rows":
                                rows_before - rows_after,
                            "train_rows": split_counts.get(
                                "train",
                                0,
                            ),
                            "validation_rows": split_counts.get(
                                "validation",
                                0,
                            ),
                            "test_rows": split_counts.get(
                                "test",
                                0,
                            ),
                            "maximum_applied_lag_days": max(
                                applied_lags_for_group
                            ),
                        }
                    )

                    model_data.to_csv(
                        output_handle,
                        index=False,
                        header=not output_header_written,
                    )
                    output_header_written = True

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_file, index=False)

    applied_lags = pd.DataFrame(applied_lag_rows)
    applied_lags = applied_lags.sort_values(
        [
            "prefecture_code",
            "target_variable",
            "frequency_band",
            "predictor_variable",
        ]
    )
    applied_lags.to_csv(applied_lags_file, index=False)

    print("\nSaved:")
    print(f"  {output_file}")
    print(f"  {summary_file}")
    print(f"  {applied_lags_file}")

    print("\nDataset groups:")
    print(summary["status"].value_counts())

    created = summary[
        summary["status"] == "created"
    ]

    if not created.empty:
        print("\nRows by split:")
        print(
            created[
                [
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                ]
            ]
            .sum()
        )

        print("\nMaximum applied lag distribution:")
        print(
            created[
                "maximum_applied_lag_days"
            ]
            .value_counts()
            .sort_index()
        )


if __name__ == "__main__":
    main()
