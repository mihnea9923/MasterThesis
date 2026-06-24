from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "modeling"
    / "frequency_lagged_dataset.csv.gz"
)

RIDGE_METRICS_FILE = (
    ROOT
    / "outputs"
    / "models"
    / "ridge"
    / "ridge_metrics_overall.csv"
)

OUT_DIR = ROOT / "outputs" / "models" / "hist_gradient_boosting"
MODEL_DIR = OUT_DIR / "serialized_models"

TARGET_VARIABLES = [
    "cases",
    "incidence_per_100k",
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

NUMERIC_FEATURES = [
    "t2m_lagged",
    "t2m_std_anom_lagged",
    "c2w_event_lagged",
    "w2c_event_lagged",
    "c2w_transition_intensity_lagged",
    "w2c_transition_intensity_lagged",
    "ah_lagged",
    "rh_lagged",
    "tp_lagged",
    "nw_wind_lagged",
]

CATEGORICAL_FEATURES = [
    "prefecture_code",
]

# A small, explicit search rather than a large expensive grid.
PARAMETER_CANDIDATES = [
    {
        "candidate_name": "shallow",
        "learning_rate": 0.05,
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
    {
        "candidate_name": "medium",
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
    {
        "candidate_name": "regularized",
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 10.0,
    },
]

# Hyperparameter search uses only a training subset for speed.
# The selected configuration is then fitted on the full training data.
MAX_TUNING_ROWS = 120_000

# Permutation importance is calculated on a validation subset.
MAX_IMPORTANCE_ROWS = 20_000
PERMUTATION_REPEATS = 5

RANDOM_STATE = 42


def ensure_output_directories() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_preprocessor() -> ColumnTransformer:
    prefecture_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "prefecture",
                prefecture_encoder,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def make_model(parameters: dict) -> HistGradientBoostingRegressor:
    # The final transformed column is the ordinal-encoded prefecture.
    prefecture_feature_index = len(NUMERIC_FEATURES)

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=parameters["learning_rate"],
        max_iter=parameters["max_iter"],
        max_leaf_nodes=parameters["max_leaf_nodes"],
        min_samples_leaf=parameters["min_samples_leaf"],
        l2_regularization=parameters["l2_regularization"],
        max_bins=255,
        categorical_features=[prefecture_feature_index],
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    target_std = float(np.std(y_true))

    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    if target_std > 0:
        nrmse_std = rmse / target_std
    else:
        nrmse_std = np.nan

    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        pearson_correlation = float(
            np.corrcoef(y_true, y_pred)[0, 1]
        )
    else:
        pearson_correlation = np.nan

    return {
        "n_observations": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "nrmse_std": nrmse_std,
        "r2": r2,
        "pearson_correlation": pearson_correlation,
    }


def append_overall_metrics(
    rows: list[dict],
    *,
    target_variable: str,
    frequency_band: str,
    split_name: str,
    selected_candidate: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    rows.append(
        {
            "target_variable": target_variable,
            "frequency_band": frequency_band,
            "model_name": "hist_gradient_boosting",
            "split": split_name,
            "selected_candidate": selected_candidate,
            **calculate_metrics(y_true, y_pred),
        }
    )


def append_prefecture_metrics(
    rows: list[dict],
    *,
    data: pd.DataFrame,
    predictions: np.ndarray,
    target_variable: str,
    frequency_band: str,
    split_name: str,
    selected_candidate: str,
) -> None:
    evaluation = data[
        [
            "prefecture_code",
            "target_component",
        ]
    ].copy()

    evaluation["prediction"] = predictions

    for prefecture_code, group in evaluation.groupby(
        "prefecture_code"
    ):
        rows.append(
            {
                "prefecture_code": prefecture_code,
                "target_variable": target_variable,
                "frequency_band": frequency_band,
                "model_name": "hist_gradient_boosting",
                "split": split_name,
                "selected_candidate": selected_candidate,
                **calculate_metrics(
                    group["target_component"].to_numpy(dtype=float),
                    group["prediction"].to_numpy(dtype=float),
                ),
            }
        )


def create_prediction_rows(
    data: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    result = data[
        [
            "date",
            "prefecture_code",
            "target_variable",
            "frequency_band",
            "split",
            "target_component",
        ]
    ].copy()

    result["model_name"] = "hist_gradient_boosting"
    result["prediction"] = predictions
    result["residual"] = (
        result["target_component"]
        - result["prediction"]
    )

    return result


def sample_rows(
    data: pd.DataFrame,
    maximum_rows: int,
) -> pd.DataFrame:
    if len(data) <= maximum_rows:
        return data

    return data.sample(
        n=maximum_rows,
        random_state=RANDOM_STATE,
    ).sort_index()


def clean_feature_name(name: str) -> str:
    return (
        name
        .removeprefix("numeric__")
        .removeprefix("prefecture__")
    )


def main() -> None:
    ensure_output_directories()

    print("Reading frequency-specific lagged dataset...")
    data = pd.read_csv(INPUT_FILE)

    data["date"] = pd.to_datetime(data["date"])
    data["prefecture_code"] = data["prefecture_code"].astype(str)
    data["target_variable"] = data["target_variable"].astype(str)
    data["frequency_band"] = data["frequency_band"].astype(str)
    data["split"] = data["split"].astype(str)

    required_columns = {
        "date",
        "prefecture_code",
        "target_variable",
        "frequency_band",
        "target_component",
        "split",
        *NUMERIC_FEATURES,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "The modeling dataset is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    metric_rows: list[dict] = []
    prefecture_metric_rows: list[dict] = []
    search_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    task_rows: list[dict] = []

    raw_feature_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    for target_variable in TARGET_VARIABLES:
        for frequency_band in FREQUENCY_BANDS:
            task = data[
                (data["target_variable"] == target_variable)
                & (data["frequency_band"] == frequency_band)
            ].copy()

            if task.empty:
                print(
                    f"Skipping {target_variable}/{frequency_band}: "
                    "no rows."
                )
                continue

            train_data = task[
                task["split"] == "train"
            ].copy()
            validation_data = task[
                task["split"] == "validation"
            ].copy()
            test_data = task[
                task["split"] == "test"
            ].copy()

            if (
                train_data.empty
                or validation_data.empty
                or test_data.empty
            ):
                print(
                    f"Skipping {target_variable}/{frequency_band}: "
                    "one or more splits are empty."
                )
                continue

            print(
                f"\nTask {target_variable}/{frequency_band}: "
                f"{len(train_data):,} train, "
                f"{len(validation_data):,} validation, "
                f"{len(test_data):,} test"
            )

            # ------------------------------------------------------
            # 1. Hyperparameter selection using training only.
            # ------------------------------------------------------
            tuning_data = sample_rows(
                train_data,
                MAX_TUNING_ROWS,
            )

            tuning_preprocessor = make_preprocessor()

            X_tuning = tuning_preprocessor.fit_transform(
                tuning_data[raw_feature_columns]
            )
            X_validation_for_search = (
                tuning_preprocessor.transform(
                    validation_data[raw_feature_columns]
                )
            )

            y_tuning = tuning_data[
                "target_component"
            ].to_numpy(dtype=float)
            y_validation = validation_data[
                "target_component"
            ].to_numpy(dtype=float)

            best_parameters: dict | None = None
            best_validation_rmse = np.inf

            for parameters in PARAMETER_CANDIDATES:
                candidate_model = make_model(parameters)
                candidate_model.fit(X_tuning, y_tuning)

                validation_prediction = (
                    candidate_model.predict(
                        X_validation_for_search
                    )
                )

                validation_rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_validation,
                            validation_prediction,
                        )
                    )
                )

                search_rows.append(
                    {
                        "target_variable": target_variable,
                        "frequency_band": frequency_band,
                        "tuning_rows": len(tuning_data),
                        **parameters,
                        "validation_rmse": validation_rmse,
                    }
                )

                print(
                    f"  {parameters['candidate_name']}: "
                    f"validation RMSE={validation_rmse:.6g}"
                )

                if validation_rmse < best_validation_rmse:
                    best_validation_rmse = validation_rmse
                    best_parameters = parameters.copy()

            if best_parameters is None:
                raise RuntimeError(
                    f"Parameter selection failed for "
                    f"{target_variable}/{frequency_band}."
                )

            selected_candidate = best_parameters[
                "candidate_name"
            ]

            print(
                f"  Selected: {selected_candidate}, "
                f"RMSE={best_validation_rmse:.6g}"
            )

            # ------------------------------------------------------
            # 2. Fit the selected model on the full training period.
            # ------------------------------------------------------
            training_preprocessor = make_preprocessor()

            X_train = training_preprocessor.fit_transform(
                train_data[raw_feature_columns]
            )
            X_validation = training_preprocessor.transform(
                validation_data[raw_feature_columns]
            )

            y_train = train_data[
                "target_component"
            ].to_numpy(dtype=float)

            training_model = make_model(best_parameters)
            training_model.fit(X_train, y_train)

            train_prediction = training_model.predict(X_train)
            validation_prediction = training_model.predict(
                X_validation
            )

            append_overall_metrics(
                metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                split_name="train",
                selected_candidate=selected_candidate,
                y_true=y_train,
                y_pred=train_prediction,
            )
            append_overall_metrics(
                metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                split_name="validation",
                selected_candidate=selected_candidate,
                y_true=y_validation,
                y_pred=validation_prediction,
            )

            append_prefecture_metrics(
                prefecture_metric_rows,
                data=train_data,
                predictions=train_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                split_name="train",
                selected_candidate=selected_candidate,
            )
            append_prefecture_metrics(
                prefecture_metric_rows,
                data=validation_data,
                predictions=validation_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                split_name="validation",
                selected_candidate=selected_candidate,
            )

            prediction_parts.append(
                create_prediction_rows(
                    validation_data,
                    validation_prediction,
                )
            )

            # ------------------------------------------------------
            # 3. Permutation importance on validation data.
            # ------------------------------------------------------
            importance_data = sample_rows(
                validation_data,
                MAX_IMPORTANCE_ROWS,
            )

            X_importance = training_preprocessor.transform(
                importance_data[raw_feature_columns]
            )
            y_importance = importance_data[
                "target_component"
            ].to_numpy(dtype=float)

            importance = permutation_importance(
                training_model,
                X_importance,
                y_importance,
                scoring="neg_root_mean_squared_error",
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            transformed_feature_names = (
                training_preprocessor
                .get_feature_names_out()
            )

            for feature_name, mean_value, std_value in zip(
                transformed_feature_names,
                importance.importances_mean,
                importance.importances_std,
            ):
                importance_rows.append(
                    {
                        "target_variable": target_variable,
                        "frequency_band": frequency_band,
                        "selected_candidate": selected_candidate,
                        "feature_name": clean_feature_name(
                            str(feature_name)
                        ),
                        "importance_mean_rmse_increase":
                            float(mean_value),
                        "importance_std":
                            float(std_value),
                        "importance_rows":
                            len(importance_data),
                    }
                )

            # ------------------------------------------------------
            # 4. Refit on train + validation and evaluate test once.
            # ------------------------------------------------------
            train_validation_data = pd.concat(
                [train_data, validation_data],
                ignore_index=True,
            )

            final_preprocessor = make_preprocessor()

            X_train_validation = (
                final_preprocessor.fit_transform(
                    train_validation_data[
                        raw_feature_columns
                    ]
                )
            )
            X_test = final_preprocessor.transform(
                test_data[raw_feature_columns]
            )

            y_train_validation = train_validation_data[
                "target_component"
            ].to_numpy(dtype=float)
            y_test = test_data[
                "target_component"
            ].to_numpy(dtype=float)

            final_model = make_model(best_parameters)
            final_model.fit(
                X_train_validation,
                y_train_validation,
            )

            test_prediction = final_model.predict(X_test)

            append_overall_metrics(
                metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                split_name="test",
                selected_candidate=selected_candidate,
                y_true=y_test,
                y_pred=test_prediction,
            )

            append_prefecture_metrics(
                prefecture_metric_rows,
                data=test_data,
                predictions=test_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                split_name="test",
                selected_candidate=selected_candidate,
            )

            prediction_parts.append(
                create_prediction_rows(
                    test_data,
                    test_prediction,
                )
            )

            model_filename = (
                f"hgb_{target_variable}_{frequency_band}.joblib"
            )
            model_path = MODEL_DIR / model_filename

            joblib.dump(
                {
                    "preprocessor": final_preprocessor,
                    "model": final_model,
                    "target_variable": target_variable,
                    "frequency_band": frequency_band,
                    "selected_parameters": best_parameters,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "train_period": "2005-01-01/2016-12-31",
                    "validation_period": "2017-01-01/2017-12-31",
                    "test_period": "2018-01-01/2019-12-31",
                },
                model_path,
            )

            task_rows.append(
                {
                    "target_variable": target_variable,
                    "frequency_band": frequency_band,
                    "selected_candidate": selected_candidate,
                    **{
                        key: value
                        for key, value in best_parameters.items()
                        if key != "candidate_name"
                    },
                    "train_rows": len(train_data),
                    "validation_rows": len(validation_data),
                    "test_rows": len(test_data),
                    "n_prefectures":
                        task["prefecture_code"].nunique(),
                    "model_file": str(
                        model_path.relative_to(ROOT)
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    prefecture_metrics = pd.DataFrame(
        prefecture_metric_rows
    )
    search_results = pd.DataFrame(search_rows)
    importance_results = pd.DataFrame(importance_rows)
    task_summary = pd.DataFrame(task_rows)
    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    metrics_file = OUT_DIR / "hgb_metrics_overall.csv"
    prefecture_metrics_file = (
        OUT_DIR
        / "hgb_metrics_by_prefecture.csv"
    )
    search_file = OUT_DIR / "hgb_parameter_search.csv"
    importance_file = OUT_DIR / "hgb_permutation_importance.csv"
    predictions_file = OUT_DIR / "hgb_predictions.csv.gz"
    task_summary_file = OUT_DIR / "hgb_task_summary.csv"
    configuration_file = OUT_DIR / "hgb_configuration.json"
    comparison_file = OUT_DIR / "ridge_vs_hgb_test_metrics.csv"

    metrics.to_csv(metrics_file, index=False)
    prefecture_metrics.to_csv(
        prefecture_metrics_file,
        index=False,
    )
    search_results.to_csv(search_file, index=False)
    importance_results.to_csv(
        importance_file,
        index=False,
    )
    predictions.to_csv(
        predictions_file,
        index=False,
        compression="gzip",
    )
    task_summary.to_csv(task_summary_file, index=False)

    configuration = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "model": "HistGradientBoostingRegressor",
        "targets": TARGET_VARIABLES,
        "frequency_bands": FREQUENCY_BANDS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_feature": "prefecture_code",
        "parameter_candidates": PARAMETER_CANDIDATES,
        "maximum_tuning_rows": MAX_TUNING_ROWS,
        "maximum_importance_rows": MAX_IMPORTANCE_ROWS,
        "permutation_repeats": PERMUTATION_REPEATS,
        "selection_metric": "validation RMSE",
        "final_fit": "train plus validation",
        "test_period": "2018-01-01 through 2019-12-31",
    }

    configuration_file.write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    if RIDGE_METRICS_FILE.exists():
        ridge_metrics = pd.read_csv(RIDGE_METRICS_FILE)

        ridge_test = ridge_metrics[
            ridge_metrics["split"] == "test"
        ].copy()

        hgb_test = metrics[
            metrics["split"] == "test"
        ].copy()

        common_columns = [
            "target_variable",
            "frequency_band",
            "model_name",
            "split",
            "n_observations",
            "mae",
            "rmse",
            "nrmse_std",
            "r2",
            "pearson_correlation",
        ]

        comparison = pd.concat(
            [
                ridge_test[common_columns],
                hgb_test[common_columns],
            ],
            ignore_index=True,
        ).drop_duplicates()

        comparison.to_csv(
            comparison_file,
            index=False,
        )

    print("\nSaved:")
    for path in [
        metrics_file,
        prefecture_metrics_file,
        search_file,
        importance_file,
        predictions_file,
        task_summary_file,
        configuration_file,
    ]:
        print(f"  {path}")

    if comparison_file.exists():
        print(f"  {comparison_file}")

    print(f"  {MODEL_DIR}/")

    print("\nHGB test metrics:")
    test_metrics = metrics[
        metrics["split"] == "test"
    ].sort_values(
        [
            "target_variable",
            "frequency_band",
        ]
    )

    print(
        test_metrics[
            [
                "target_variable",
                "frequency_band",
                "selected_candidate",
                "mae",
                "rmse",
                "nrmse_std",
                "r2",
                "pearson_correlation",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
