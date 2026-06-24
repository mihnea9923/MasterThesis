from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "modeling"
    / "frequency_lagged_dataset.csv.gz"
)

OUT_DIR = ROOT / "outputs" / "models" / "ridge"
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

ALPHA_CANDIDATES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

RANDOM_STATE = 42


def ensure_output_directories() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Support both recent and older scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "prefecture",
                make_one_hot_encoder(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    target_std = float(np.std(y_true))

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


def create_prefecture_mean_baseline(
    train_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> np.ndarray:
    """
    Baseline prediction: each prefecture's mean target value in training.
    """
    prefecture_means = (
        train_data
        .groupby("prefecture_code")["target_component"]
        .mean()
    )

    global_mean = float(train_data["target_component"].mean())

    predictions = (
        evaluation_data["prefecture_code"]
        .map(prefecture_means)
        .fillna(global_mean)
        .to_numpy(dtype=float)
    )

    return predictions


def fit_ridge(
    transformed_features: sparse.spmatrix | np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> Ridge:
    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="lsqr",
        max_iter=5000,
        tol=1e-6,
        random_state=RANDOM_STATE,
    )

    model.fit(transformed_features, target)
    return model


def append_metric_rows(
    rows: list[dict],
    *,
    target_variable: str,
    frequency_band: str,
    model_name: str,
    split_name: str,
    alpha: float | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    metrics = calculate_metrics(y_true, y_pred)

    rows.append(
        {
            "target_variable": target_variable,
            "frequency_band": frequency_band,
            "model_name": model_name,
            "split": split_name,
            "selected_alpha": alpha,
            **metrics,
        }
    )


def append_prefecture_metric_rows(
    rows: list[dict],
    *,
    data: pd.DataFrame,
    predictions: np.ndarray,
    target_variable: str,
    frequency_band: str,
    model_name: str,
    split_name: str,
    alpha: float | None,
) -> None:
    evaluation = data[
        [
            "prefecture_code",
            "target_component",
        ]
    ].copy()

    evaluation["prediction"] = predictions

    for prefecture_code, prefecture_data in evaluation.groupby(
        "prefecture_code"
    ):
        metrics = calculate_metrics(
            prefecture_data["target_component"].to_numpy(dtype=float),
            prefecture_data["prediction"].to_numpy(dtype=float),
        )

        rows.append(
            {
                "prefecture_code": prefecture_code,
                "target_variable": target_variable,
                "frequency_band": frequency_band,
                "model_name": model_name,
                "split": split_name,
                "selected_alpha": alpha,
                **metrics,
            }
        )


def build_prediction_rows(
    data: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model_name: str,
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

    result["model_name"] = model_name
    result["prediction"] = predictions
    result["residual"] = (
        result["target_component"] - result["prediction"]
    )

    return result


def clean_feature_name(feature_name: str) -> tuple[str, str]:
    if feature_name.startswith("numeric__"):
        return (
            feature_name.removeprefix("numeric__"),
            "environmental_predictor",
        )

    if feature_name.startswith("prefecture__"):
        return (
            feature_name.removeprefix("prefecture__"),
            "prefecture_fixed_effect",
        )

    return feature_name, "other"


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

    overall_metric_rows: list[dict] = []
    prefecture_metric_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    alpha_search_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    task_summary_rows: list[dict] = []

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

            train_data = task[task["split"] == "train"].copy()
            validation_data = task[
                task["split"] == "validation"
            ].copy()
            test_data = task[task["split"] == "test"].copy()

            if (
                train_data.empty
                or validation_data.empty
                or test_data.empty
            ):
                print(
                    f"Skipping {target_variable}/{frequency_band}: "
                    "one or more temporal splits are empty."
                )
                continue

            print(
                f"\nTraining {target_variable}/{frequency_band}: "
                f"{len(train_data):,} train, "
                f"{len(validation_data):,} validation, "
                f"{len(test_data):,} test rows"
            )

            X_train_raw = train_data[
                NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ]
            X_validation_raw = validation_data[
                NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ]
            X_test_raw = test_data[
                NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ]

            y_train = train_data[
                "target_component"
            ].to_numpy(dtype=float)
            y_validation = validation_data[
                "target_component"
            ].to_numpy(dtype=float)
            y_test = test_data[
                "target_component"
            ].to_numpy(dtype=float)

            # ----------------------------------------------------------
            # 1. Fit preprocessing on training only.
            # ----------------------------------------------------------
            training_preprocessor = make_preprocessor()

            X_train = training_preprocessor.fit_transform(
                X_train_raw
            )
            X_validation = training_preprocessor.transform(
                X_validation_raw
            )

            # ----------------------------------------------------------
            # 2. Select Ridge alpha using validation RMSE.
            # ----------------------------------------------------------
            best_alpha: float | None = None
            best_validation_rmse = np.inf
            best_training_model: Ridge | None = None

            for alpha in ALPHA_CANDIDATES:
                candidate_model = fit_ridge(
                    X_train,
                    y_train,
                    alpha,
                )

                validation_prediction = candidate_model.predict(
                    X_validation
                )

                validation_rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_validation,
                            validation_prediction,
                        )
                    )
                )

                alpha_search_rows.append(
                    {
                        "target_variable": target_variable,
                        "frequency_band": frequency_band,
                        "alpha": alpha,
                        "validation_rmse": validation_rmse,
                    }
                )

                if validation_rmse < best_validation_rmse:
                    best_validation_rmse = validation_rmse
                    best_alpha = alpha
                    best_training_model = candidate_model

            if best_alpha is None or best_training_model is None:
                raise RuntimeError(
                    f"Alpha selection failed for "
                    f"{target_variable}/{frequency_band}."
                )

            print(
                f"Selected alpha={best_alpha:g}, "
                f"validation RMSE={best_validation_rmse:.6g}"
            )

            # ----------------------------------------------------------
            # 3. Evaluate the selected train-only model on train/valid.
            # ----------------------------------------------------------
            train_prediction = best_training_model.predict(X_train)
            validation_prediction = best_training_model.predict(
                X_validation
            )

            append_metric_rows(
                overall_metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="ridge",
                split_name="train",
                alpha=best_alpha,
                y_true=y_train,
                y_pred=train_prediction,
            )

            append_metric_rows(
                overall_metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="ridge",
                split_name="validation",
                alpha=best_alpha,
                y_true=y_validation,
                y_pred=validation_prediction,
            )

            append_prefecture_metric_rows(
                prefecture_metric_rows,
                data=train_data,
                predictions=train_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="ridge",
                split_name="train",
                alpha=best_alpha,
            )

            append_prefecture_metric_rows(
                prefecture_metric_rows,
                data=validation_data,
                predictions=validation_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="ridge",
                split_name="validation",
                alpha=best_alpha,
            )

            validation_baseline_prediction = (
                create_prefecture_mean_baseline(
                    train_data,
                    validation_data,
                )
            )

            append_metric_rows(
                overall_metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="prefecture_mean_baseline",
                split_name="validation",
                alpha=None,
                y_true=y_validation,
                y_pred=validation_baseline_prediction,
            )

            append_prefecture_metric_rows(
                prefecture_metric_rows,
                data=validation_data,
                predictions=validation_baseline_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="prefecture_mean_baseline",
                split_name="validation",
                alpha=None,
            )

            prediction_parts.append(
                build_prediction_rows(
                    validation_data,
                    validation_prediction,
                    model_name="ridge",
                )
            )

            # ----------------------------------------------------------
            # 4. Refit preprocessing and Ridge on train + validation.
            # ----------------------------------------------------------
            train_validation_data = pd.concat(
                [train_data, validation_data],
                ignore_index=True,
            )

            X_train_validation_raw = train_validation_data[
                NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ]
            y_train_validation = train_validation_data[
                "target_component"
            ].to_numpy(dtype=float)

            final_preprocessor = make_preprocessor()

            X_train_validation = (
                final_preprocessor.fit_transform(
                    X_train_validation_raw
                )
            )
            X_test = final_preprocessor.transform(X_test_raw)

            final_model = fit_ridge(
                X_train_validation,
                y_train_validation,
                best_alpha,
            )

            test_prediction = final_model.predict(X_test)

            append_metric_rows(
                overall_metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="ridge",
                split_name="test",
                alpha=best_alpha,
                y_true=y_test,
                y_pred=test_prediction,
            )

            append_prefecture_metric_rows(
                prefecture_metric_rows,
                data=test_data,
                predictions=test_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="ridge",
                split_name="test",
                alpha=best_alpha,
            )

            test_baseline_prediction = create_prefecture_mean_baseline(
                train_validation_data,
                test_data,
            )

            append_metric_rows(
                overall_metric_rows,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="prefecture_mean_baseline",
                split_name="test",
                alpha=None,
                y_true=y_test,
                y_pred=test_baseline_prediction,
            )

            append_prefecture_metric_rows(
                prefecture_metric_rows,
                data=test_data,
                predictions=test_baseline_prediction,
                target_variable=target_variable,
                frequency_band=frequency_band,
                model_name="prefecture_mean_baseline",
                split_name="test",
                alpha=None,
            )

            prediction_parts.append(
                build_prediction_rows(
                    test_data,
                    test_prediction,
                    model_name="ridge",
                )
            )

            # ----------------------------------------------------------
            # 5. Save coefficients and the final fitted model bundle.
            # ----------------------------------------------------------
            transformed_feature_names = (
                final_preprocessor.get_feature_names_out()
            )

            for feature_name, coefficient in zip(
                transformed_feature_names,
                final_model.coef_,
            ):
                clean_name, feature_type = clean_feature_name(
                    str(feature_name)
                )

                coefficient_rows.append(
                    {
                        "target_variable": target_variable,
                        "frequency_band": frequency_band,
                        "selected_alpha": best_alpha,
                        "feature_name": clean_name,
                        "feature_type": feature_type,
                        "coefficient": float(coefficient),
                        "absolute_coefficient": float(
                            abs(coefficient)
                        ),
                    }
                )

            model_filename = (
                f"ridge_{target_variable}_{frequency_band}.joblib"
            )
            model_path = MODEL_DIR / model_filename

            joblib.dump(
                {
                    "preprocessor": final_preprocessor,
                    "model": final_model,
                    "target_variable": target_variable,
                    "frequency_band": frequency_band,
                    "selected_alpha": best_alpha,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "train_end": "2016-12-31",
                    "validation_period": "2017-01-01/2017-12-31",
                    "test_period": "2018-01-01/2019-12-31",
                },
                model_path,
            )

            task_summary_rows.append(
                {
                    "target_variable": target_variable,
                    "frequency_band": frequency_band,
                    "selected_alpha": best_alpha,
                    "train_rows": len(train_data),
                    "validation_rows": len(validation_data),
                    "test_rows": len(test_data),
                    "n_prefectures_train":
                        train_data["prefecture_code"].nunique(),
                    "n_prefectures_validation":
                        validation_data["prefecture_code"].nunique(),
                    "n_prefectures_test":
                        test_data["prefecture_code"].nunique(),
                    "model_file": str(model_path.relative_to(ROOT)),
                }
            )

    overall_metrics = pd.DataFrame(overall_metric_rows)
    prefecture_metrics = pd.DataFrame(prefecture_metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    alpha_search = pd.DataFrame(alpha_search_rows)
    task_summary = pd.DataFrame(task_summary_rows)

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    overall_metrics_file = OUT_DIR / "ridge_metrics_overall.csv"
    prefecture_metrics_file = (
        OUT_DIR
        / "ridge_metrics_by_prefecture.csv"
    )
    coefficients_file = OUT_DIR / "ridge_coefficients.csv"
    alpha_search_file = OUT_DIR / "ridge_alpha_search.csv"
    predictions_file = OUT_DIR / "ridge_predictions.csv.gz"
    task_summary_file = OUT_DIR / "ridge_task_summary.csv"
    configuration_file = OUT_DIR / "ridge_configuration.json"

    overall_metrics.to_csv(
        overall_metrics_file,
        index=False,
    )
    prefecture_metrics.to_csv(
        prefecture_metrics_file,
        index=False,
    )
    coefficients.to_csv(
        coefficients_file,
        index=False,
    )
    alpha_search.to_csv(
        alpha_search_file,
        index=False,
    )
    predictions.to_csv(
        predictions_file,
        index=False,
        compression="gzip",
    )
    task_summary.to_csv(
        task_summary_file,
        index=False,
    )

    configuration = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "targets": TARGET_VARIABLES,
        "frequency_bands": FREQUENCY_BANDS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "alpha_candidates": ALPHA_CANDIDATES,
        "model": "Ridge",
        "prefecture_handling": "one-hot fixed effects",
        "alpha_selection_metric": "validation RMSE",
        "final_fit": "train plus validation",
        "test_period": "2018-01-01 through 2019-12-31",
    }

    configuration_file.write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    for path in [
        overall_metrics_file,
        prefecture_metrics_file,
        coefficients_file,
        alpha_search_file,
        predictions_file,
        task_summary_file,
        configuration_file,
    ]:
        print(f"  {path}")

    print(f"  {MODEL_DIR}/")

    print("\nTest metrics:")
    test_metrics = overall_metrics[
        (overall_metrics["split"] == "test")
        & (overall_metrics["model_name"] == "ridge")
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
                "selected_alpha",
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
