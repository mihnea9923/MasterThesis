from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from analysis_plot import (
    plot_contributions_peak_vs_normal,
    plot_fit,
    plot_flipflop_vs_cases,
    plot_flipflop_lag,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

@dataclass
class Config:
    out_dir: Path = Path("outputs")

    weather_file: str = "daily_meteo_prefs_2005-2019.csv"
    ohca_file: str = "cardio_clean.csv"
    population_file: str = "population_ts.csv"

    exclude_prefectures: tuple[str, ...] = ("JP-12",)  # Alejandro's suggestion to remove this prefecture

    temp_col: str = "t2m"
    max_gap: int = 5
    rolling_temp_window: int = 5
    climatology_window: int = 31
    std_threshold: float = 1.0
    detrend_order: int = 3

    # Lag days to create for temperature and flip-flop exposure variables.
    # lag0 = same day, lag1 = previous day, etc.
    lags: tuple[int, ...] = (0, 1, 2, 3)


CFG = Config()

def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", errors="coerce").dt.normalize()


# We need to detrend because the standardized anomaly is computed relative to a climatology and if there's a long-term trend in the data,
# it could bias the anomaly and thus the flip-flop detection. And we already know there is trend of global warming in the data.
# Detrending removes this long-term trend allowing us to focus on the short-term sudden changes that define the flip-flops 
def detrend_poly_by_prefecture(df, value_col, order=3):
    values = df[value_col].to_numpy(dtype=float)
    time_index = np.arange(len(values), dtype=float)

    valid_mask = np.isfinite(values)
    
    trend = np.polyval(
        np.polyfit(time_index[valid_mask], values[valid_mask], deg=order),
        time_index
    )

    detrended = values - trend
    return pd.Series(detrended, index=df.index)

# To better match the flip-flop events we apply a mean to the temperature data over a 1 month window centered on each day
def circular_rolling_climatology(doy_stats: pd.DataFrame, window: int = 31) -> pd.DataFrame:
    all_doys = pd.Index(range(1, 367), name="doy")
    base = doy_stats.reindex(all_doys)

    # We need to make the data like a circular list to apply the rolling mean correctly for begining and end of years
    extended = pd.concat([base, base, base], axis=0, ignore_index=True)
    rolled = extended.rolling(window=window, center=True, min_periods=max(5, window // 3)).mean()
    # Since I concatenated the data 3 times the middle part was surronded by neighbors on both edges.
    middle = rolled.iloc[366:732].copy()
    middle.index = all_doys
    return middle


def cool2warm_whiplash(flag: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Return [start, end] indices for cold-to-warm events: -1 followed by +1."""
    events: list[list[int]] = []
    n = len(flag)

    for i in range(n):
        if flag[i] == -1:
            for j in range(i + 1, min(i + max_gap + 1, n)):
                if flag[j] == 1:
                    events.append([i, j])
                    break

    if not events:
        return np.empty((0, 2), dtype=int)

    events_arr = np.asarray(events, dtype=int)

    # Remove repeated events that end on the same warm day because we want to count only one flip-flop per warm day
    if len(events_arr) > 1:
        repeated_idx = np.where(events_arr[:-1, 1] - events_arr[1:, 1] == 0)[0]
        events_arr = np.delete(events_arr, repeated_idx, axis=0)

    return events_arr


def warm2cool_whiplash(flag: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Return [start, end] indices for warm-to-cold events: +1 followed by -1."""
    events: list[list[int]] = []
    n = len(flag)

    for i in range(n):
        if flag[i] == 1:
            for j in range(i + 1, min(i + max_gap + 1, n)):
                if flag[j] == -1:
                    events.append([i, j])
                    break

    if not events:
        return np.empty((0, 2), dtype=int)

    events_arr = np.asarray(events, dtype=int)

    if len(events_arr) > 1:
        repeated_idx = np.where(events_arr[:-1, 1] - events_arr[1:, 1] == 0)[0]
        events_arr = np.delete(events_arr, repeated_idx, axis=0)

    return events_arr


def add_flipflops_one_prefecture(
    df: pd.DataFrame,
    temp_col: str,
    max_gap: int,
    rolling_temp_window: int,
    climatology_window: int,
    std_threshold: float,
    detrend_order: int,
) -> pd.DataFrame:
   
    df = df.sort_values("date").copy().reset_index(drop=True)

    df[f"{temp_col}_detrended"] = detrend_poly_by_prefecture(df, temp_col, order=detrend_order)

    # Sometimes there can be random temperature changes from one day to another. A flip-flop is valid only if there is a real change 
    # in temperature over multiple days, so that's why I apply a rolling mean
    df[f"{temp_col}_roll5_detrended"] = (
        df[f"{temp_col}_detrended"]
        .rolling(window=rolling_temp_window, center=True, min_periods=3)
        .mean()
    )

    # For each day of the year and for each prefecture, I compute the average and std across all years.
    # Std in this case tells us how much variability is normal for that day of the year.
    # This gives me a baseline to determine if a day is unusual compared to the "normal" weather for that day of yar
    # This is called a "climatology"
    df["doy"] = df["date"].dt.dayofyear
    doy_stats = (
        df.groupby("doy")[f"{temp_col}_roll5_detrended"]
        .agg(clim_mean="mean", clim_std="std")
    )
    clim = circular_rolling_climatology(doy_stats, window=climatology_window)
    df = df.merge(clim, left_on="doy", right_index=True, how="left")

    # Standardized anomaly.
    df[f"{temp_col}_std_anom"] = (
        (df[f"{temp_col}_roll5_detrended"] - df["clim_mean"]) / df["clim_std"]
    )

    # At this point we flag each day as hot/cold compared to the normal variability of thay day of the year.
    # More precisely, over that period of the year
    df["temp_extreme_flag"] = 0
    df.loc[df[f"{temp_col}_std_anom"] > std_threshold, "temp_extreme_flag"] = 1
    df.loc[df[f"{temp_col}_std_anom"] < -std_threshold, "temp_extreme_flag"] = -1

    # Initialize event columns.
    event_cols = [
        "c2w_event", "w2c_event",
        "c2w_transition_duration", "w2c_transition_duration",
        "c2w_transition_intensity", "w2c_transition_intensity",
        "c2w_start_date", "c2w_end_date", "w2c_start_date", "w2c_end_date",
    ]
    
    # Only those 2 events will be described as booleans, the rest is numeric
    for col in event_cols:
        df[col] = 0 if col.endswith("event") else np.nan

    flag = df["temp_extreme_flag"].to_numpy()

    c2w = cool2warm_whiplash(flag, max_gap=max_gap)
    w2c = warm2cool_whiplash(flag, max_gap=max_gap)

    z = df[f"{temp_col}_std_anom"].to_numpy(dtype=float)
    for start, end in c2w:
        # We mark the end day of the flip-flop event with 1 and compute duration and intensity
        df.loc[end, "c2w_event"] = 1
        df.loc[end, "c2w_transition_duration"] = end - start
        df.loc[end, "c2w_transition_intensity"] = abs(z[end] - z[start])
        df.loc[end, "c2w_start_date"] = df.loc[start, "date"]
        df.loc[end, "c2w_end_date"] = df.loc[end, "date"]

    for start, end in w2c:
        df.loc[end, "w2c_event"] = 1
        df.loc[end, "w2c_transition_duration"] = end - start
        df.loc[end, "w2c_transition_intensity"] = abs(z[end] - z[start])
        df.loc[end, "w2c_start_date"] = df.loc[start, "date"]
        df.loc[end, "w2c_end_date"] = df.loc[end, "date"]

    return df


def add_lags(df: pd.DataFrame, variables: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["prefecture_code", "date"]).copy()
    for var in variables:
        for lag in lags:
            if lag == 0:
                out[f"{var}_lag0"] = out[var]
            else:
                out[f"{var}_lag{lag}"] = out.groupby("prefecture_code")[var].shift(lag)
    return out


def aggregate_ohca_daily(cardio: pd.DataFrame) -> pd.DataFrame:
    return (
        cardio.groupby(["prefecture_code", "date"])
        .size()
        .reset_index(name="cases")
    )


def aggregate_population(pop: pd.DataFrame) -> pd.DataFrame:
    return (
        pop.groupby(["prefecture_code", "year"], as_index=False)["population"]
        .sum()
    )


def fit_poisson_model(formula: str, data: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    return smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Poisson(),
        offset=data["log_population"],
    ).fit(cov_type="HC0")


def model_summary_row(name: str, model, n: int) -> dict:
    return {
        "model": name,
        "n_obs": n,
        "aic": model.aic,
        "bic_llf": model.bic_llf,
        "deviance": model.deviance,
        "df_resid": model.df_resid,
    }


def extract_coefficients(model_name: str, model) -> pd.DataFrame:
    ci = model.conf_int()
    out = pd.DataFrame({
        "model": model_name,
        "term": model.params.index,
        "estimate_log_rr": model.params.values,
        "rr": np.exp(model.params.values),
        "ci_low_rr": np.exp(ci[0].values),
        "ci_high_rr": np.exp(ci[1].values),
        "p_value": model.pvalues.values,
    })
    return out



def main(cfg: Config = CFG) -> None:
    ensure_out_dir(cfg.out_dir)

    meteo = pd.read_csv(cfg.weather_file)
    meteo["date"] = parse_date(meteo["date"])
    meteo = meteo.dropna(subset=["date", "prefecture_code", cfg.temp_col]).copy()
    meteo = meteo[~meteo["prefecture_code"].isin(cfg.exclude_prefectures)].copy()
    meteo = meteo.sort_values(["prefecture_code", "date"])

    cardio = pd.read_csv(cfg.ohca_file)
    cardio["date"] = parse_date(cardio["date"])
    cardio = cardio.dropna(subset=["date", "prefecture_code"]).copy()
    cardio = cardio[~cardio["prefecture_code"].isin(cfg.exclude_prefectures)].copy()

    pop = pd.read_csv(cfg.population_file)
    pop = pop[~pop["prefecture_code"].isin(cfg.exclude_prefectures)].copy()

    flipflops = (
        meteo.groupby("prefecture_code", group_keys=False)
        .apply(
            add_flipflops_one_prefecture,
            temp_col=cfg.temp_col,
            max_gap=cfg.max_gap,
            rolling_temp_window=cfg.rolling_temp_window,
            climatology_window=cfg.climatology_window,
            std_threshold=cfg.std_threshold,
            detrend_order=cfg.detrend_order,
        )
        .reset_index(drop=True)
    )

    flipflops_file = cfg.out_dir / "temperature_flipflops_prefectures.csv"
    flipflops.to_csv(flipflops_file, index=False)
    print(f"Saved: {flipflops_file}")

    daily_cases = aggregate_ohca_daily(cardio)
    pop_total = aggregate_population(pop)

    # We combine the flip-flop data with the OHCA cases
    analysis = flipflops.merge(daily_cases, on=["prefecture_code", "date"], how="left")
    analysis["cases"] = analysis["cases"].fillna(0).astype(int)
    analysis["year"] = analysis["date"].dt.year
    analysis["month"] = analysis["date"].dt.month
    analysis["dow"] = analysis["date"].dt.dayofweek
    analysis["time_index"] = (analysis["date"] - analysis["date"].min()).dt.days

    # Now add the population data to compute incidence rates and use as offset in the models. I also merge on year because population changes
    analysis = analysis.merge(pop_total, on=["prefecture_code", "year"], how="left")
    analysis = analysis.dropna(subset=["population"]).copy()
    analysis["log_population"] = np.log(analysis["population"])
    # I compute this incidence for descriptive purposes, but I'm not using this for the model
    analysis["incidence_per_100k"] = analysis["cases"] / analysis["population"] * 100000

    # Fill event intensity/duration missing values with 0 for modeling.
    for col in [
        "c2w_transition_duration", "w2c_transition_duration",
        "c2w_transition_intensity", "w2c_transition_intensity",
    ]:
        analysis[col] = analysis[col].fillna(0)

    # Add lagged exposure variables. Thos variable can have potential delayed effect so that is why I chose them
    lag_vars = [
        cfg.temp_col,
        f"{cfg.temp_col}_std_anom",
        "c2w_event", "w2c_event",
        "c2w_transition_intensity", "w2c_transition_intensity",
        "rh", "ah", "tp", "nw_wind",
    ]
    lag_vars = [v for v in lag_vars if v in analysis.columns]
    
    analysis = add_lags(analysis, lag_vars, cfg.lags)
    # We need to standardize the time index and temperature to make the model coefficients more comparable. The time series
    # is quite long so it will end-up artificially dominating its contribution in the plot below simply because it has a large value 
    scale_cols = [
    "time_index",
    "t2m_lag0", "t2m_lag1", "t2m_lag2", "t2m_lag3",
    "rh_lag0", "tp_lag0", "nw_wind_lag0"]

    scale_cols = [c for c in scale_cols if c in analysis.columns]
    scaler = StandardScaler()
    analysis[scale_cols] = scaler.fit_transform(analysis[scale_cols])
    
    analysis_file = cfg.out_dir / "ohca_temperature_flipflop_analysis_dataset.csv"
    analysis.to_csv(analysis_file, index=False)
    print(f"Saved: {analysis_file}")
    
    model_df = analysis.dropna(subset=["population", cfg.temp_col, f"{cfg.temp_col}_std_anom_lag0"]).copy()

    weather_controls = []
    for v in ["rh_lag0", "tp_lag0", "nw_wind_lag0"]:
        if v in model_df.columns:
            weather_controls.append(v)

    # We want to eliminate the influence of these fields that could confound the relationship between temperature/flip-flops and OHCA
    controls = "C(prefecture_code) + C(month) + C(dow) + time_index + I(time_index ** 2)"
    if weather_controls:
        controls += " + " + " + ".join(weather_controls)

    formulas = {
        "raw_temperature": f"cases ~ {cfg.temp_col}_lag0 + {controls}",
        "flipflops": (
            f"cases ~ c2w_event_lag0 + w2c_event_lag0 + {cfg.temp_col}_lag0 "
            "+ c2w_transition_intensity_lag0 + w2c_transition_intensity_lag0 "
            f"+ {controls}"
        ),
        "lags_raw_plus_flipflops": (
            f"cases ~ {cfg.temp_col}_lag0 + {cfg.temp_col}_lag1 + {cfg.temp_col}_lag2 + {cfg.temp_col}_lag3 "
            "+ c2w_event_lag0 + c2w_event_lag1 + c2w_event_lag2 + c2w_event_lag3 "
            "+ w2c_event_lag0 + w2c_event_lag1 + w2c_event_lag2 + w2c_event_lag3 "
            "+ c2w_transition_intensity_lag0 + w2c_transition_intensity_lag0 "
            f"+ {controls}"
        ),
    }

    models = {}
    rows = []
    coefs = []

    for name, formula in formulas.items():
        model = fit_poisson_model(formula, model_df)
        models[name] = model
        rows.append(model_summary_row(name, model, int(model.nobs)))
        coefs.append(extract_coefficients(name, model))
        print(f"{name:25s} AIC = {model.aic:,.2f}")

    for name, model in models.items():
        model_df[f"pred_{name}"] = model.predict(model_df) * model_df["population"]    
    for name in models:
        model_df[f"resid_{name}"] = model_df["cases"] - model_df[f"pred_{name}"]
    
    comparison = pd.DataFrame(rows).sort_values("aic")
    comparison["delta_aic"] = comparison["aic"] - comparison["aic"].min()
    comparison_file = cfg.out_dir / "model_comparison_results.csv"
    comparison.to_csv(comparison_file, index=False)

    coef_df = pd.concat(coefs, ignore_index=True)
    coef_file = cfg.out_dir / "model_coefficients.csv"
    coef_df.to_csv(coef_file, index=False)

    print("\nBest model by AIC:")
    print(comparison.head(1).to_string(index=False))
    
    model_df["any_flip_lag0"] = ((model_df["c2w_event_lag0"] == 1) | (model_df["w2c_event_lag0"] == 1)).astype(int)
    peak_threshold = model_df["cases"].quantile(0.90)
    model_df["is_peak"] = (model_df["cases"] > peak_threshold).astype(int)

    p_peak_flip = model_df.loc[model_df["any_flip_lag0"] == 1, "is_peak"].mean()
    p_peak_no_flip = model_df.loc[model_df["any_flip_lag0"] == 0, "is_peak"].mean()

    print("P(peak | any flip):", p_peak_flip)
    print("P(peak | no flip):", p_peak_no_flip)
    print("Relative increase:", p_peak_flip / p_peak_no_flip)
    
    # Choosing Tokyo for plotting because it has the most cases
    pref = "JP-13"
    for model_name in models.keys():
        plot_fit(model_df, model_name, pref)
    
    plot_flipflop_lag(model_df, pref, lag=2)
    plot_flipflop_vs_cases(model_df, pref)
    plot_contributions_peak_vs_normal(models["flipflops"], model_df, quantile=0.90)

if __name__ == "__main__":
    main()
