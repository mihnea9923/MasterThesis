from pathlib import Path

from config import Config
import pandas as pd

def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def export_intermediate_series(
    cfg: Config,
    meteo_all: pd.DataFrame,
    meteo_used: pd.DataFrame,
    daily_cases: pd.DataFrame,
    pop_total: pd.DataFrame,
    flipflops: pd.DataFrame,
    analysis: pd.DataFrame,
) -> None:

    intermediate_dir = cfg.out_dir / cfg.intermediate_dir_name
    ensure_out_dir(intermediate_dir)

    weather_cols = [
        "prefecture_code", "prefecture", "date",
        "t2m", "ah", "rh", "tp", "nw_wind"
    ]
    weather_cols = [c for c in weather_cols if c in meteo_all.columns]

    # The original daily weather covariates, before excluding Chiba.
    meteo_all[weather_cols].sort_values(["prefecture_code", "date"]).to_csv(
        intermediate_dir / "raw_weather_covariates_all_prefectures.csv",
        index=False,
    )
    # The daily weather covariates used in the analysis, after excluding Chiba.
    meteo_used[weather_cols].sort_values(["prefecture_code", "date"]).to_csv(
        intermediate_dir / "raw_weather_covariates_used_excluding_chiba.csv",
        index=False,
    )

    # This comes from cardio_clean.csv, but instead of storing individual OHCA records, I aggregate them to daily prefecture counts.
    daily_cases.sort_values(["prefecture_code", "date"]).to_csv(
        intermediate_dir / "raw_ohca_daily_cases_used_excluding_chiba.csv",
        index=False,
    )
    
    # This contains population by prefecture and year.
    pop_total.sort_values(["prefecture_code", "year"]).to_csv(
        intermediate_dir / "raw_population_yearly_used_excluding_chiba.csv",
        index=False,
    )

    flip_cols = [
        "prefecture_code", "prefecture", "date",
        "t2m", "t2m_detrended", "t2m_roll5_detrended",
        "doy", "clim_mean", "clim_std", "t2m_std_anom",
        "temp_extreme_flag",
        "c2w_event", "w2c_event",
        "c2w_transition_duration", "w2c_transition_duration",
        "c2w_transition_intensity", "w2c_transition_intensity",
        "c2w_start_date", "c2w_end_date",
        "w2c_start_date", "w2c_end_date",
        "ah", "rh", "tp", "nw_wind",
    ]
    flip_cols = [c for c in flip_cols if c in flipflops.columns]

    # This contains the raw weather plus all variables that I created during flip-flop detection.
    flipflops[flip_cols].sort_values(["prefecture_code", "date"]).to_csv(
        intermediate_dir / "postprocessed_temperature_flipflops_by_prefecture.csv",
        index=False,
    )

    # This is the final dataset used for modeling which merges the flip-flop variables with the OHCA cases and population.
    analysis.sort_values(["prefecture_code", "date"]).to_csv(
        intermediate_dir / "model_ready_analysis_dataset.csv",
        index=False,
    )

    # This is a data dictionary describing all columns from the final analysis dataset. I am adding it because it's hard to remember
    # what each column represents, especially for someone who is not familiar with this code.
    data_dictionary = pd.DataFrame([
        ["prefecture_code", "Japanese prefecture identifier"],
        ["prefecture", "Prefecture name"],
        ["date", "Daily date"],
        ["t2m", "Raw daily 2m temperature, Celsius"],
        ["ah", "Absolute humidity"],
        ["rh", "Relative humidity"],
        ["tp", "Daily precipitation"],
        ["nw_wind", "North-west wind speed"],
        ["t2m_detrended", "Temperature after removing long-term polynomial trend"],
        ["t2m_roll5_detrended", "5-day centered rolling mean of detrended temperature"],
        ["doy", "Day of year"],
        ["clim_mean", "Day-of-year climatological mean for each prefecture"],
        ["clim_std", "Day-of-year climatological standard deviation for each prefecture"],
        ["t2m_std_anom", "Standardized temperature anomaly"],
        ["temp_extreme_flag", "-1 cold extreme, 0 normal, 1 warm extreme"],
        ["c2w_event", "Cold-to-warm flip-flop event indicator"],
        ["w2c_event", "Warm-to-cold flip-flop event indicator"],
        ["c2w_transition_duration", "Number of days from cold extreme to warm extreme"],
        ["w2c_transition_duration", "Number of days from warm extreme to cold extreme"],
        ["c2w_transition_intensity", "Absolute standardized anomaly difference for cold-to-warm event"],
        ["w2c_transition_intensity", "Absolute standardized anomaly difference for warm-to-cold event"],
        ["cases", "Daily OHCA case count"],
        ["population", "Yearly prefecture population"],
        ["incidence_per_100k", "Daily OHCA cases per 100,000 population"],
        ["log_population", "Log population used as Poisson offset"],
    ], columns=["column", "description"])

    data_dictionary.to_csv(
        intermediate_dir / "data_dictionary.csv",
        index=False,
    )

    print(f"Saved intermediate series to: {intermediate_dir}")