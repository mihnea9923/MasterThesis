from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    out_dir: Path = Path("../outputs")
    intermediate_dir_name: str = "intermediate_series"

    data_path: str = "../data"
    weather_file: str = f"{data_path}/daily_meteo_prefs_2005-2019.csv"
    ohca_file: str = f"{data_path}/cardio_clean.csv"
    population_file: str = f"{data_path}/population_ts.csv"

    exclude_prefectures: tuple[str, ...] = ("JP-12",)

    temp_col: str = "t2m"
    max_gap: int = 5
    rolling_temp_window: int = 5
    climatology_window: int = 31
    std_threshold: float = 1.0
    detrend_order: int = 3

    lags: tuple[int, ...] = (0, 1, 2, 3)