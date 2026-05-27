import matplotlib.pyplot as plt
import pandas as pd

dominant = pd.read_csv("../outputs/ssa_component_spectral_analysis/ssa_component_dominant_periods.csv")
band = pd.read_csv("../outputs/ssa_component_spectral_analysis/ssa_component_band_power.csv")

# I am looking only at the strongest period for each component
rank1 = dominant[dominant["rank"] == 1].copy()

summary_dominant = (
    rank1
    .groupby(["variable", "component"], as_index=False)
    .agg(
        mean_period_days=("period_days", "mean"),
        median_period_days=("period_days", "median"),
        min_period_days=("period_days", "min"),
        max_period_days=("period_days", "max"),
        n_series=("period_days", "count"),
    )
)

summary_dominant.to_csv(
    "../outputs/ssa_component_spectral_analysis/summary_dominant_periods_by_variable_component.csv",
    index=False
)

print(summary_dominant)


band_summary = (
    band
    .groupby(["variable", "component", "band"], as_index=False)
    .agg(
        mean_band_power_fraction=("band_power_fraction", "mean"),
        median_band_power_fraction=("band_power_fraction", "median"),
        min_band_power_fraction=("band_power_fraction", "min"),
        max_band_power_fraction=("band_power_fraction", "max"),
    )
)

band_summary.to_csv(
    "../outputs/ssa_component_spectral_analysis/summary_band_power_by_variable_component.csv",
    index=False
)

# For each variable/component, get the strongest band
top_band = (
    band_summary
    .sort_values(["variable", "component", "mean_band_power_fraction"], ascending=[True, True, False])
    .groupby(["variable", "component"], as_index=False)
    .first()
)

top_band.to_csv(
    "../outputs/ssa_component_spectral_analysis/top_band_by_variable_component.csv",
    index=False
)

print(top_band)


ssa = pd.read_csv("../outputs/ssa/ssa_reconstructed_series.csv.gz")
ssa["date"] = pd.to_datetime(ssa["date"])

def plot_ssa_components(prefecture_code, variable):
    df = ssa[
        (ssa["prefecture_code"] == prefecture_code) &
        (ssa["variable"] == variable)
    ].copy()

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["original"], label="Original", alpha=0.35)
    plt.plot(df["date"], df["trend"], label="Trend")
    plt.plot(df["date"], df["annual"], label="Annual")
    plt.plot(df["date"], df["subseasonal"], label="Subseasonal")
    plt.legend()
    plt.title(f"{prefecture_code} - {variable} SSA components")
    plt.tight_layout()
    plt.show()

plot_ssa_components("JP-13", "cases")
plot_ssa_components("JP-13", "t2m")
plot_ssa_components("JP-13", "c2w_event")