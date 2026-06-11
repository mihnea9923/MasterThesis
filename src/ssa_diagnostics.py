from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "outputs" / "intermediate_series" / "model_ready_analysis_dataset.csv"
OUT_DIR = ROOT / "outputs" / "ssa_diagnostics"

WINDOW_SIZE = 365

DIAGNOSTIC_PREFECTURES = [
    "JP-13",
]

DIAGNOSTIC_VARIABLES = [
    "cases",
    "incidence_per_100k",
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

def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_daily_series(df: pd.DataFrame, variable: str) -> pd.Series:
    s = df[["date", variable]].copy()
    s["date"] = pd.to_datetime(s["date"])

    s = s.groupby("date")[variable].mean().sort_index()

    full_index = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_index)

    zero_fill_variables = {
        "cases",
        "c2w_event",
        "w2c_event",
        "c2w_transition_intensity",
        "w2c_transition_intensity",
    }

    if variable in zero_fill_variables:
        s = s.fillna(0)
    else:
        s = s.interpolate(limit_direction="both")

    return s


def make_trajectory_matrix(x: np.ndarray, window_size: int) -> np.ndarray:
    n = len(x)
    k = n - window_size + 1

    if k <= 0:
        raise ValueError("Window size is too large for the time series length.")

    return np.column_stack([
        x[i:i + window_size]
        for i in range(k)
    ])


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    l, k = matrix.shape
    n = l + k - 1

    out = np.zeros(n)
    counts = np.zeros(n)

    for i in range(l):
        for j in range(k):
            out[i + j] += matrix[i, j]
            counts[i + j] += 1

    return out / counts


def run_basic_ssa(series: pd.Series, window_size: int = WINDOW_SIZE) -> dict:
    x = series.to_numpy(dtype=float)

    if not np.isfinite(x).all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()

    window_size = min(window_size, len(x) // 2)

    trajectory = make_trajectory_matrix(x, window_size)
    U, s, Vt = np.linalg.svd(trajectory, full_matrices=False)

    elementary_components = []

    for i in range(len(s)):
        elementary_matrix = s[i] * np.outer(U[:, i], Vt[i, :])
        component = diagonal_averaging(elementary_matrix)
        elementary_components.append(component)

    elementary_components = np.asarray(elementary_components)

    variance_fraction = s**2 / np.sum(s**2)
    cumulative_variance_fraction = np.cumsum(variance_fraction)

    return {
        "x": x,
        "dates": series.index,
        "window_size": window_size,
        "U": U,
        "s": s,
        "Vt": Vt,
        "components": elementary_components,
        "variance_fraction": variance_fraction,
        "cumulative_variance_fraction": cumulative_variance_fraction,
    }


def dominant_period_days(component: np.ndarray) -> float:
    x = np.asarray(component, dtype=float)
    x = x - np.nanmean(x)

    if np.nanstd(x) == 0:
        return np.inf

    freqs = np.fft.rfftfreq(len(x), d=1.0)
    power = np.abs(np.fft.rfft(x)) ** 2

    freqs = freqs[1:]
    power = power[1:]

    if len(freqs) == 0:
        return np.inf

    idx = int(np.argmax(power))
    dominant_freq = freqs[idx]

    if dominant_freq <= 0:
        return np.inf

    return float(1 / dominant_freq)


def classify_period(period_days: float) -> str:
    if not np.isfinite(period_days):
        return "trend"

    if period_days >= 365 * 3:
        return "trend"
    if 365 * 1.2 <= period_days < 365 * 3:
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

    return "noise"


def build_component_metadata(ssa_result: dict) -> pd.DataFrame:
    rows = []

    components = ssa_result["components"]
    s = ssa_result["s"]
    variance_fraction = ssa_result["variance_fraction"]
    cumulative_variance_fraction = ssa_result["cumulative_variance_fraction"]

    for i, component in enumerate(components):
        period = dominant_period_days(component)
        band = classify_period(period)

        rows.append({
            "component": i + 1,
            "singular_value": s[i],
            "norm": s[i],
            "variance_fraction": variance_fraction[i],
            "cumulative_variance_fraction": cumulative_variance_fraction[i],
            "dominant_period_days": period,
            "assigned_band": band,
        })

    return pd.DataFrame(rows)


def build_groups(metadata: pd.DataFrame, max_components: int = 80) -> dict:
    metadata = metadata[metadata["component"] <= max_components].copy()

    groups = {}

    for band, g in metadata.groupby("assigned_band"):
        components = g["component"].astype(int).tolist()
        groups[band] = components

    return groups


def reconstruct_groups(ssa_result: dict, groups: dict) -> pd.DataFrame:
    components = ssa_result["components"]
    dates = ssa_result["dates"]
    x = ssa_result["x"]

    out = pd.DataFrame({
        "date": dates,
        "original": x,
    })

    reconstructed_sum = np.zeros_like(x, dtype=float)

    for group_name, component_numbers in groups.items():
        idx = [c - 1 for c in component_numbers]
        group_series = components[idx, :].sum(axis=0)
        out[group_name] = group_series

        if group_name != "noise":
            reconstructed_sum += group_series

    out["reconstructed_without_noise"] = reconstructed_sum
    out["residual"] = out["original"] - out["reconstructed_without_noise"]

    return out


def weighted_correlation_matrix(components: np.ndarray, window_size: int) -> np.ndarray:
    n_components, n = components.shape
    l = window_size
    k = n - l + 1

    weights = np.array([
        min(i + 1, l, k, n - i)
        for i in range(n)
    ], dtype=float)

    corr = np.eye(n_components)

    for i in range(n_components):
        xi = components[i]
        xi_centered = xi - np.average(xi, weights=weights)

        for j in range(i + 1, n_components):
            xj = components[j]
            xj_centered = xj - np.average(xj, weights=weights)

            numerator = np.sum(weights * xi_centered * xj_centered)
            denom = np.sqrt(
                np.sum(weights * xi_centered**2)
                * np.sum(weights * xj_centered**2)
            )

            value = numerator / denom if denom > 0 else 0.0

            corr[i, j] = value
            corr[j, i] = value

    return corr


def plot_component_norms(metadata: pd.DataFrame, out_file: Path, max_components: int = 50) -> None:
    df = metadata.head(max_components)

    plt.figure(figsize=(9, 6))
    plt.plot(df["component"], df["norm"], marker="o")
    plt.yscale("log")
    plt.xlabel("Component index")
    plt.ylabel("Component norm / singular value")
    plt.title("SSA component norms")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_eigenvector_pairs(ssa_result: dict, metadata: pd.DataFrame, out_file: Path, max_pairs: int = 30) -> None:
    U = ssa_result["U"]

    n_pairs = min(max_pairs, U.shape[1] - 1)
    n_cols = 6
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.7 * n_rows))
    axes = axes.flatten()

    for pair_idx in range(n_pairs):
        ax = axes[pair_idx]

        i = pair_idx
        j = pair_idx + 1

        ax.plot(U[:, i], U[:, j], linewidth=1)

        vi = metadata.loc[metadata["component"] == i + 1, "variance_fraction"].iloc[0] * 100
        vj = metadata.loc[metadata["component"] == j + 1, "variance_fraction"].iloc[0] * 100

        ax.set_title(f"{i + 1} ({vi:.2f}%) vs {j + 1} ({vj:.2f}%)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.suptitle("Pairs of SSA eigenvectors", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_w_correlation(ssa_result: dict, out_file: Path, max_components: int = 50) -> None:
    components = ssa_result["components"][:max_components]
    window_size = ssa_result["window_size"]

    corr = weighted_correlation_matrix(components, window_size)

    plt.figure(figsize=(9, 8))
    plt.imshow(np.abs(corr), cmap="Greys", origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="Absolute weighted correlation")
    plt.title("SSA W-correlation matrix")
    plt.xlabel("Component")
    plt.ylabel("Component")

    ticks = np.arange(max_components)
    labels = [f"F{i + 1}" for i in ticks]

    plt.xticks(ticks, labels, rotation=90, fontsize=6)
    plt.yticks(ticks, labels, fontsize=6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_reconstructed_components(reconstructed: pd.DataFrame, out_file: Path) -> None:
    possible_components = [
        "original",
        "trend",
        "interannual",
        "annual",
        "semiannual",
        "seasonal_subannual",
        "subseasonal",
        "high_frequency",
        "noise",
        "residual",
    ]

    components = [c for c in possible_components if c in reconstructed.columns]

    n_cols = 2
    n_rows = int(np.ceil(len(components) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, component in enumerate(components):
        ax = axes[i]
        ax.plot(reconstructed["date"], reconstructed[component], linewidth=1)
        ax.set_title(component)
        ax.grid(alpha=0.3)

    for ax in axes[len(components):]:
        ax.axis("off")

    fig.suptitle("SSA reconstructed series", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def save_group_code(groups: dict, out_file: Path) -> None:
    lines = []
    lines.append("# Python equivalent of the Rssa grouping list")
    lines.append("groups = {")

    for group_name, component_numbers in groups.items():
        lines.append(f'    "{group_name}": {component_numbers},')

    lines.append("}")

    out_file.write_text("\n".join(lines))


def run_diagnostics_for_series(
    df: pd.DataFrame,
    prefecture_code: str,
    variable: str,
) -> None:
    example_dir = OUT_DIR / f"{prefecture_code}_{variable}"
    ensure_out_dir(example_dir)

    pref_df = df[df["prefecture_code"] == prefecture_code].copy()

    if pref_df.empty:
        print(f"Skipping {prefecture_code}-{variable}: prefecture not found")
        return

    if variable not in pref_df.columns:
        print(f"Skipping {prefecture_code}-{variable}: variable not found")
        return

    series = prepare_daily_series(pref_df, variable)

    print(f"Running SSA diagnostics for {prefecture_code} - {variable}")

    ssa_result = run_basic_ssa(series, window_size=WINDOW_SIZE)
    metadata = build_component_metadata(ssa_result)
    groups = build_groups(metadata, max_components=80)
    reconstructed = reconstruct_groups(ssa_result, groups)

    metadata["prefecture_code"] = prefecture_code
    metadata["variable"] = variable

    metadata.to_csv(example_dir / "component_period_estimates.csv", index=False)
    reconstructed.to_csv(example_dir / "reconstructed_components.csv", index=False)
    save_group_code(groups, example_dir / "component_groups.py")

    plot_component_norms(
        metadata,
        example_dir / "component_norms.png",
        max_components=50,
    )

    plot_eigenvector_pairs(
        ssa_result,
        metadata,
        example_dir / "eigenvector_pairs.png",
        max_pairs=30,
    )

    plot_w_correlation(
        ssa_result,
        example_dir / "w_correlation_matrix.png",
        max_components=50,
    )

    plot_reconstructed_components(
        reconstructed,
        example_dir / "reconstructed_components.png",
    )

    print(f"Saved diagnostics to {example_dir}")


def main() -> None:
    ensure_out_dir(OUT_DIR)

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    for prefecture_code in DIAGNOSTIC_PREFECTURES:
        for variable in DIAGNOSTIC_VARIABLES:
            run_diagnostics_for_series(df, prefecture_code, variable)


if __name__ == "__main__":
    main()