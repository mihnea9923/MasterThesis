from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

SSA_FILE = ROOT / "outputs" / "ssa" / "ssa_reconstructed_series.csv.gz"
OUT_DIR = ROOT / "outputs" / "ssa_diagnostics" / "panel_plots"


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_component_panel(
    variable: str,
    component: str,
    n_cols: int = 6,
    share_y: bool = True,
) -> None:
    ensure_out_dir(OUT_DIR)

    usecols = [
        "date",
        "prefecture_code",
        "variable",
        component,
    ]

    df = pd.read_csv(SSA_FILE, usecols=usecols)
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["variable"] == variable].copy()

    if df.empty:
        print(f"No data found for variable={variable}")
        return

    prefectures = sorted(df["prefecture_code"].unique())

    n_rows = int(np.ceil(len(prefectures) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 2.4 * n_rows),
        sharex=True,
        sharey=share_y,
    )

    axes = axes.flatten()

    for ax, prefecture_code in zip(axes, prefectures):
        g = df[df["prefecture_code"] == prefecture_code].sort_values("date")

        ax.plot(g["date"], g[component], color="black", linewidth=0.8)
        ax.set_title(prefecture_code, fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(prefectures):]:
        ax.axis("off")

    fig.suptitle(
        f"{variable} - reconstructed SSA {component} component by prefecture",
        fontsize=16,
    )

    fig.supxlabel("Date")
    fig.supylabel(component)

    plt.tight_layout()

    out_file = OUT_DIR / f"{variable}_{component}_all_prefectures.png"
    plt.savefig(out_file, dpi=200)
    plt.close()

    print(f"Saved: {out_file}")


def main() -> None:
    # This seems to not show any clear interannual signal, so I'm skipping it for now.
    # plot_component_panel("cases", "interannual")

    plot_component_panel("cases", "annual")
    plot_component_panel("cases", "subseasonal")
    plot_component_panel("t2m", "annual")
    plot_component_panel("t2m_std_anom", "subseasonal")
    plot_component_panel("c2w_event", "high_frequency")
    plot_component_panel("w2c_event", "high_frequency")
    plot_component_panel("tp", "high_frequency")
    plot_component_panel("rh", "annual")
    plot_component_panel("ah", "annual")


if __name__ == "__main__":
    main()