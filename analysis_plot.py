import matplotlib.pyplot as plt
import pandas as pd

def plot_fit(df, model_name, prefecture):
    sub = df[df["prefecture_code"] == prefecture].copy()
    plt.figure(figsize=(12,5))
    plt.plot(sub["date"], sub["cases"], label="Observed", alpha=0.7)
    plt.plot(sub["date"], sub[f"pred_{model_name}"], label="Predicted", alpha=0.7)

    plt.title(f"{model_name} - {prefecture}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_contributions(model, df):
    params = model.params
    contrib = pd.DataFrame(index=df.index)

    for term in params.index:
        if term == "Intercept":
            continue
        if term in df.columns:
            contrib[term] = df[term] * params[term]

    return contrib

def plot_contributions_peak_vs_normal(model, df, quantile=0.90):
    contrib = compute_contributions(model, df)

    threshold = df["cases"].quantile(quantile)
    peak_idx = df["cases"] >= threshold
    normal_idx = df["cases"] < threshold

    peak_contrib = contrib.loc[peak_idx].abs().mean()
    normal_contrib = contrib.loc[normal_idx].abs().mean()

    comparison = pd.DataFrame({
        "peak_days": peak_contrib,
        "normal_days": normal_contrib,
    }).fillna(0)

    comparison = comparison.sort_values("peak_days")

    comparison.plot(kind="barh", figsize=(10, 6))
    plt.title(f"Mean absolute contribution: peak vs normal days")
    plt.tight_layout()
    plt.show()

def plot_flipflop_vs_cases(df, prefecture):
    sub = df[df["prefecture_code"] == prefecture]

    plt.figure(figsize=(12,5))
    plt.plot(sub["date"], sub["cases"], label="OHCA cases")

    # mark flip-flops
    flips = sub[sub["c2w_event"] == 1]
    plt.scatter(flips["date"], flips["cases"], color="red", label="c2w events")

    flips2 = sub[sub["w2c_event"] == 1]
    plt.scatter(flips2["date"], flips2["cases"], color="green", label="w2c events")

    plt.legend()
    plt.title("Flip-flops vs OHCA")
    plt.show()

def plot_flipflop_lag(df, prefecture, lag=2):
    sub = df[df["prefecture_code"] == prefecture].copy()

    sub["c2w_lag"] = sub["c2w_event"].shift(lag)
    sub["w2c_lag"] = sub["w2c_event"].shift(lag)

    plt.figure(figsize=(12,5))
    plt.plot(sub["date"], sub["cases"], label="OHCA")

    plt.scatter(sub[sub["c2w_lag"] == 1]["date"],
                sub[sub["c2w_lag"] == 1]["cases"],
                color="red", label=f"c2w lag{lag}")

    plt.scatter(sub[sub["w2c_lag"] == 1]["date"],
                sub[sub["w2c_lag"] == 1]["cases"],
                color="green", label=f"w2c lag{lag}")

    plt.legend()
    plt.title(f"Flip-flops lag {lag}")
    plt.show()