#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "matplotlib-label-lines==0.8.1",
#     "numpy==2.4.2",
#     "pandas==3.0.0",
#     "scienceplots==2.2.0",
#     "scipy==1.17.0",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.12"
# ///

__generated_with = "0.19.9"

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

# %%
pd.read_csv("analysis/constant/predictions_age_ibd.csv")["distance_km"].unique()

# %%
data = pd.read_csv("analysis/constant/predictions_age_ibd.csv")
data = data.dropna()
data["distance_km"] = [f"{x:01} km" for x in data["distance_km"]]
data


# %%
def set_size(width, fraction=1):
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    return fig_width_in, fig_height_in


# %%
data[data.scale == "long"]["distance_km"].unique()

# %%
data

# %%
fig1, ax1 = plt.subplots(figsize=set_size(240), dpi=300)

sns.lineplot(
    data=data[data.scale == "short"],
    x="time",
    y="density",
    hue="distance_km",
    ax=ax1,
    palette=["C0", "C1", "C2"],
)
ax1.legend(title="")
plt.ylabel("Density of expected shared \n   blocks per pair and Morgan")
plt.xlabel("Time (generations ago)")
plt.savefig("analysis/constant/age_ibd_short_scale.pdf")
plt.show()
plt.close()

# %%
fig2, ax2 = plt.subplots(figsize=set_size(240), dpi=300)

sns.lineplot(
    data=data[data.scale == "long"],
    x="time",
    y="density",
    hue="distance_km",
    ax=ax2,
    palette=["C3", "C4", "C5"],
)
ax2.legend(title="")
plt.ylabel("Density of expected shared \n   blocks per pair and Morgan")
plt.xlabel("Time (generations ago)")
plt.savefig("analysis/constant/age_ibd_long_scale.pdf")
plt.show()
plt.close()

# %%
data2 = pd.read_csv("analysis/constant/short_predictions.csv")
data2["BIN_INDEX"] = data2["BIN_INDEX"].map(
    {1: "0.4-1.0 cM", 2: "1.0-2.5 cM", 3: "2.5-4.0 cM"}
)
data2

# %%
data3 = pd.read_csv("analysis/constant/long_predictions.csv")
data3["BIN_INDEX"] = data3["BIN_INDEX"].map(
    {1: "0.4-1.0 cM", 2: "1.0-2.5 cM", 3: "2.5-4.0 cM"}
)
data3

# %%
import scipy.stats

rng = np.random.default_rng(1234)
n = data2.shape[0]
# Parametric bootstrapping
num_draws = 10_000
pred_conf = np.array(
    [
        scipy.stats.poisson(rate).rvs((n, num_draws), rng).mean(axis=0)
        for rate, n in zip(data2["prediction"], data2["n"])
    ]
)
lower_pred, upper_pred = np.quantile(pred_conf, [0.025, 0.975], axis=1)
data2["lower_pred"] = lower_pred
data2["upper_pred"] = upper_pred

# %%
palette = {"0.4-1.0 cM": "C3", "1.0-2.5 cM": "C0", "2.5-4.0 cM": "C1"}

# %%
fig3, ax3 = plt.subplots(figsize=set_size(240), dpi=300)
for i, label in enumerate(["0.4-1.0 cM", "1.0-2.5 cM", "2.5-4.0 cM"]):
    subset = data2[data2["BIN_INDEX"] == label]
    subset = data2[data2["BIN_INDEX"] == label].sort_values("distance_bin")
    g1 = sns.lineplot(
        data=subset,
        x="distance_bin",
        y="prediction",
        ax=ax3,
        color=palette[label],
        label=label,
    )
    ax3.fill_between(
        subset["distance_bin"],
        subset["lower_pred"],
        subset["upper_pred"],
        color=palette[label],
        alpha=0.2,
        linewidth=0,
    )
    ax3.errorbar(
        subset["distance_bin"],
        subset["mean"],
        yerr=[subset["mean"] - subset["lower"], subset["upper"] - subset["mean"]],
        fmt="o",
        color=palette[label],
        solid_capstyle="round",
        zorder=1,
        linewidth=1,
        markersize=2,
    )
ax3.legend([])
fig3.legend(title="", loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, 0.9))
plt.ylabel("Number of shared IBD blocks")
plt.xlabel("Geographic distance (kilometers)")
plt.savefig("analysis/constant/predictions_short.pdf")
plt.show()

# %%
n2 = data3.shape[0]
pred_conf2 = np.array(
    [
        scipy.stats.poisson(rate).rvs((n, num_draws), rng).mean(axis=0)
        for rate, n in zip(data3["prediction"], data3["n"])
    ]
)
lower_pred2, upper_pred2 = np.quantile(pred_conf2, [0.025, 0.975], axis=1)
data3["lower_pred"] = lower_pred2
data3["upper_pred"] = upper_pred2

# %%
fig4, ax4 = plt.subplots(figsize=set_size(240), dpi=300)
for j, label2 in enumerate(["0.4-1.0 cM", "1.0-2.5 cM", "2.5-4.0 cM"]):
    subset2 = data3[data3["BIN_INDEX"] == label2]
    subset2 = data3[data3["BIN_INDEX"] == label2].sort_values("distance_bin")
    g3 = sns.lineplot(
        data=subset2,
        x="distance_bin",
        y="prediction",
        ax=ax4,
        color=palette[label2],
        label=label2,
    )
    ax4.fill_between(
        subset2["distance_bin"],
        subset2["lower_pred"],
        subset2["upper_pred"],
        color=palette[label2],
        alpha=0.2,
        linewidth=0,
    )
    ax4.errorbar(
        subset2["distance_bin"],
        subset2["mean"],
        yerr=[subset2["mean"] - subset2["lower"], subset2["upper"] - subset2["mean"]],
        fmt="o",
        color=palette[label2],
        solid_capstyle="round",
        zorder=1,
        linewidth=1,
        markersize=2,
    )
ax4.legend([])
fig4.legend(title="", loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, 0.9))

plt.ylabel("Number of shared IBD blocks")
plt.xlabel("Geographic distance (kilometers)")
plt.savefig("analysis/constant/predictions_long.pdf")
plt.show()
