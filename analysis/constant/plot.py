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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

plt.style.use("science")

# %%
data = pd.read_csv("analysis/constant/predictions_age_ibd.csv")
data = data.dropna()
data["distance_km"] = [f"{x:01} km" for x in data["distance_km"]]
print(data)


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
# Hack to split the legend into two keys
def dist_order(subset):
    return sorted(subset["distance_km"].unique(), key=lambda s: float(s.split()[0]))


def split_legend(ax, dist_labels, bin_labels):
    handles, labels = ax.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    leg_dist = ax.legend(
        [lookup[k] for k in dist_labels],
        dist_labels,
        title="Distance",
        loc="upper right",
        fontsize="small",
        title_fontsize="small",
    )
    ax.add_artist(leg_dist)
    # Stack the second key right below the first one
    ax.figure.canvas.draw()
    bottom = leg_dist.get_window_extent().transformed(ax.transAxes.inverted()).y0
    ax.legend(
        [lookup[k] for k in bin_labels],
        bin_labels,
        title="Block length",
        loc="upper right",
        bbox_to_anchor=(1.0, bottom),
        fontsize="small",
        title_fontsize="small",
    )


# %%
fig1, ax1 = plt.subplots(figsize=set_size(240, fraction=1.0), dpi=300)

sub_ax1 = data[data.scale == "short"]
order_ax1 = dist_order(sub_ax1)
bins_ax1 = sorted(sub_ax1["bin"].unique())

sns.lineplot(
    data=sub_ax1,
    x="time",
    y="density",
    hue="distance_km",
    hue_order=order_ax1,
    style="bin",
    style_order=bins_ax1,
    ax=ax1,
    palette=["C0", "C1", "C2"],
)
split_legend(ax1, order_ax1, bins_ax1)
plt.ylabel("Density of expected shared \n   blocks per pair and Morgan")
plt.xlabel("Time (generations ago)")
plt.xlim(0, 200)
plt.savefig("analysis/constant/age_ibd_short_scale.pdf")
plt.show()
plt.close()

# %%
fig2, ax2 = plt.subplots(figsize=set_size(240), dpi=300)

sub_ax2 = data[data.scale == "long"]
order_ax2 = dist_order(sub_ax2)
bins_ax2 = sorted(sub_ax2["bin"].unique())

sns.lineplot(
    data=sub_ax2,
    x="time",
    y="density",
    hue="distance_km",
    hue_order=order_ax2,
    style="bin",
    style_order=bins_ax2,
    ax=ax2,
    palette=["C3", "C4", "C5"],
)
split_legend(ax2, order_ax2, bins_ax2)
plt.ylabel("Density of expected shared \n   blocks per pair and Morgan")
plt.xlabel("Time (generations ago)")
plt.xlim(0, 200)
plt.savefig("analysis/constant/age_ibd_long_scale.pdf")
plt.show()
plt.close()

# %%
data2 = pd.read_csv("analysis/constant/short_predictions.csv")
data2["BIN_INDEX"] = data2["BIN_INDEX"].map({1: "1.0-2.5 cM", 2: "2.5-5.0 cM"})
print(data2)

# %%
data3 = pd.read_csv("analysis/constant/long_predictions.csv")
data3["BIN_INDEX"] = data3["BIN_INDEX"].map({1: "1.0-2.5 cM", 2: "2.5-5.0 cM"})
print(data3)

# %% Add predictions from the model
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
palette = {"1.0-2.5 cM": "C0", "2.5-5.0 cM": "C1"}

# %%
def plot_predictions(df, filename, log=False):
    fig, ax = plt.subplots(figsize=set_size(240), dpi=300)
    for label in ["1.0-2.5 cM", "2.5-5.0 cM"]:
        subset = df[df["BIN_INDEX"] == label].sort_values("distance_bin")
        sns.lineplot(
            data=subset,
            x="distance_bin",
            y="prediction",
            ax=ax,
            color=palette[label],
            label=label,
        )
        ax.fill_between(
            subset["distance_bin"],
            subset["lower_pred"],
            subset["upper_pred"],
            color=palette[label],
            alpha=0.2,
            linewidth=0,
        )
        ax.errorbar(
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
    ax.legend([])
    fig.legend(title="", loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, 0.9))
    if log:
        ax.set_yscale("log")
        plt.ylabel("Number of shared \n IBD blocks (log-scale)")
    else:
        plt.ylabel("Number of shared IBD blocks")
    plt.xlabel("Geographic distance (kilometers)")
    plt.savefig(filename)
    plt.show()
    plt.close()


# %%
plot_predictions(data2, "analysis/constant/predictions_short.pdf")

# %%
plot_predictions(data2, "analysis/constant/predictions_log_short.pdf", log=True)

# %% Same but with the long-scale predictions
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
plot_predictions(data3, "analysis/constant/predictions_long.pdf")

# %%
plot_predictions(data3, "analysis/constant/predictions_log_long.pdf", log=True)
