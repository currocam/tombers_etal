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
from labellines import labelLine, labelLines

plt.style.use("science")


# %%
def set_size(width, fraction=1, subplots=(1, 1)):

    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


# %%
data = pd.read_csv("simulations/steps/estimates_constant.csv")
data

# %%


# %%
from scipy.stats import pearsonr

fig, axes = plt.subplots(
    2, 1, figsize=set_size(240, subplots=(2, 1)), dpi=300, sharey=True
)
# Get unique scales
scales = data["SCALE"].unique()
palette = ["C0", "C1"]
labels = [r"Sampling area $0.5 \times 0.5$", r"Sampling area $1.0 \times 1.0$"]

for idx, scale in enumerate(scales):
    ax = axes[idx]
    # Filter data for this scale
    scale_data = data[data["SCALE"] == scale]

    # Calculate Pearson correlation
    corr, p_value = pearsonr(scale_data["σ_obs"], scale_data["σ_est"])

    # Create scatter plot
    sns.scatterplot(data=scale_data, x="σ_obs", y="σ_est", ax=ax, color=palette[idx])
    sns.lineplot(data=scale_data, x="σ_obs", y="σ_obs", ax=ax, color="black")
    # Add identity line
    l1 = ax.axhline(scale / 2, color="grey", linestyle="--")
    labelLine(
        l1,
        0.09,
        label=f"$\\sigma={scale / 2}$",
        align=True,
        color="black",
        backgroundcolor="white",
    )

    # Add correlation text in bottom right corner
    ax.text(
        0.95,
        0.05,
        f"$\\footnotesize\\text{{Pearson's }} r = {corr:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    # Set labels and scale
    ax.set_xlabel("Ground truth dispersal $\sigma$")
    ax.set_ylabel(r"Estimated dispersal $\displaystyle \sigma_{\text{MLE}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(labels[idx])

plt.tight_layout()
plt.savefig("simulations/estimated_dispersal.pdf")
plt.show()

# %%
fig2, ax2 = plt.subplots(figsize=set_size(240), dpi=300)
sns.scatterplot(data=data, x="σ_obs", y="D_est", hue="D_exp", ax=ax2, color="C1")
plt.xlabel(r"Ground truth dispersal $\sigma$")
plt.ylabel(r"Estimated effective density")
plt.title("Effect dispersal rate in estimated density")
plt.show()
