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

# --- Dispersal plots for each model ---
models = {
    "constant": {"σ_col": "σ_est", "title_suffix": "(Constant density)"},
    "power": {"σ_col": "σ_est_power", "title_suffix": "(Power-law density)"},
}

scales = data["SCALE"].unique()
palette = ["C0", "C1"]
scale_labels = [r"Sampling area $0.5 \times 0.5$", r"Sampling area $1.0 \times 1.0$"]

for model_name, model_info in models.items():
    σ_col = model_info["σ_col"]
    fig, axes = plt.subplots(
        2, 1, figsize=set_size(240, subplots=(2, 1)), dpi=300, sharey=True
    )

    for idx, scale in enumerate(scales):
        ax = axes[idx]
        scale_data = data[data["SCALE"] == scale].dropna(subset=[σ_col])

        corr, p_value = pearsonr(scale_data["σ_obs"], scale_data[σ_col])

        sns.scatterplot(
            data=scale_data, x="σ_obs", y=σ_col, ax=ax, color=palette[idx]
        )
        sns.lineplot(data=scale_data, x="σ_obs", y="σ_obs", ax=ax, color="black")
        l1 = ax.axhline(scale / 2, color="grey", linestyle="--")
        labelLine(
            l1,
            0.09,
            label=f"$\\sigma={scale / 2}$",
            align=True,
            color="black",
            backgroundcolor="white",
        )

        ax.text(
            0.95,
            0.05,
            f"$\\footnotesize\\text{{Pearson's }} r = {corr:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        ax.set_xlabel("Ground truth dispersal $\sigma$")
        ax.set_ylabel(r"Estimated dispersal $\displaystyle \sigma_{\text{MLE}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{scale_labels[idx]} {model_info['title_suffix']}")

    plt.tight_layout()
    plt.savefig(f"simulations/estimated_dispersal_{model_name}.pdf")
    plt.show()

# --- Density plots for each model ---
density_models = {
    "constant": {"D_col": "D_est", "title": "Effect of dispersal rate on estimated density (Constant)"},
    "power": {"D_col": "D_est_power", "title": "Effect of dispersal rate on estimated density (Power-law)"},
}

for model_name, model_info in density_models.items():
    D_col = model_info["D_col"]
    plot_data = data.dropna(subset=[D_col])
    fig2, ax2 = plt.subplots(figsize=set_size(240), dpi=300)
    sns.scatterplot(data=plot_data, x="σ_obs", y=D_col, hue="D_exp", ax=ax2, color="C1")
    plt.xlabel(r"Ground truth dispersal $\sigma$")
    plt.ylabel(r"Estimated effective density")
    plt.title(model_info["title"])
    plt.savefig(f"simulations/estimated_density_{model_name}.pdf")
    plt.show()
