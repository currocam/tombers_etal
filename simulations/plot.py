# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==3.0.0",
#     "scienceplots==2.2.0",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use("science")
    return pd, plt, sns


@app.function
def set_size(width, fraction=1):
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    return fig_width_in, fig_height_in


@app.cell
def _(pd):
    data = pd.read_csv("simulations/steps/estimates_constant.csv")
    data = data.dropna()
    data
    return (data,)


@app.cell
def _(data, plt, sns):
    fig2, ax2 = plt.subplots(figsize=set_size(240), dpi=300)
    fraction_error = (data["D_exp"] - data["D_est"]) / data["D_exp"]
    plt.axhline(0, color = "black")
    sns.scatterplot(data=data, x="σ_obs", y = fraction_error, ax=ax2, color="C1")
    plt.xlabel(r"Ground truth dispersal $\sigma$")
    plt.ylabel(r"Density fraction error $\displaystyle \frac{D - \hat D}{D}$")
    plt.xscale("log")
    plt.title("Effect dispersal rate in $D_e$ error")
    plt.show()
    return


@app.cell
def _(data, sns):
    sns.histplot(data["mean_axial_distance"])
    return


@app.cell
def _(data, plt, sns):
    fig1, ax1 = plt.subplots(figsize=set_size(240), dpi=300)
    sns.lineplot(data=data, x="σ_obs", y = "σ_obs", color = "black", ax = ax1)
    sns.scatterplot(data=data, x="σ_obs", y = "σ_est", hue = "mean_axial_distance", ax = ax1)
    plt.xlabel(r"Ground truth dispersal $\sigma$")
    plt.ylabel(r"Estimated dispersal $\displaystyle \sigma_{\text{MLE}}$")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(data, plt, sns):
    fraction_error2 = (data["σ_obs"] - data["σ_est"]) / data["σ_obs"]
    plt.axvline(1, color = "black")
    sns.scatterplot(data=data, x=data["mean_axial_distance"] / data["σ_obs"], y = fraction_error2)
    plt.xlabel(r"Max distance")
    plt.ylabel(r"Dispersal fraction error ")
    plt.title("Effect dispersal rate in $D_e$ error")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
