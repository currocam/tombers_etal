using CSV, DataFrames, IdentityByDescentDispersal, Turing, StatsBase, Random

mle_short = CSV.read("short_mle.csv")
mle_long = CSV.read("long_mle.csv")
# First, we make predictions about the age density of IBD blocks
#
dens = age_density_ibd_blocks_custom(
    grid_times,
    r,
    De,
    params,
    sigma,
    L,
    data["contig_lengths"],
)
