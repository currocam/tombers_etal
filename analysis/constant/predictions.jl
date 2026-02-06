using CSV, DataFrames, IdentityByDescentDispersal, Turing, StatsBase, Random
using QuadGK # Using Gaussian quadrature rules

mle_short = CSV.read("short_mle.csv", DataFrame)
mle_long = CSV.read("long_mle.csv", DataFrame)

# Constants
contig_lengths = [0.57, 0.64, 0.52, 0.55, 0.49, 0.53, 0.52, 0.52, 0.57, 0.56, 0.45, 0.54, 0.67, 0.71, 0.59, 0.67, 0.57, 0.57, 0.58, 0.61, 0.54, 0.57, 0.59, 0.51]
De(t, params) = params[1] # Custom De(t) parametrization
grid_times = 1:100
L = 0.04 # Smallest IBD block considered
grid_r1 = [2.5, 20, 180] # Geographic distances in km
grid_r2 = [100, 1000, 2000] # Geographic distances in km

# Extract MLE parameters
mle_short_D = mle_short[1, "Coef."]
mle_short_sigma = mle_short[2, "Coef."]
mle_long_D = mle_long[1, "Coef."]
mle_long_sigma = mle_long[2, "Coef."]

# Create tidy dataframe with predictions for all combinations
predictions = DataFrame(
    time=Int[],
    distance_km=Float64[],
    scale=String[],
    density=Float64[]
)

for r in grid_r1
    # scale format predictions
    dens_short = age_density_ibd_blocks_custom(
        grid_times,
        r,
        De,
        [mle_short_D],
        mle_short_sigma,
        L,
        contig_lengths
    )
    for (i, t) in enumerate(grid_times)
        push!(predictions, (time=t, distance_km=r, scale="short", density=dens_short[i]))
    end
end

for r in grid_r2
    # scale format predictions
    dens_short = age_density_ibd_blocks_custom(
        grid_times,
        r,
        De,
        [mle_long_D],
        mle_long_sigma,
        L,
        contig_lengths
    )
    for (i, t) in enumerate(grid_times)
        push!(predictions, (time=t, distance_km=r, scale="long", density=dens_short[i]))
    end
end

# Save the tidy predictions dataframe
CSV.write("predictions_age_ibd.csv", predictions)

# Absolute goodness of fit
predict_short(distance, left, right) = quadgk(x -> expected_ibd_blocks_constant_density(distance, mle_short_D, mle_short_sigma, x, contig_lengths), left, right)[1]
predict_long(distance, left, right) = quadgk(x -> expected_ibd_blocks_constant_density(distance, mle_long_D, mle_long_sigma, x, contig_lengths), left, right)[1]

# Short scale predictions for each row
short_agg = CSV.read("../short_agg_10km.csv", DataFrame)
short_agg = short_agg[short_agg.distance_bin.>0, :]
short_agg.prediction = [predict_short(row.distance_bin, row.IBD_LEFT, row.IBD_RIGHT) for row in eachrow(short_agg)]
CSV.write("short_predictions.csv", short_agg)

# Long scale predictions for each row
long_agg = CSV.read("../long_agg_100km.csv", DataFrame)
long_agg = long_agg[long_agg.distance_bin.>0, :]
long_agg.prediction = [predict_long(row.distance_bin, row.IBD_LEFT, row.IBD_RIGHT) for row in eachrow(long_agg)]
CSV.write("long_predictions.csv", long_agg)

