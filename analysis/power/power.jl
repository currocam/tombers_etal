using CSV, DataFrames, IdentityByDescentDispersal, Turing, StatsBase, Random
using Base.Threads
# %% Set seed for reproducibility
Random.seed!(8376128)
# %% Load data
df_short = CSV.read("../short_data.csv", DataFrame);
contig_lengths = [0.57, 0.64, 0.52, 0.55, 0.49, 0.53, 0.52, 0.52, 0.57, 0.56, 0.45, 0.54, 0.67, 0.71, 0.59, 0.67, 0.57, 0.57, 0.58, 0.61, 0.54, 0.57, 0.59, 0.51];
df2_short = let
    df_short.DISTANCE_INDEX = ceil.(df_short.DISTANCE ./ 5)
    combine(
        groupby(df_short, [:DISTANCE_INDEX, :IBD_LEFT, :IBD_RIGHT]),
        :NR_PAIRS => sum => :NR_PAIRS,
        :COUNT => sum => :COUNT,
        :DISTANCE => mean => :DISTANCE,
    )
end
# %% Find MLE
@model function power_density(df, contig_lengths)
    D ~ Uniform(0, 1)
    β ~ Uniform(-3, 3)
    σ ~ Uniform(0, 1000)
    # Some custom parallelization
    rows = eachrow(df)
    loglikes = map(rows) do row
        Threads.@spawn begin
            try
                return composite_loglikelihood_power_density(D, β, σ, DataFrame(row), contig_lengths)
            catch e
                return -Inf
            end
        end
    end
    Turing.@addlogprob! sum(fetch.(loglikes))
end
m = power_density(df2_short, contig_lengths);
mle_estimate = maximum_a_posteriori(m)
coef_table = mle_estimate |> coeftable |> DataFrame
select!(coef_table, Not(:z, Symbol("Pr(>|z|)")))
# %% Save results
CSV.write("short_mle.csv", coef_table)

# %% Repeat with the long dataset
df_long = CSV.read("../long_data.csv", DataFrame);
df2_long = let
    df_long.DISTANCE_INDEX = ceil.(df_long.DISTANCE ./ 5)
    combine(
        groupby(df_long, [:DISTANCE_INDEX, :IBD_LEFT, :IBD_RIGHT]),
        :NR_PAIRS => sum => :NR_PAIRS,
        :COUNT => sum => :COUNT,
        :DISTANCE => mean => :DISTANCE,
    )
end
m = power_density(df2_long, contig_lengths);
mle_estimate = maximum_a_posteriori(m)
coef_table = mle_estimate |> coeftable |> DataFrame
select!(coef_table, Not(:z, Symbol("Pr(>|z|)")))
# %% Save results
CSV.write("long_mle.csv", coef_table)
