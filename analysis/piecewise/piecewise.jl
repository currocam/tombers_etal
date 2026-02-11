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
# %% Sigmoid transition
sigmoid(x) = 1 / (1 + exp(-x))
function De(t, θ)
    D0, D1, t0 = θ
    D0 + (D1 - D0) * sigmoid(10 * (t - t0))
end
# %% Find MLE
@model function piecewise_density(df, contig_lengths)
    D0 ~ Uniform(0, 1)
    D1 ~ Uniform(0, 1)
    t0 ~ Uniform(0, 500)
    σ ~ Uniform(0, 1000)
    # Some custom parallelization
    rows = eachrow(df)
    loglikes = map(rows) do row
        Threads.@spawn begin
            try
                return composite_loglikelihood_custom(De, [D0, D1, t0], σ, DataFrame(row), contig_lengths)
            catch e
                return -Inf
            end
        end
    end
    Turing.@addlogprob! sum(fetch.(loglikes))
end
function fit(m; n=3)
    estimates = [maximum_a_posteriori(m) for _ in 1:n]
    _, best = findmax(e -> e.lp, estimates)
    return estimates[best]
end
m = piecewise_density(df2_short, contig_lengths);
mle_estimate = fit(m)
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
m = piecewise_density(df2_long, contig_lengths);
mle_estimate = fit(m)
coef_table = mle_estimate |> coeftable |> DataFrame
select!(coef_table, Not(:z, Symbol("Pr(>|z|)")))
# %% Save results
CSV.write("long_mle.csv", coef_table)
