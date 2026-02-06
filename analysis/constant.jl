using CSV, DataFrames, IdentityByDescentDispersal, Turing, StatsBase, Random
using Base.Threads
using Optim
# %% Set seed for reproducibility
Random.seed!(8376128)
# %% Load data
df = CSV.read("data.csv", DataFrame);
contig_lengths = [0.57, 0.64, 0.52, 0.55, 0.49, 0.53, 0.52, 0.52, 0.57, 0.56, 0.45, 0.54, 0.67, 0.71, 0.59, 0.67, 0.57, 0.57, 0.58, 0.61, 0.54, 0.57, 0.59, 0.51];
df2 = let
    df.DISTANCE_INDEX = ceil.(df.DISTANCE ./ 5)
    combine(
        groupby(df, [:DISTANCE_INDEX, :IBD_LEFT, :IBD_RIGHT]),
        :NR_PAIRS => sum => :NR_PAIRS,
        :COUNT => sum => :COUNT,
        :DISTANCE => mean => :DISTANCE,
    )
end
# %% Find MLE
@model function constant_density(df, contig_lengths)
    D ~ Uniform(0, 1)
    σ ~ Uniform(0, 200)
    # Some custom parallelization
    rows = eachrow(df)
    loglikes = map(rows) do row
        Threads.@spawn begin
            try
                return composite_loglikelihood_constant_density(D, σ, DataFrame(row), contig_lengths)
            catch e
                return -Inf
            end
        end
    end
    Turing.@addlogprob! sum(fetch.(loglikes))
end
m = constant_density(df, contig_lengths);
mle_estimate = maximum_a_posteriori(m, LBFGS())
coef_table = mle_estimate |> coeftable |> DataFrame
select!(coef_table, Not(:z, Symbol("Pr(>|z|)")))
# %% Save results
mkpath("models/constant")
CSV.write("models/constant/mle.csv", coef_table)
