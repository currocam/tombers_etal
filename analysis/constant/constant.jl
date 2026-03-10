using CSV, DataFrames, IdentityByDescentDispersal, Turing, StatsBase, Random, CodecZlib, DelimitedFiles
using Base.Threads
# %% Set seed for reproducibility
Random.seed!(8376128)
# %% Read preprocessed data
contig_lengths = vec(readdlm(GzipDecompressorStream(open("../../data/input_contig_lengths.txt.gz")), ',', Float64, '\n'))
df_short = CSV.read(GzipDecompressorStream(open("../short_data.csv.gz")), DataFrame)
df_long = CSV.read(GzipDecompressorStream(open("../long_data.csv.gz")), DataFrame)

# %% Find MLE
@model function constant_density(df, contig_lengths)
    D ~ Uniform(0, 1)
    σ ~ Uniform(0, 1000)
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
m = constant_density(df_short, contig_lengths);
mle_estimate = maximum_a_posteriori(m)
coef_table = mle_estimate |> coeftable |> DataFrame
select!(coef_table, Not(:z, Symbol("Pr(>|z|)")))
# %% Save results
CSV.write("short_mle.csv", coef_table)

# %% Repeat with the long dataset
m = constant_density(df_long, contig_lengths);
mle_estimate = maximum_a_posteriori(m)
coef_table = mle_estimate |> coeftable |> DataFrame
select!(coef_table, Not(:z, Symbol("Pr(>|z|)")))
# %% Save results
CSV.write("long_mle.csv", coef_table)
