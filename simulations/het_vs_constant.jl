using PyCall
using Random
using DataFrames
using StatsBase
using LinearAlgebra
using IdentityByDescentDispersal
using Turing
using CSV

include("utils.jl")

const SLIM_BINARY = get(ENV, "SLIM_BINARY", "slim")
const SLIM_FILE = "het_habitat.slim"
const OUTPUT_DIR = "steps"

function simulation(NE, SD, SM, SCALE, seed)
    unique_id = "het_vs_constant_$(NE)_$(SD)_$(SM)_$(SCALE)_$(seed)"
    outpath = joinpath(OUTPUT_DIR, "$(unique_id).trees")
    rng = MersenneTwister(seed)

    run(`$SLIM_BINARY -p -s $seed -d NE=$NE -d SD=$SD -d SM=$SM -d OUTPATH="\"$outpath\"" $SLIM_FILE`)

    # Data preprocessing
    tskit = pyimport("tskit")
    ts = tskit.load(outpath)
    pyslim = pyimport("pyslim")
    ts = pyslim.recapitate(ts, ancestral_Ne=NE, recombination_rate=1e-8)

    # Sample randomly from individuals within a square centered at the origin
    n_samples = 100

    all_locations = reduce(hcat, [collect(row) for row in ts.individual_locations])'
    all_locations = all_locations[:, 1:2]
    mean_pos = mean(all_locations, dims=1)
    centered_locations = all_locations .- mean_pos

    within_square = findall(row -> abs(row[1]) <= SCALE && abs(row[2]) <= SCALE, eachrow(centered_locations))
    n_available = length(within_square)
    if n_available >= n_samples
        all_sampled_individuals = sample(rng, within_square, n_samples, replace=false)
    else
        @warn "Only $n_available individuals within square, sampling all of them"
        all_sampled_individuals = within_square
    end

    nodes = reduce(vcat, [ts.individual(i - 1).nodes for i in all_sampled_individuals])
    ts = ts.simplify(samples=nodes)

    df_dist, mean_distance = let
        points = reduce(hcat, [collect(row) for row in ts.individual_locations])'
        dist_matrix = euclidean_distance(points)
        n = size(points, 1)
        df = DataFrame(ID1=Int[], ID2=Int[], distance=Float64[])
        for i = 1:n
            for j = (i+1):n
                push!(df, (i - 1, j - 1, dist_matrix[i, j]))
            end
        end
        df, mean(df.distance)
    end

    df_ibds = let
        function blocks(i, j)
            ibds = ts.ibd_segments(
                between=[ts.individual(i).nodes, ts.individual(j).nodes],
                min_span=1 / 100 * 4 * 1e8,
                store_pairs=true,
                store_segments=true,
            )
            spans = [
                (block.right - block.left) / 1e8 for pair in ibds for
                block in ibds.get(pair)
            ]
            DataFrame(ID1=i, ID2=j, span=spans)
        end
        vcat([blocks(i, j) for i = 0:(n_samples-1) for j = (i+1):(n_samples-1)]...)
    end

    # Preprocessing
    bins, min_threshold = default_ibd_bins()
    df_preprocessed = preprocess_dataset(df_ibds, df_dist, bins, min_threshold)
    df2 = let
        h = fit(Histogram, df_preprocessed.DISTANCE, nbins=50)
        df_preprocessed.DISTANCE_INDEX =
            cut(df_preprocessed.DISTANCE, collect(h.edges[1]))
        combine(
            groupby(df_preprocessed, [:DISTANCE_INDEX, :IBD_LEFT, :IBD_RIGHT]),
            :NR_PAIRS => sum => :NR_PAIRS,
            :COUNT => sum => :COUNT,
            :DISTANCE => mean => :DISTANCE,
        )
    end

    # Ground truth from SLiM metadata
    local_density_obs = ts.metadata["SLiM"]["user_metadata"]["OBS_D"][1]
    local_density_exp = ts.metadata["SLiM"]["user_metadata"]["EXP_D"][1]
    dispersal_rate_obs = ts.metadata["SLiM"]["user_metadata"]["OBS_SIGMA"][1]
    dispersal_rate_exp = ts.metadata["SLiM"]["user_metadata"]["EXP_SIGMA"][1]

    contig_lengths = [1.0]
    n_starts = 5

    # --- Fit constant model ---
    @model function constant_density_model(df, contig_lengths)
        D ~ Uniform(0, 10000)
        σ ~ Uniform(0, 500.0)
        try
            Turing.@addlogprob! composite_loglikelihood_constant_density(D, σ, df, contig_lengths)
        catch e
            @warn "Error in constant_density model: $e"
            Turing.@addlogprob! -Inf
        end
    end

    m_const = constant_density_model(df2, contig_lengths)
    best_mle_const = nothing
    best_lp_const = -Inf
    for i in 1:n_starts
        try
            init_D = rand(rng) * 10000
            init_σ = rand(rng) * 500
            mle_result = maximum_a_posteriori(m_const; initial_params=[init_D, init_σ])
            if mle_result.lp > best_lp_const
                best_lp_const = mle_result.lp
                best_mle_const = mle_result
            end
        catch e
            @warn "Constant MLE attempt $i failed: $e"
        end
    end
    if best_mle_const === nothing
        error("All constant MLE attempts failed")
    end

    # --- Fit power-law model ---
    @model function power_density_model(df, contig_lengths)
        D ~ Uniform(0, 10000)
        β ~ Uniform(-1, 1)
        σ ~ Uniform(0, 500.0)
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

    m_power = power_density_model(df2, contig_lengths)
    best_mle_power = nothing
    best_lp_power = -Inf
    for i in 1:n_starts
        try
            init_D = rand(rng) * 10000
            init_β = rand(rng) * 2 - 1
            init_σ = rand(rng) * 500
            mle_result = maximum_a_posteriori(m_power; initial_params=[init_D, init_β, init_σ])
            if mle_result.lp > best_lp_power
                best_lp_power = mle_result.lp
                best_mle_power = mle_result
            end
        catch e
            @warn "Power MLE attempt $i failed: $e"
        end
    end
    if best_mle_power === nothing
        error("All power-law MLE attempts failed")
    end

    # --- Compare models ---
    k_const = 2  # D, σ
    k_power = 3  # D, β, σ
    aic_const = 2 * k_const - 2 * best_lp_const
    aic_power = 2 * k_power - 2 * best_lp_power

    coef_const = best_mle_const |> coeftable |> DataFrame
    coef_power = best_mle_power |> coeftable |> DataFrame

    println("\n" * "="^60)
    println("  Heterogeneous Habitat: Constant vs Power-law Density")
    println("="^60)
    println("\nSimulation parameters:")
    println("  NE=$NE, SD=$SD, SM=$SM, SCALE=$SCALE")
    println("  Expected density: $local_density_exp")
    println("  Observed density: $local_density_obs")
    println("  Expected σ: $dispersal_rate_exp")
    println("  Observed σ: $dispersal_rate_obs")
    println("\n--- Constant model (D, σ) ---")
    println("  D = $(coef_const[1, "Coef."]) ± $(coef_const[1, "Std. Error"])")
    println("  σ = $(coef_const[2, "Coef."]) ± $(coef_const[2, "Std. Error"])")
    println("  Log-likelihood = $best_lp_const")
    println("  AIC = $aic_const")
    println("\n--- Power-law model (D·t^-β, σ) ---")
    println("  D = $(coef_power[1, "Coef."]) ± $(coef_power[1, "Std. Error"])")
    println("  β = $(coef_power[2, "Coef."]) ± $(coef_power[2, "Std. Error"])")
    println("  σ = $(coef_power[3, "Coef."]) ± $(coef_power[3, "Std. Error"])")
    println("  Log-likelihood = $best_lp_power")
    println("  AIC = $aic_power")
    println("\n--- Comparison ---")
    println("  ΔAIC (constant - power) = $(aic_const - aic_power)")
    if aic_power < aic_const
        println("  → Power-law model is preferred")
    else
        println("  → Constant model is preferred")
    end
    println("="^60)

    # Build results DataFrame
    results = DataFrame(
        model=["constant", "power"],
        D_est=[coef_const[1, "Coef."], coef_power[1, "Coef."]],
        D_se=[coef_const[1, "Std. Error"], coef_power[1, "Std. Error"]],
        beta_est=[NaN, coef_power[2, "Coef."]],
        beta_se=[NaN, coef_power[2, "Std. Error"]],
        sigma_est=[coef_const[2, "Coef."], coef_power[3, "Coef."]],
        sigma_se=[coef_const[2, "Std. Error"], coef_power[3, "Std. Error"]],
        loglik=[best_lp_const, best_lp_power],
        k=[k_const, k_power],
        AIC=[aic_const, aic_power],
        NE=[NE, NE],
        SD=[SD, SD],
        SM=[SM, SM],
        SCALE=[SCALE, SCALE],
        D_exp=[local_density_exp, local_density_exp],
        D_obs=[local_density_obs, local_density_obs],
        sigma_exp=[dispersal_rate_exp, dispersal_rate_exp],
        sigma_obs=[dispersal_rate_obs, dispersal_rate_obs],
    )
    return results
end

function main()
    NE = 5000
    SD = 0.5043275288009783  # Index 15 of 30-point log-spaced grid
    SM = 0.01
    SCALE = 1.0
    seed = 42

    mkpath(OUTPUT_DIR)

    @info "Running heterogeneous habitat simulation" NE SD SM SCALE seed
    results = simulation(NE, SD, SM, SCALE, seed)

    output_file = joinpath(OUTPUT_DIR, "het_vs_constant.csv")
    CSV.write(output_file, results)
    @info "Results saved to $output_file"
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
