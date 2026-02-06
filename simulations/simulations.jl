
using Distributed

# Boiler plate code
if haskey(ENV, "SLURM_CPUS_PER_TASK")
    n_workers = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
    addprocs(n_workers)
    @info "Added $n_workers workers from SLURM allocation"
else
    addprocs(1)  # Add workers based on available cores
    @info "Running locally"
end

# Load packages on all workers
@everywhere begin
    using PyCall
    using Random
    using DataFrames
    using StatsBase
    using IdentityByDescentDispersal
    using Turing
    using CSV
    include("utils.jl")
    const SLIM_BINARY = "/scratch/antwerpen/grp/asvardal/software/slim"
    const SLIM_FILE = "het_habitat.slim"
    const OUTPUT_DIR = "steps"
end

@everywhere function simulation(NE, SD, SM, seed)
    # Create unique output path using worker ID and parameters to avoid conflicts
    worker_id = myid()
    unique_id = "$(NE)_$(SD)_$(SM)_$(seed)_w$(worker_id)"
    outpath = joinpath(OUTPUT_DIR, "het_habitat_$(unique_id).trees")
    run(
        `$SLIM_BINARY -p -s $seed -d NE=$NE -d SD=$SD -d SM=$SM -d OUTPATH="\"$outpath\"" $SLIM_FILE`,
    )

    # Data preprocessing
    tskit = pyimport("tskit")
    ts = tskit.load(outpath)
    pyslim = pyimport("pyslim")
    ts = pyslim.recapitate(ts, ancestral_Ne=NE, recombination_rate=1e-8)

    # Sample 100 diploid individuals
    rng = MersenneTwister(seed)
    n_samples = 100
    sampled = randperm(rng, ts.num_individuals)[1:n_samples] .- 1
    nodes = reduce(vcat, [ts.individual(i).nodes for i in sampled])
    ts = ts.simplify(samples=nodes)

    df_dist = let
        points = reduce(hcat, [collect(row) for row in ts.individual_locations])'
        dist_matrix = euclidean_distance(points)
        n = size(points, 1)
        df = DataFrame(ID1=Int[], ID2=Int[], distance=Float64[])
        for i = 1:n
            for j = (i+1):n
                push!(df, (i - 1, j - 1, dist_matrix[i, j]))
            end
        end
        df
    end

    # IBD blocks
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

    # Inference
    # We compare against the actual measurements we got
    local_density_obs, local_density_exp = ts.metadata["SLiM"]["user_metadata"]["OBS_D"][1], ts.metadata["SLiM"]["user_metadata"]["EXP_D"][1]
    dispersal_rate_obs, dispersal_rate_exp = ts.metadata["SLiM"]["user_metadata"]["OBS_SIGMA"][1], ts.metadata["SLiM"]["user_metadata"]["EXP_SIGMA"][1]


    @model function constant_density(df, contig_lengths)
        D ~ Uniform(0, 5000)
        σ ~ Uniform(0, 500.0)
        try
            Turing.@addlogprob! composite_loglikelihood_constant_density(
                D,
                σ,
                df,
                contig_lengths,
            )
        catch e
            @warn "Error in constant_density model: $e"
            Turing.@addlogprob! -Inf
        end
    end

    contig_lengths = [1.0]
    m = constant_density(df2, contig_lengths)

    # Run MLE from 5 different starting points and keep the best
    n_starts = 5
    best_mle = nothing
    best_lp = -Inf
    for i in 1:n_starts
        try
            # Use random initial values within the prior bounds
            init_D = rand(rng) * 5000
            init_σ = rand(rng) * 500
            mle_result = maximum_a_posteriori(m; initial_params=[init_D, init_σ])
            lp = mle_result.lp
            if lp > best_lp
                best_lp = lp
                best_mle = mle_result
            end
        catch e
            @warn "MLE attempt $i failed: $e"
        end
    end

    if best_mle === nothing
        error("All MLE attempts failed")
    end
    mle_estimate = best_mle

    # Check if estimates are at the bounds
    coef_table = mle_estimate |> coeftable |> DataFrame
    D_est = coef_table[1, "Coef."]
    σ_est = coef_table[2, "Coef."]

    bound_tol = 1e-3  # Tolerance for boundary check
    if D_est < bound_tol || D_est > 5000 - bound_tol
        error("D estimate ($D_est) is at the boundary of the prior [0, 5000]")
    end
    if σ_est < bound_tol || σ_est > 500 - bound_tol
        error("σ estimate ($σ_est) is at the boundary of the prior [0, 500]")
    end

    # Extract coefficient table with standard errors
    coef_table = mle_estimate |> coeftable |> DataFrame
    D_row = coef_table[1, :]
    σ_row = coef_table[2, :]
    return DataFrame(
        Dict(
            :D_obs => local_density_obs,
            :D_exp => local_density_exp,
            :D_est => D_row["Coef."],
            :D_std_error => D_row["Std. Error"],
            :σ_obs => dispersal_rate_obs,
            :σ_exp => dispersal_rate_exp,
            :σ_est => σ_row["Coef."],
            :σ_std_error => σ_row["Std. Error"],
            :status => "success",
            :NE => NE,
            :SD => SD,
            :SM => SM,
            :seed => seed,
            :sample_size => n_samples,
            :max_distance => maximum(df2.DISTANCE),
            :min_distance => minimum(df2.DISTANCE),
        ),
    )
end

function main()
    # Simulation parameters
    Ne_values = [500]
    kernels_sd = exp10.(range(log10(0.001), stop=log10(5.0), length=60))
    SM = 0.01
    base_seed = 1000

    mkpath(OUTPUT_DIR)

    # Generate all parameter combinations
    # Each combination gets a unique seed based on parameters
    params = [
        (NE=ne, SD=sd, SM=SM, seed=base_seed + i + j * 100) for
        (i, sd) in enumerate(kernels_sd) for (j, ne) in enumerate(Ne_values)
    ]

    @info "Starting $(length(params)) simulations across $(nprocs()-1) workers"
    @info "Parameter space: $(length(Ne_values)) NE values × $(length(kernels_sd)) SD values"

    results = pmap(params; on_error=identity) do p
        @info "Worker $(myid()): Running NE=$(p.NE), SD=$(p.SD)"
        simulation(p.NE, p.SD, p.SM, p.seed)
    end

    # Handle any errors that were returned
    valid_results = DataFrame[]
    for (i, r) in enumerate(results)
        if r isa Exception
            @error "Task $i failed with exception" exception = r
            push!(
                valid_results,
                DataFrame(
                    :NE => params[i].NE,
                    :SD => params[i].SD,
                    :SM => params[i].SM,
                    :seed => params[i].seed,
                    :D_truth => NaN,
                    :D_estimate => NaN,
                    :D_std_error => NaN,
                    :σ_truth => NaN,
                    :σ_estimate => NaN,
                    :σ_std_error => NaN,
                    :status => "exception: $(sprint(showerror, r))",
                ),
            )
        else
            push!(valid_results, r)
        end
    end

    # Write final output
    df_sims = reduce(vcat, valid_results)
    output_file = joinpath(OUTPUT_DIR, "estimates_constant.csv")
    CSV.write(output_file, df_sims)
    @info "Results saved to: $output_file"
    @info "Completed $(sum(df_sims.status .== "success")) / $(nrow(df_sims)) simulations successfully"
    return df_sims
end

# Run if executed as main script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
