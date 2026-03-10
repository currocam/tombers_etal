using CSV, DataFrames, IdentityByDescentDispersal, DelimitedFiles, CodecZlib, StatsBase

# Coarse binning scheme for plotting
bin_edges = [0.025, 0.05]
min_span = 0.01

function read_dataset(input_ibs, input_distances)
    ibd_blocks = let
        raw = readdlm(GzipDecompressorStream(open(input_ibs)), ',', '\n')
        DataFrame(
            ID1=Int.(raw[1, :]),
            ID2=Int.(raw[2, :]),
            span=Float64.(raw[3, :])
        )
    end
    individual_distances = let
        raw = readdlm(GzipDecompressorStream(open(input_distances)), ',', '\n')
        DataFrame(
            ID1=Int.(raw[1, :]),
            ID2=Int.(raw[2, :]),
            distance=Float64.(raw[3, :])
        )
    end
    ibd_blocks = ibd_blocks[ibd_blocks.span.>=min_span, :]
    ibd_blocks = ibd_blocks[ibd_blocks.span.<=maximum(bin_edges), :]
    preprocess_dataset(ibd_blocks, individual_distances, bin_edges, min_span)
end

function aggregate_by_distance(df, bin_size_km)
    df_binned = copy(df)
    df_binned.distance_bin = floor.(df_binned.DISTANCE ./ bin_size_km) .* bin_size_km
    combine(groupby(df_binned, [:distance_bin, :BIN_INDEX, :IBD_LEFT, :IBD_RIGHT, :IBD_MID])) do sub
        # Compute mean number of counts per pair
        n = sum(sub.NR_PAIRS)
        m = sum(sub.COUNT) / n
        # Compute the CI using bootstrap
        boots = zeros(1000)
        for i in 1:1000
            resampled = sub[sample(1:nrow(sub), nrow(sub), replace=true), :]
            boots[i] = sum(resampled.COUNT) / sum(resampled.NR_PAIRS)
        end
        (; mean=m, std=std(boots), n=n, lower=quantile(boots, 0.025), upper=quantile(boots, 0.975))
    end
end

df_short = read_dataset(
    "../data/input_ibd_blocks_short.txt.gz",
    "../data/input_distances_short.txt.gz"
)
df_agg_short = aggregate_by_distance(df_short, 10)
open("short_agg_10km.csv.gz", "w") do io
    stream = GzipCompressorStream(io)
    CSV.write(stream, df_agg_short)
    close(stream)
end

df_long = read_dataset(
    "../data/input_ibd_blocks_long.txt.gz",
    "../data/input_distances_long.txt.gz"
)
df_agg_long = aggregate_by_distance(df_long, 100)
open("long_agg_100km.csv.gz", "w") do io
    stream = GzipCompressorStream(io)
    CSV.write(stream, df_agg_long)
    close(stream)
end
