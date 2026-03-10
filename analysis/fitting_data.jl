using CSV, DataFrames, IdentityByDescentDispersal, DelimitedFiles, CodecZlib

# Binning scheme (fitting MLE only)
min_span = 0.01 # 1 cM
max_span = 0.05 # 5 cM
bin_width = 0.001 # 0.1 cM

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
    bin_edges = (min_span+bin_width):bin_width:max_span
    ibd_blocks = ibd_blocks[ibd_blocks.span.>=min_span, :]
    ibd_blocks = ibd_blocks[ibd_blocks.span.<=max_span, :]
    preprocess_dataset(ibd_blocks, individual_distances, bin_edges, min_span)
end

df_short = read_dataset(
    "../data/input_ibd_blocks_short.txt.gz",
    "../data/input_distances_short.txt.gz"
)
df_long = read_dataset(
    "../data/input_ibd_blocks_long.txt.gz",
    "../data/input_distances_long.txt.gz"
)

open("short_data.csv.gz", "w") do io
    stream = GzipCompressorStream(io)
    CSV.write(stream, df_short)
    close(stream)
end
open("long_data.csv.gz", "w") do io
    stream = GzipCompressorStream(io)
    CSV.write(stream, df_long)
    close(stream)
end
