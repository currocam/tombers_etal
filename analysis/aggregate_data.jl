using CSV, DataFrames, StatsBase

function aggregate_by_distance(df, bin_size_km)
    df_binned = copy(df)
    df_binned.distance_bin = floor.(df_binned.DISTANCE ./ bin_size_km) .* bin_size_km
    df_agg = combine(groupby(df_binned, [:distance_bin, :BIN_INDEX, :IBD_LEFT, :IBD_RIGHT, :IBD_MID])) do sub
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
    filter(row -> row.mean > 1e-3 && row.n > 1 && row.lower > 0, df_agg)
end

df_short = CSV.read("short_data.csv", DataFrame)
# Short scale: aggregate by 10 km
df_agg_short = aggregate_by_distance(df_short, 10)
CSV.write("short_agg_10km.csv", df_agg_short)

# Long scale: aggregate by 100 km
df_long = CSV.read("long_data.csv", DataFrame)
df_agg_long = aggregate_by_distance(df_long, 100)
CSV.write("long_agg_100km.csv", df_agg_long)

