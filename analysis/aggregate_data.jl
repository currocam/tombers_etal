using CSV, DataFrames, StatsBase
df = CSV.read("data.csv", DataFrame);
# Pre-process empirical data
bin_size_km = 10
df.distance_bin = floor.(df.DISTANCE ./ bin_size_km) .* bin_size_km
df_agg = combine(groupby(df, [:distance_bin, :BIN_INDEX])) do sub
    # Compute mean number of counts per pair
    n = sum(sub.NR_PAIRS)
    m = sum(sub.COUNT) / n
    # Before, CI was computed assuming normality (e.g. with the 1.96 rule)
    # However, for some cases where the number of pairs is small, the normal approximation may not be accurate.
    #lower = m - 1.96 * sqrt(m / n)
    #upper = m + 1.96 * sqrt(m / n)
    # Instead, let's compute the CI using bootstrap
    boots = zeros(1000)
    for i in 1:1000
        resampled = sub[sample(1:nrow(sub), nrow(sub), replace=true), :]
        boots[i] = sum(resampled.COUNT) / sum(resampled.NR_PAIRS)
    end
    (; mean=m, std=std(boots), n=n, lower=quantile(boots, 0.025), upper=quantile(boots, 0.975))
end
df_agg_nozero = filter(row -> row.mean > 1e-3 && row.n > 1 && row.lower > 0, df_agg)
CSV.write("aggregated_data_10km.csv", df_agg_nozero)
