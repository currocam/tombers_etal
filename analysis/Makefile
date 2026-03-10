RAW = ../data/input_ibd_blocks_short.txt.gz \
      ../data/input_distances_short.txt.gz \
      ../data/input_ibd_blocks_long.txt.gz \
      ../data/input_distances_long.txt.gz

all: constant/predictions_short.pdf power/predictions_short.pdf

# Data preparation
short_data.csv.gz long_data.csv.gz: fitting_data.jl $(RAW)
	julia --project=. fitting_data.jl

short_agg_10km.csv.gz long_agg_100km.csv.gz: plotting_data.jl $(RAW)
	julia --project=. plotting_data.jl

# MLE fitting
constant/short_mle.csv constant/long_mle.csv: constant/constant.jl short_data.csv.gz long_data.csv.gz
	cd constant && julia --project=.. -t4 constant.jl

power/short_mle.csv power/long_mle.csv: power/power.jl short_data.csv.gz long_data.csv.gz
	cd power && julia --project=.. -t4 power.jl

# Predictions
constant/%_predictions.csv: constant/predictions.jl constant/%_mle.csv short_agg_10km.csv.gz long_agg_100km.csv.gz
	cd constant && julia --project=.. predictions.jl

power/%_predictions.csv: power/predictions.jl power/%_mle.csv short_agg_10km.csv.gz long_agg_100km.csv.gz
	cd power && julia --project=.. -t4 predictions.jl

# Plots
constant/%.pdf: constant/plot.py constant/short_predictions.csv constant/long_predictions.csv
	cd .. && uv run analysis/constant/plot.py

power/%.pdf: power/plot.py power/short_predictions.csv power/long_predictions.csv
	cd .. && uv run analysis/power/plot.py
