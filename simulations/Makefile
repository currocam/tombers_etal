all: estimated_dispersal_constant.pdf estimated_dispersal_power.pdf

# Simulations (produces the CSV with both constant and power-law estimates)
steps/estimates_constant.csv: simulations.jl utils.jl het_habitat.slim
	julia --project=. simulations.jl

# Plots
estimated_dispersal_constant.pdf estimated_dispersal_power.pdf estimated_density_constant.pdf estimated_density_power.pdf: plot.py steps/estimates_constant.csv
	cd .. && uv run simulations/plot.py
