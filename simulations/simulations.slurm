#!/bin/bash
#SBATCH --job-name=large_scale
#SBATCH --output=logs/large_scale_%j.out
#SBATCH --error=logs/large_scale_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

module load Julia/1.12.4 foss/2025a
julia --project=. -p auto simulations.jl
