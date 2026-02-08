function euclidean_distance(points)
    coords = points[:, 1:2]
    n = size(coords, 1)
    dist_matrix = zeros(n, n)
    for i = 1:n
        for j = 1:n
            dx = coords[i, 1] - coords[j, 1]
            dy = coords[i, 2] - coords[j, 2]
            dist_matrix[i, j] = sqrt(dx^2 + dy^2)
        end
    end
    return dist_matrix
end

function mean_axial_distance(points)
    coords = points[:, 1:2]
    n = size(coords, 1)
    acc = 0.0
    counter = 0
    for i = 1:n
        for j = i+1:n
            dx = coords[i, 1] - coords[j, 1]
            dy = coords[i, 2] - coords[j, 2]
            acc += sqrt(dx^2) + sqrt(dy^2)
            counter += 2
        end
    end
    return acc / counter
end

function cut(values, edges)
    n = length(values)
    bins = Vector{Float64}(undef, n)
    for i = 1:n
        x = values[i]
        bin_idx = findfirst(j -> edges[j] <= x < edges[j+1], 1:(length(edges)-1))
        @assert !isnothing(bin_idx)
        bins[i] = edges[bin_idx]
    end
    return bins
end
