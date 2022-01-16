struct WeakClassifier
    feature::Int64
    threshold::Float64
    bigger::Bool
    weight::Float64

    function WeakClassifier(feature::Int64, threshold::Float64, bigger::Bool, error::Float64)
        weight = log((1 - error) / error) / 2
        return new(feature, threshold, bigger, weight)
    end

    function WeakClassifier(feature::Int64, threshold::Float64, bigger::Bool)
        return new(feature, threshold, bigger, 0.0)
    end

end

function Base.show(io::IO, h::WeakClassifier)
    print(io, "h: x[$(h.feature)] $(h.bigger ? ">=" : "<") $(h.threshold) [α = $(h.weight)]")
end

struct StrongClassifier
    weights::Vector{Float64}
    weaks::Vector{WeakClassifier}
    Z::Vector{Float64}

    samples::Int64

    function StrongClassifier(samples::Int64)
        initial = 1 / samples
        return new(fill(initial, samples), Vector{WeakClassifier}(undef, 0), zeros(0), samples)
    end
end

function Base.show(io::IO, H::StrongClassifier)
    println(io, "=== Strong classifier H ===")
    for i in 1:length(H.weaks)
        println("$(H.weaks[i]) ... Z = $(H.Z[i])")
    end
end

function (h::WeakClassifier)(x::Vector{Float64})
    value = x[h.feature]
    if h.bigger
        return value >= h.threshold ? 1 : -1
    else
        return value < h.threshold ? 1 : -1
    end
end

function (H::StrongClassifier)(x::Vector{Float64})
    return Int64(sign(sum([h.weight * h(x) for h in H.weaks])))
end

function updateWeights!(H::StrongClassifier, h::WeakClassifier, X::Matrix{Float64}, y::Vector{Int64})
    newweights = [H.weights[i] * exp(-h.weight * y[i] * h(X[:, i])) for i in 1:size(X, 2)]
    Z = sum(newweights)
    push!(H.Z, Z * (length(H.Z) > 0 ? H.Z[end] : 1))
    H.weights .= newweights / Z
end

function bestWeak(H::StrongClassifier, X::Matrix{Float64}, ranges::Vector{Tuple{Float64, Float64}}, y::Vector{Int64})
    dim, n = size(X)
    opterror = Inf64
    local opth

    for feature in 2:dim
        l, u = ranges[feature]
        for threshold in l:0.5:u
            h = WeakClassifier(feature, threshold, true)
            comp = [h(X[:, i]) for i in 1:n]
            diff = comp .!= y
            err = dot(diff, H.weights)
            if err <= opterror
                opterror = err
                opth = h
            end

            h = WeakClassifier(feature, threshold, false)
            comp = [h(X[:, i]) for i in 1:n]
            diff = comp .!= y
            err = dot(diff, H.weights)
            if err <= opterror
                opterror = err
                opth = h
            end
        end
    end

    return WeakClassifier(opth.feature, opth.threshold, opth.bigger, opterror), opterror
end

addWeak!(H::StrongClassifier, h::WeakClassifier) = push!(H.weaks, h)

function boost(data::Dataset, labels::Labels; limit::Int64 = 100)
    X = data()
    y = labels()

    n = data.passengers

    H = StrongClassifier(n)

    for _ in 1:limit
        h, error = bestWeak(H, X, getranges(data), y)

        if error >= 0.5
            break
        end

        addWeak!(H, h)

        updateWeights!(H, h, X, y)

    end

    return H
end

function classify(data::Dataset, H::StrongClassifier)
    return Labels(data, [H(data()[:, i]) for i in 1:data.passengers])
end
