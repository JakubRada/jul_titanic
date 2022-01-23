"""
    Weak classifier `h`

structure representing a weak classifier which selects class for a measurement with given weight

# Fields
- `feature`: index in a feature vector defining by which part the decision is made
- `threshold`: separating value for given `feature`
- `bigger`: if set to ``true``, the feature value is compared by ``≥`` with the `threshold`, otherwise by `<`
- `weight`: how much this weak classifier contributes to overall decision
"""
struct WeakClassifier
    feature::Integer
    threshold::Real
    bigger::Bool
    weight::Real

    """
        WeakClassifier(feature::Integer, threshold::Real, bigger::Bool, error::Real)

    constructs weak classifer, which is meant to be saved into strong classifier

    how many data samples were misclassified is given by `error` argument, which determines the `weight` of the weak classifier
    """
    function WeakClassifier(feature::Integer, threshold::Real, bigger::Bool, error::Real)
        weight = log((1 - error) / error) / 2
        return new(feature, threshold, bigger, weight)
    end

    """
        WeakClassifier(feature::Integer, threshold::Real, bigger::Bool, error::Real)

    constructs weak classifer, which is meant for finding the best weak classifier on current data and does not incorporate the `weight calculation`
    """
    function WeakClassifier(feature::Integer, threshold::Real, bigger::Bool)
        return new(feature, threshold, bigger, 0.0)
    end
end

function Base.show(io::IO, h::WeakClassifier)
    print(io, "h: x[$(h.feature)] $(h.bigger ? ">=" : "<") $(h.threshold) [α = $(h.weight)]")
end

"""
    (h::WeakClassifier)(x::Vector{<:Real})

classifies one data sample `x` using weak classifier `h`

output is either `1`, if the sample satisfies the classifiers condition, or `-1`
"""
function (h::WeakClassifier)(x::Vector{<:Real})
    value = x[h.feature]
    if h.bigger
        return value >= h.threshold ? 1 : -1
    else
        return value < h.threshold ? 1 : -1
    end
end

"""
    Strong classifier `H`

structure representing adaboost classifier that selects class of a given sample using mulititude of weighted weak classifiers

# Fields
- `weights` - determine how important is correct classification of a given sample; initially set to 1/N
- `weaks` - set of weak classifiers used to making decisions
- `Z` - upper bound on classification error in individual steps of learning
- `samples` - number of data samples used for learning
"""
struct StrongClassifier
    weights::Vector{<:Real}
    weaks::Vector{WeakClassifier}
    Z::Vector{<:Real}
    samples::Integer

    """
        StrongClassifier(samples::Integer)

        constructs empty strong classifier with uniform initial data weights
    """
    function StrongClassifier(samples::Integer)
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

"""
    (H::StrongClassifier)(x::Vector{<:Real})

classify given data sample `x` using strong classifier `H`

output is either `1` or `-1` based on weighted decisions of all weak classifiers
"""
function (H::StrongClassifier)(x::Vector{<:Real})
    return Integer(sign(sum([h.weight * h(x) for h in H.weaks])))
end

"""
    updateWeights!(H::StrongClassifier, h::WeakClassifier, X::Matrix{<:Real}, y::Vector{<:Integer})

update data weights based on correct labels `y` and those assigned by new weak classifier `h`
- increase weight for misclassified samples and decrease for correctly classified ones
- recompute error upper bound Z
"""
function updateWeights!(H::StrongClassifier, h::WeakClassifier, X::Matrix{<:Real}, y::Vector{<:Integer})
    newweights = [H.weights[i] * exp(-h.weight * y[i] * h(X[:, i])) for i in 1:size(X, 2)]
    Z = sum(newweights)
    push!(H.Z, Z * (length(H.Z) > 0 ? H.Z[end] : 1))
    H.weights .= newweights / Z
end

"""
    bestWeak(H::StrongClassifier, X::Matrix{<:Real}, Xsorted::Matrix{<:Real}, y::Vector{<:Integer})

select a weak classifier that misclassifies the least amount of samples in current state of data weights

`Xsorted` - matrix X with presorted rows to try different `thresholds`

returns optimal weak classifier and its classification error
"""
function bestWeak(H::StrongClassifier, X::Matrix{<:Real}, Xsorted::Matrix{<:Real}, y::Vector{<:Integer})
    dim, n = size(X)
    opterror = Inf64
    local opth

    for feature in 2:dim
        x = Xsorted[feature, :]
        for threshold in x
            h = WeakClassifier(feature, threshold, true)
            comp = [h(X[:, i]) for i in 1:n]
            diff = comp .!= y
            err = dot(diff, H.weights)
            if err < opterror
                opterror = err
                opth = h
            end

            h = WeakClassifier(feature, threshold, false)
            comp = [h(X[:, i]) for i in 1:n]
            diff = comp .!= y
            err = dot(diff, H.weights)
            if err < opterror
                opterror = err
                opth = h
            end
        end
    end

    return WeakClassifier(opth.feature, opth.threshold, opth.bigger, opterror), opterror
end

"""
    boost(data::Dataset, labels::Labels; limit::Integer = 100)

select at most `limit` weak classifiers for given dataset `data` and its correct `labels` 
"""
function boost(data::Dataset, labels::Labels; limit::Integer = 100)
    X = data()
    y = labels()

    sorted = vcat([sort(row) for row in eachrow(X)]'...)

    n = data.passengers

    H = StrongClassifier(n)

    for _ in 1:limit
        h, error = bestWeak(H, X, sorted, y)

        if error >= 0.5
            break
        end

        push!(H.weaks, h)

        updateWeights!(H, h, X, y)

    end

    return H
end

"""
    classify(data::Dataset, H::StrongClassifier)

classify samples in `data` dataset with strong classifier `H`
"""
function classify(data::Dataset, H::StrongClassifier)
    return Labels(data, [H(data()[:, i]) for i in 1:data.passengers])
end
