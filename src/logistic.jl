function regression(data::Dataset, labels::Labels)
    trainX = data()
    trainy = labels()

    w = gd(trainX, trainy)

    return w
end

function classify(data::Dataset, w::Vector{<:Real})
    X = data()
    _, n = size(X)
    return Labels(data, [dot(X[:, i], w) >= 0 ? 1 : -1 for i in 1:n])
end

function computeE(X::Matrix{<:Real}, k::Vector{<:Integer}, w::Vector{<:Real})
    N = length(k)
    return sum([log(1 + exp(-k[i] * dot(X[:, i], w))) for i in 1:N]) ./ N
end

function gradE(X::Matrix{<:Real}, k::Vector{<:Integer}, w::Vector{<:Real})
    N = length(k)
    return - sum([(k[i] * X[:, i]) / (1 + exp(k[i] * dot(X[:, i], w))) for i in 1:N]; dims=1)[1] ./ N
end

function l2norm(x)
    return sqrt.(dot(x, x))
end

function gd(X::Matrix{<:Real}, k::Vector{<:Integer}; epsilon=1e-4, step=1.0)
    dim, _ = size(X)

    w = zeros(Float64, dim)
    lastw = fill(Inf64, dim)
    E = Inf64

    while l2norm(w - lastw) > epsilon
        grad = gradE(X, k, w)
        neww = w - step * grad
        newE = computeE(X, k, neww)

        if newE < E
            step *= 2
            E = newE
            lastw = w
            w = neww
        else
            step /= 2
        end

    end

    return w
end
