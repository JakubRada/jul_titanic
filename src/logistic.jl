"""
    regression(data::Dataset, labels::Labels; kwargs...)

perform learning of logistic regression classifier on given `data` and correct `labels`

outputs vector `w` which detrmines a hyperplane optimally separating the data samples

- `kwargs` can contain `epsilon` to specify accuracy of optimization method and initial `step` size
"""
function regression(data::Dataset, labels::Labels; kwargs...)
    trainX = data()
    trainy = labels()

    w = gd(trainX, trainy; kwargs...)

    return w
end

"""
    classify(data::Dataset, w::Vector{<:Real})

classify the `data` using separating hyperplane determined by `w`
"""
function classify(data::Dataset, w::Vector{<:Real})
    X = data()
    _, n = size(X)
    return Labels(data, [dot(X[:, i], w) >= 0 ? 1 : -1 for i in 1:n])
end

"""
    computeE(X::Matrix{<:Real}, k::Vector{<:Integer}, w::Vector{<:Real})

returns energy of hyperplane `w` with respect to data `X` and labels `k`; this objective is to be minimized
"""
function computeE(X::Matrix{<:Real}, k::Vector{<:Integer}, w::Vector{<:Real})
    N = length(k)
    return sum([log(1 + exp(-k[i] * dot(X[:, i], w))) for i in 1:N]) ./ N
end

"""
    gradE(X::Matrix{<:Real}, k::Vector{<:Integer}, w::Vector{<:Real})

returns gradient of energy with respect to vector w
"""
function gradE(X::Matrix{<:Real}, k::Vector{<:Integer}, w::Vector{<:Real})
    N = length(k)
    return - sum([(k[i] * X[:, i]) / (1 + exp(k[i] * dot(X[:, i], w))) for i in 1:N]; dims=1)[1] ./ N
end

"""
    l2norm(x)

computes l2-norm of given vector `x`
"""
function l2norm(x)
    return sqrt.(dot(x, x))
end

"""
    gd(X::Matrix{<:Real}, k::Vector{<:Integer}; epsilon=1e-4, step=1.0)

performs gradient descent method to find optimal separating hyperplane `w` on data `X` and labels `k`

keyword arguments can be used to alter the optimization:
- `epsilon` to specify accuracy/terminal condition
- `step` to adjust initial step size
"""
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
