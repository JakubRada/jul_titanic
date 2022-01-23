function svm(dataset::Dataset, labels::Labels; C::Real = 10.0)
    X = dataset()
    y = labels()

    alpha = dual(X, y, C)

    return alpha
end

function linearkernel(xi::Vector{<:Real}, xj::Vector{<:Real})
    return dot(xi, xj)
end

function polynomialkernel(xi::Vector{<:Real}, xj::Vector{<:Real}; degree::Integer = 1)
    return (1 + dot(xi, xj))^degree
end

function gaussiankernel(xi::Vector{<:Real}, xj::Vector{<:Real}; variance::Real = 1.0)
    return exp(-dot(xi - xj, xi - xj) / (2 * variance))
end

function kernel(xi::Vector{<:Real}, xj::Vector{<:Real})
    return linearkernel(xi, xj)
    # return guassiankernel(xi, xj)
    # return polynomialkernel(xi, xj; degree=2)
end

function dual(X::Matrix{<:Real}, y::Vector{<:Integer}, C::Real)
    _, n = size(X)

    qp = Model(OSQP.Optimizer)
    set_silent(qp)

    @variable(qp, alpha[1:n])
    @objective(qp, Max, sum(alpha) - sum([alpha[i] * alpha[j] * y[i] * y[j] * kernel(X[:, i], X[:, j]) for i in 1:n for j in 1:n]) / 2)
    @constraint(qp, dot(alpha, y) == 0)
    @constraint(qp, 0 .<= alpha .<= C)

    optimize!(qp)

    return value.(alpha)
end

function classify(x::Vector{<:Real}, traindata::Dataset, trainlabels::Labels, alpha::Vector{<:Real})
    X = traindata()
    y = trainlabels()
    n = length(y)
    f = sum([alpha[i] * y[i] * kernel(X[:, i], x) for i in 1:n])
    return f >= 0 ? 1 : -1
end

function classify(test::Dataset, traindata::Dataset, trainlabels::Labels, alpha::Vector{<:Real})
    X = test()
    _, n = size(X)
    return Labels(test, [classify(X[:, i], traindata, trainlabels, alpha) for i in 1:n])
end
