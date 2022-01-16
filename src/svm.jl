function svm(dataset::Dataset, labels::Labels; C::Float64 = 10.0)
    X = dataset()
    y = labels()

    alpha = dual(X, y, C)

    return alpha
end

function kernel(xi::Vector{Float64}, xj::Vector{Float64})
    return dot(xi, xj)
end

function dual(X::Matrix{Float64}, y::Vector{Int64}, C::Float64)
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

function classify(x::Vector{Float64}, traindata::Dataset, trainlabels::Labels, alpha::Vector{Float64})
    X = traindata()
    y = trainlabels()
    n = length(y)
    f = sum([alpha[i] * y[i] * kernel(X[:, i], x) for i in 1:n])
    return f >= 0 ? 1 : -1
end

function classify(test::Dataset, traindata::Dataset, trainlabels::Labels, alpha::Vector{Float64})
    X = test()
    _, n = size(X)
    return Labels(test, [classify(X[:, i], traindata, trainlabels, alpha) for i in 1:n])
end
