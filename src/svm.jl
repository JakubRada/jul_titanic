"""
    svm(dataset::Dataset, labels::Labels; C::Real = 10.0)

perform suppor vector machines learning on `dataset` and `labels`

returns optimal vector alpha which defines which samples are suppor vectors

- keyword argument `C` defines penalty for misclassified points in soft-margin SVM
"""
function svm(dataset::Dataset, labels::Labels; C::Real = 10.0)
    X = dataset()
    y = labels()

    alpha = dual(X, y, C)

    return alpha
end

"""
    linearkernel(xi::Vector{<:Real}, xj::Vector{<:Real})

simplest svm kernel using a simple dot product
"""
function linearkernel(xi::Vector{<:Real}, xj::Vector{<:Real})
    return dot(xi, xj)
end

"""
    polynomialkernel(xi::Vector{<:Real}, xj::Vector{<:Real}; degree::Integer = 1)

polynomial kernel with variable `degree` of the polynomial
"""
function polynomialkernel(xi::Vector{<:Real}, xj::Vector{<:Real}; degree::Integer = 1)
    return (1 + dot(xi, xj))^degree
end

"""
    gaussiankernel(xi::Vector{<:Real}, xj::Vector{<:Real}; variance::Real = 1.0)

kernel using normal distribution and can be parametrized by `variance`
"""
function gaussiankernel(xi::Vector{<:Real}, xj::Vector{<:Real}; variance::Real = 1.0)
    return exp(-dot(xi - xj, xi - xj) / (2 * variance))
end

"""
    kernel(xi::Vector{<:Real}, xj::Vector{<:Real})

kernel wrapper to select from prepared kernels without the need to alter more complicated functions
"""
function kernel(xi::Vector{<:Real}, xj::Vector{<:Real})
    return linearkernel(xi, xj)
    # return guassiankernel(xi, xj)
    # return polynomialkernel(xi, xj; degree=2)
end

"""
    dual(X::Matrix{<:Real}, y::Vector{<:Integer}, C::Real)

finds optimal alpha by solving the dual formulation using quadratic programming solver
"""
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

"""
    classify(x::Vector{<:Real}, traindata::Dataset, trainlabels::Labels, alpha::Vector{<:Real})

classify one data sample `x` with given `alpha` weights
"""
function classify(x::Vector{<:Real}, traindata::Dataset, trainlabels::Labels, alpha::Vector{<:Real})
    X = traindata()
    y = trainlabels()
    n = length(y)
    f = sum([alpha[i] * y[i] * kernel(X[:, i], x) for i in 1:n])
    return f >= 0 ? 1 : -1
end

"""
    classify(test::Dataset, traindata::Dataset, trainlabels::Labels, alpha::Vector{<:Real})

classify the whole dataset `test` with given `alpha` weights and `traindata`, `trainlabels`
"""
function classify(test::Dataset, traindata::Dataset, trainlabels::Labels, alpha::Vector{<:Real})
    X = test()
    _, n = size(X)
    return Labels(test, [classify(X[:, i], traindata, trainlabels, alpha) for i in 1:n])
end
