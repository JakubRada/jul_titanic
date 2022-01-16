module titanic

using CSV
using DataFrames
using LinearAlgebra
using Parameters
using JuMP
using OSQP

include("dataset.jl")
include("io.jl")
include("logistic.jl")
include("adaboost.jl")
include("svm.jl")

export loadtrain
export loadtest
export savepredictions

export regression
export boost
export svm
export classify

end
