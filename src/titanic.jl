module titanic

using CSV
using DataFrames
using JuMP
using LinearAlgebra
using OSQP
using Parameters
using Statistics

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
export classificationerror

end
