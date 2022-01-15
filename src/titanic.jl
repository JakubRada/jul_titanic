module titanic

using CSV
using DataFrames
using LinearAlgebra
using Parameters

include("dataset.jl")
include("io.jl")
include("logistic.jl")

export loadtrain
export loadtest
export savepredictions

export regression
export classify


end
