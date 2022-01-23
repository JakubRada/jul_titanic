"""
    loadtrain(path::String; kwargs...)

load training set on `path` with `normalize` option in kwargs; output Dataset struture
"""
function loadtrain(path::String; kwargs...)
    data = CSV.read(path, DataFrame; normalizenames=true)
    return Dataset(data; kwargs...), Labels(data)
end

"""
    loadtest(path::String; kwargs...)

load testing set on `path` with `normalize` option in kwargs; output Dataset struture
"""
function loadtest(path::String; kwargs...)
    data = CSV.read(path, DataFrame; normalizenames=true)
    return Dataset(data; kwargs...)
end

"""
    savepredictions(path::String, labels::Labels)

save `labels` predicted by a classifier into file on `path` in correct format
"""
function savepredictions(path::String, labels::Labels)
    ids, survived = out(labels)
    df = DataFrame(PassengerId = ids, Survived = survived)
    CSV.write(path, df)
end
