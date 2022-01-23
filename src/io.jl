function loadtrain(path::String; kwargs...)
    data = CSV.read(path, DataFrame; normalizenames=true)
    return Dataset(data; kwargs...), Labels(data)
end

function loadtest(path::String; kwargs...)
    data = CSV.read(path, DataFrame; normalizenames=true)
    return Dataset(data; kwargs...)
end

function savepredictions(path::String, labels::Labels)
    ids, survived = out(labels)
    df = DataFrame(PassengerId = ids, Survived = survived)
    CSV.write(path, df)
end
