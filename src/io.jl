function loadtrain(path::String)
    data = CSV.read(path, DataFrame; normalizenames=true)
    return Dataset(data), Labels(data)
end

function loadtest(path::String)
    data = CSV.read(path, DataFrame; normalizenames=true)
    return Dataset(data)
end

function savepredictions(path::String, labels::Labels)
    ids, survived = out(labels)
    df = DataFrame(PassengerId = ids, Survived = survived)
    CSV.write(path, df)
end
