struct Dataset
    passengers::Int64
    ids::Vector{Int64}
    X::Matrix{Float64}

    function Dataset(data::DataFrame)
        ids = data[!, :PassengerId]
        passengers = length(ids)
        X = zeros(passengers, 7)

        sex = ones(Float64, passengers)
        sex[data[!, :Sex] .== "female"] .= -1
        X[:, 1] = sex

        X[:, 2] = data[!, :Pclass]

        age = data[!, :Age]
        avgage = round(sum(skipmissing(age)) / passengers; digits=2)
        age[ismissing.(age)] .= avgage
        X[:, 3] = age

        X[:, 4] = data[!, :SibSp]
        X[:, 5] = data[!, :Parch]
        X[:, 6] = data[!, :Fare]

        embarked = data[!, :Embarked]
        embarkedprime = skipmissing(embarked)
        miss = argmax([sum(embarkedprime .== "C"), sum(embarkedprime .== "Q"), sum(embarkedprime .== "S")])
        embarked[ismissing.(embarked)] .= "M"
        X[embarked .== "M", 7] .= miss
        X[embarked .== "C", 7] .= 1
        X[embarked .== "Q", 7] .= 2
        X[embarked .== "S", 7] .= 3

        return new(passengers, ids, X)
    end
end

(dataset::Dataset)() = dataset.X

struct Labels
    ids::Vector{Int64}
    survived::Vector{Int64}

    function Labels(data::DataFrame)
        ids = data[!, :PassengerId]
        survived = data[!, :Survived]
        labels = ones(Int64, length(survived))
        labels[survived .== 0] .= -1
        return new(ids, labels)
    end

    function Labels(ids::Vector{Int64})
        return new(ids, zeros(Int64, length(ids)))
    end
end

(labels::Labels)() = labels.survived

function out(labels::Labels)
    survived = copy(labels.survived)
    survived[survived .== -1] .= 0
    return labels.ids, survived
end
