struct Dataset
    passengers::Int64
    ids::Vector{Int64}
    X::Matrix{Float64}
    ranges::Vector{Tuple{Float64, Float64}}

    function Dataset(data::DataFrame)
        ids = data[!, :PassengerId]
        passengers = length(ids)
        X = ones(8, passengers)

        sex = ones(Float64, passengers)
        sex[data[!, :Sex] .== "female"] .= -1
        X[2, :] = sex

        X[3, :] = data[!, :Pclass]

        age = data[!, :Age]
        avgage = round(sum(skipmissing(age)) / (passengers - sum(ismissing.(age))); digits=2)
        age[ismissing.(age)] .= avgage
        X[4, :] = age

        X[5, :] = data[!, :SibSp]
        X[6, :] = data[!, :Parch]
        fare = data[!, :Fare]
        avgfare = round(sum(skipmissing(fare)) / (passengers - sum(ismissing.(fare))); digits=2)
        fare[ismissing.(fare)] .= avgfare
        X[7, :] = data[!, :Fare]

        embarked = data[!, :Embarked]
        embarkedprime = skipmissing(embarked)
        miss = argmax([sum(embarkedprime .== "C"), sum(embarkedprime .== "Q"), sum(embarkedprime .== "S")])
        embarked[ismissing.(embarked)] .= "M"
        X[8, embarked .== "M"] .= miss
        X[8, embarked .== "C"] .= 1
        X[8, embarked .== "Q"] .= 2
        X[8, embarked .== "S"] .= 3

        ranges = [(minimum(x), maximum(x)) for x in eachrow(X)]

        return new(passengers, ids, X, ranges)
    end
end

(dataset::Dataset)() = dataset.X

function getranges(dataset::Dataset)
    return dataset.ranges
end

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

    function Labels(data::Dataset, survived::Vector{Int64})
        return new(data.ids, survived)
    end
end

(labels::Labels)() = labels.survived

function classificationerror(correct::Labels, prediction::Labels)
    return sum(correct.survived .!= prediction.survived) / length(correct.survived)
end

function out(labels::Labels)
    survived = copy(labels.survived)
    survived[survived .== -1] .= 0
    return labels.ids, survived
end
