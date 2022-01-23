struct Dataset
    passengers::Integer
    ids::Vector{<:Integer}
    X::Matrix{<:Real}

    function Dataset(data::DataFrame; normalize=true)
        ids = data[!, :PassengerId]
        passengers = length(ids)
        X = ones(8, passengers)

        sex = ones(Real, passengers)
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

        if normalize
            normalize!(X)
            X[1, :] .= 1
        end

        return new(passengers, ids, X)
    end

    function Dataset(data::Matrix{<:Real})
        _, n = size(data)
        return new(n, collect(1:n), data)
    end
end

function normalize!(X::Matrix{<:Real})
    X .-= mean(X; dims=2)
    X ./= std(X; dims=2)
end

(dataset::Dataset)() = dataset.X

struct Labels
    ids::Vector{<:Integer}
    survived::Vector{<:Integer}

    function Labels(data::DataFrame)
        ids = data[!, :PassengerId]
        survived = data[!, :Survived]
        labels = ones(Integer, length(survived))
        labels[survived .== 0] .= -1
        return new(ids, labels)
    end

    function Labels(data::Dataset, survived::Vector{<:Integer})
        return new(data.ids, survived)
    end

    function Labels(survived::Vector{<:Integer})
        n = length(survived)
        return new(collect(1:n), survived)
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
