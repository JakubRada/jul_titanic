"""
    Dataset

structure holding inner representation of input data samples and provides replacement of missing values so they can be used by universal solver

# Fields
- `passengers`: number of data samples present
- `ids`: id numbers assigned to individual samples to distinguish them easily
- `X`: matrix of real numbers where one column represents features of a single data sample
"""
struct Dataset
    passengers::Integer
    ids::Vector{<:Integer}
    X::Matrix{<:Real}

    """
        Dataset(data::DataFrame; normalize=true)

    constucts Dataset object from `DataFrame`

    when keyword argument `normalize` is set to ``true``, the data are normalized for better function of some classifiers
    """
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

    """
        Dataset(data::Matrix{<:Real})

    construct Dataset object when the data are already prepared in matrix `data`

    skips normalization and filling of missing values
    """
    function Dataset(data::Matrix{<:Real})
        _, n = size(data)
        return new(n, collect(1:n), data)
    end
end

"""
    normalize!(X::Matrix{<:Real})

normalize data in `X` in place by subtracting the mean and dividing by standard deviation of the samples
"""
function normalize!(X::Matrix{<:Real})
    X .-= mean(X; dims=2)
    X ./= std(X; dims=2)
end

"""
    (dataset::Dataset)() = dataset.X

functor to return data matrix from `dataset` without the need to access the field directly
"""
(dataset::Dataset)() = dataset.X

"""
    Labels

structure holding inner representation of labels assigned to given dataset (it can be both training or testing dataset)

# Fields
- `ids`: id numbers assigned to samples; needed for output in Kaggle specified format
- `survived`: labels of given samples (-1 for negative class, 1 for positive class)
"""
struct Labels
    ids::Vector{<:Integer}
    survived::Vector{<:Integer}

    """
        Labels(data::DataFrame)

    extract correct labels from given `data` DataFrame
    """
    function Labels(data::DataFrame)
        ids = data[!, :PassengerId]
        survived = data[!, :Survived]
        labels = ones(Integer, length(survived))
        labels[survived .== 0] .= -1
        return new(ids, labels)
    end

    """
        Labels(data::DataFrame, survived::Vector{<:Integer})

    extract `ids` from DataFrame and assign them classes from `survived` vector
    """
    function Labels(data::Dataset, survived::Vector{<:Integer})
        return new(data.ids, survived)
    end

    """
        Labels(survived::Vector{<:Integer})

    wrap classes from `survived` vector into Labels object

    `ids` are set to increasing sequence starting from 1
    """
    function Labels(survived::Vector{<:Integer})
        n = length(survived)
        return new(collect(1:n), survived)
    end
end

"""
    (labels::Labels)() = labels.survived

functor to return labels vector from `Labels` without the need to access the field directly
"""
(labels::Labels)() = labels.survived

"""
    classificationerror(correct::Labels, prediction::Labels)

compute ratio of misclassified data samples from `prediction` when given `correct` labels
"""
function classificationerror(correct::Labels, prediction::Labels)
    return sum(correct.survived .!= prediction.survived) / length(correct.survived)
end

"""
    out(labels::Labels)

transform input representation of labels into required one by Kaggle (i.e. transofrm -1 to 0 label)
"""
function out(labels::Labels)
    survived = copy(labels.survived)
    survived[survived .== -1] .= 0
    return labels.ids, survived
end
