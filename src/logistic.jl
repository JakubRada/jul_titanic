function regression()
    data, labels = loadtrain("./data/train.csv")
    savepredictions("./data/predictions.csv", labels)
    return data.X
end

