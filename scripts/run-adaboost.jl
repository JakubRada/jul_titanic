using Pkg
Pkg.activate(pwd())

using titanic

traindata, trainlabels = loadtrain("../data/train.csv")
testdata = loadtest("../data/test.csv")

H = boost(traindata, trainlabels)

prediction = classify(testdata, H)

savepredictions("../results/predictions-adaboost.csv", prediction)
