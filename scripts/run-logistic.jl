using Pkg
Pkg.activate(pwd())

using Revise
using titanic

traindata, trainlabels = loadtrain("../data/train.csv")
testdata = loadtest("../data/test.csv")

w = regression(traindata, trainlabels)

prediction = classify(testdata, w)

savepredictions("../results/predictions-logisticregress.csv", prediction)
