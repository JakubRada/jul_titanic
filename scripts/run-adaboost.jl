using Pkg
Pkg.activate(pwd())

using titanic

traindata, trainlabels = loadtrain("../data/train.csv"; normalize=false)
testdata = loadtest("../data/test.csv"; normalize=false)

H = boost(traindata, trainlabels)

prediction = classify(testdata, H)

savepredictions("../results/predictions-adaboost.csv", prediction)
