using Pkg
Pkg.activate(pwd())

using titanic

traindata, trainlabels = loadtrain("../data/train.csv")
testdata = loadtest("../data/test.csv")

alpha = svm(traindata, trainlabels)

prediction = classify(testdata, traindata, trainlabels, alpha)

savepredictions("../results/predictions-svm.csv", prediction)
