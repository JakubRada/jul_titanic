# # titanic
# Titanic is one of many competitions available through the [Kaggle](https://www.kaggle.com/c/titanic) platform.
# Its goal is to train a classifier which recognizes people who survived the Titanic catastrophe based on some known characteristics, such as age, gender, etc.
# This package provides some possible methods which (at least up to some precision) solve this problem.
#
# ## Input data
# Before we describe the used methods, we need to prepare the data.

# First we need to load the package

using titanic

# Then we can load training data from provided datasets. (can be downloaded on [Kaggle](https://www.kaggle.com/c/titanic/data))

traindata, trainlabels = loadtrain("../data/train.csv")

# Note that this function call performs normalization on the data, so for some classificators later (adaboost), we may turn this procedure off with `normalize=false` keyword parameter.

# Now we do almost the same for testing data.
# Here, Kaggle does not provide labels for the testing dataset and the classificator predictions need to be uploaded to their webpage to receive evaluation.

testdata = loadtest("../data/test.csv")

# The datasets were loaded from `.csv` files into inner `Dataset` structures.
# These provide an useful abstraction, so one does not have to operate on matrices and vectors, but it usually suffices to pass these structures.
# As was said before, the constructor of this structure automacially normalizes the data so they are more suitable for most classifiers.
# Also, many of these methods need to add so-called 'bias' term into the feature vector.
# This is conducted in this structure as well by adding $1$ to the beginning of each point in the data matrix.
#
# The same can be said about `trainlabels`, which are stored inside `Labels` structure.
#
# Both of these structures can return their main field when called as functions (or functors).

traindata()
#-
trainlabels()

# ## Logistic regression
# As first and most basic classifier, we use Logistic regression.
#
# This classifier searches for a vector `w` which determines a hyperplane in the feature space that optimally separates the data points.
# However, the data are most likely not linearly separable, so there will be some some misclassified points.
#
# This search occurs in high-level function

w = regression(traindata, trainlabels; epsilon=1e-4, step=1.0)

# The keyword parameters `epsilon` and `step` are optional and default to values in the example.
# `epsilon` indicates precision for the underlying optimizer and `step` determines coefficient of initial step.
#
# As the method for finding the optimal `w` was implemented gradient descent with adaptive stepsize.
# In each iteration, the gradient descent computes so called *negative log-likelihood*
# $$ E(w) = \sum_{x}{\log(1 + e^{-kxw})} $$
# , where $k$ is the class of $x$, and its derivation
# $$ \frac{\partial E(w)}{\partial w} = - \sum_x\frac{kx}{1 + e^{kxw}} $$
# Using this derivation we find the w with minimal *negative log-likelihood* (or *cross entropy*).
#
# The adaptive step size behaves in a way that when the current step size finds better solution, we double the step in the next iteration.
# On the other hand, when the solution is worse (meaning it "jumped" over the optimum), we use half of it in next iteration (this iteration's result is not even used).
# The speed of convergence could be improved by using some more advanced step size adaptation, like the *Armijo conditions*, but the solution would be (almost) the same.

# When we obtain the optimal `w`, we are ready to clasify the test labels.
#
# Before that we can check classification error on the train dataset.

trainprediction = classify(traindata, w)
#-
using titanic: classificationerror
classificationerror(trainlabels, trainprediction)

# We can see that even on the training set, the error is not zero, so we can conclude that the data are indeed linearly unserparable.

# Then we can predict classes for the test dataset.
# For this we cannot get error so straightforwardly, because we have to upload it to Kaggle website.
# So, the package provides a function for exporting `Labels` into `.csv` file.

testprediction = classify(testdata, w)
#- 
savepredictions("./logreg.csv", testprediction)

# From the website we obtain slightly worse error than on the training set, but this is expected and the difference is not that huge.
#
# | training error | test error |
# | :------------: | :--------: |
# | 0.19978        | 0.22967    |
#
# These results are not awesome, but for a basic linear classifier not that bad.

# ## Suppor Vector Machines
# Second, we showcase more complicated linear classifier and that is **SVM**.
# We implemented *soft-margin* variant with different kernel possibilities in function

# *Soft-margin* is used because the data are not linearly separable and thus we introduce penalty $C$ for misclassified points.
# $C$ is optional keyword parameter and defaults to $10$.

# The support vector machine in its base form again searches for some optimal vector `w` which defines the hyperplane.
# However, for optimization it is more convenient to optimize its dual problem
# \begin{align*}
# \alpha = \text{argmax}\quad&\sum_{i = 1}^{N}\alpha_i - \frac{1}{2}\sum_{i,j = 1}^{N} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \\
# \text{subject to}\quad&\sum_{i=1}^N \alpha_i y_i = 0 \\
# & 0 \leq \alpha_i \leq C
# \end{align*}

alpha = svm(traindata, trainlabels; C=10.0)

# Kernels on the other hand are supposed to handle the non-separability as they modify the feature space and possibly making the data more separable.
# They usually take some form of a dot product, as in vanilla svm.
# More generally, each symetric positive semidefinite matrix is a kernel function.
#
# Used kernels:
# - Linear kernel: $$ K(x_i, x_j) = x_i \cdot x_j $$
# - Polynomial kernel of power $p$: $$ K(x_i, x_j) = (1 + x_i \cdot x_j)^p $$
# - Gaussian kernel with variance $\sigma^2$: $$ K(x_i, x_j) = e^{\frac{-||x_i - x_j||_2^2}{2\sigma^2}} $$

# Kernels can be changed with setting the kernel function to one of these functions:

titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.linearkernel(xi, xj)
linearalpha = svm(traindata, trainlabels)
#-
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.polynomialkernel(xi, xj; degree=2)
polynomialalpha = svm(traindata, trainlabels)
#-
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.gaussiankernel(xi, xj; variance=1.0)
gaussianalpha = svm(traindata, trainlabels)

# When we obtain alpha from the dual problem, the classification then goes as follows:
# $$ \text{class}(x) = \text{sign}(\sum_{i=1}^N \alpha_i y_i K(x_i, x)) $$

# Again, we can compute errors on the training set

titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.linearkernel(xi, xj)
classificationerror(trainlabels, classify(traindata, traindata, trainlabels, linearalpha))
#-
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.polynomialkernel(xi, xj; degree=2)
classificationerror(trainlabels, classify(traindata, traindata, trainlabels, polynomialalpha))
#-
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.gaussiankernel(xi, xj; variance=1.0)
classificationerror(trainlabels, classify(traindata, traindata, trainlabels, gaussianalpha))

# As in logistic regression, for getting test error on test data, we need to upload predictions to the Kaggle page.
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.linearkernel(xi, xj)
linearpredictions = classify(testdata, traindata, trainlabels, linearalpha)
savepredictions("./linearsvm.csv", linearpredictions)
#-
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.polynomialkernel(xi, xj; degree=2)
polynomialpredictions = classify(testdata, traindata, trainlabels, polynomialalpha)
savepredictions("./polynomialsvm.csv", polynomialpredictions)
#-
titanic.kernel(xi::Vector{<:Real}, xj::Vector{<:Real}) = titanic.gaussiankernel(xi, xj; variance=1.0)
gaussianpredictions = classify(testdata, traindata, trainlabels, gaussianalpha)
savepredictions("./gaussiansvm.csv", gaussianpredictions)

# Results on training and test data are summarized in table below
#
# | kernel     | training error | test error |
# | :--------- | :------------: | :--------: |
# | linear     | 0.21324        | 0.23445    |
# | polynomial | 0.16498        | 0.24642    |
# | gaussian   | 0.12233        | 0.23924    |
#
# Interestingly, we can observe that while polynomial and gaussian kernels have significantly smaller error on the training set, they perform worse than the linear kernel on the test set.

# ## Adaboost
#
# Last implemented classifier is adaboost.
# Adaboost is actually a multitude of *weak (simple) classifiers* united into one so-called *strong classifier*.
# Each of these weak classifiers classifies a point based on some criterion and returns its decision to the strong classifier, which decides based on weighted sum of all these decisions.
# Weights are computed from classification errors of the weak classifiers.
# Simultaneously, it assigns different weights to data samples based on how hard it is for the weak classifiers to classify them correctly.
# Also, thanks to partial errors of the weak classifiers we can compute upper bound of the classification error, which can be convenient.
#
# Pseudocode for adaboost follows
#
# 1. initialize data weights $$D_1(i) = \frac{1}{N} \qquad \forall i = 1, \dots, N$$

# 2. for t in $1,\dots,T$
#
#     3. find the weak classifier $h_t$ with lowest error $$ h_t = \text{argmin }\epsilon(h) \qquad \epsilon(h) = \sum_{i=1}^N D_t(i)\llbracket y_i \neq  h_t(x_i)\rrbracket $$
#     4. if $\epsilon(h_t) \geq 0.5$ then stop
#     5. compute classifer weight $$\alpha_t = \frac{1}{2}\log(\frac{1 - \epsilon_t}{\epsilon_t}) $$
#     6. adjust data weights $$D_{t+1}(i) = \frac{1}{Z_t}D_t(i) e^{-\alpha_t y_i h_t(x_i)} \quad \forall i = 1, \dots, N \qquad Z_t = \sum_{i=1}^N D_t(i) e^{-\alpha_t y_i h_t(x_i)} $$
#
# $Z_t$ normalizes the weights so they sum to $1$ and also is the already mentioned upper bound on classification error.

# This choice seemed sensible as the features used in data samples are usually not related and adaboost gives the option to classify based on some feature only.
# With this thought, we designed class of weak classifiers that selects one data feature and finds threshold that misclassifies the least amount of samples based only on this feature.
# For each pair feature and threshold there are two weak classifiers, one that classifies all featuers $<$ than threhsold as $1$ and others as $-1$, and second that classifies all features $\geq$ than threshold as $1$ and others as $-1$.

# As this classifier operates on individual features, the data normalization is not needed.
# Thus, the data can be loaded like this

traindata, trainlabels = loadtrain("../data/train.csv"; normalize=false)
#-
testdata = loadtest("../data/test.csv"; normalize=false)

# Training of our classifier is performed by

H = boost(traindata, trainlabels; limit=100)

# The keyword parameter `limit` corresponds to $T$ in the pseudocode and limits the number of weak classifiers constructing the strong one.

# Classification of a point then proceeds as follows
# $$ H(x) = \text{sign}(\sum_{t=1}^T \alpha_t h_t(x)) $$
# Again, we compare errors on train and test datasets.

trainprediction = classify(traindata, H)
classificationerror(trainlabels, trainprediction)
#-
testprediction = classify(testdata, H)
savepredictions("./adaboost.csv", testprediction)

# Again, the training error is quite low, but the test error is very similar to results from other classifiers.
# It is surprising, that it did not work as expected.
# It is possible, that some features are present more in the training set and are more discriminative than in the test set.
# Perhaps, some experiments with what features use to decide would find an answer to this possibility and the algorithm could be improved.
#
# | training error | test error |
# | :------------: | :--------: |
# | 0.16835        | 0.23924    |

# The implementation of the algorithm is sped up by presorting feature spaces, so it is not necessary to iterate over all values in range, but only until the error starts increasing again
# If this feature was removed, we would get the same results but it would be a lot slower.
#
# Another speedup could be achieved by decreasing the `limit`.
# When we reduce it to $10$, the learning is approximately $10$ times faster, but the error on training set increased only by $0.2$.
# If speed was prefered to accuracy, this is another area to take advantage of.

# ## Conclusion
#
# | method              | training error | test error |
# | :------------------ | :------------: | :--------: |
# | logistic regression | 0.19978        | 0.22967    |
# | kernel svm          | 0.21324        | 0.23445    |
# | adaboost            | 0.16835        | 0.23924    |
#
# From the results we can see, that logistic regression is the best of the algorithms on test dataset, even though it does not have the lowest error on training data.
# On the other hand, adaboost has the lowest training error, but is the worst on testing data.
# This could mean, that it is overfitted or that the test data are more different from the training set than expected.
# The kernel svm generalizes quite well as the training error and test error are similar.
# It is disappointing that the kernels with lower training error performed worse on the test data.
# Possibly, it could be improved by finding better exponent or variance.
#
# In terms of speed, the worst is probably adaboost, because it does not have an easily optimizable form as gradient descent or quadratic optimization problem.
# We need to search for the best classifier over all features and that makes it slower.
#
# In conclusion, the results cannot be considered as a success as all the methods classified with quite high errors on both training and testing datasets.
# Some more powerful classifiers, such as neural nets, would probably work better or some smarter data preprocessing could be investigated.
# For example, some features could be omitted as unimportant and better results could be achieved.
