# titanic

Package containing multiple methods for solving *Titanic* classification problem.
More information on [Kaggle](https://www.kaggle.com/c/titanic/overview)

## Package structure

- `data/` contains datasets for training and testing downloaded from Kaggle website
- `report/` contains report written in `Literate.jl` and exported to *jupyter* notebook
- `results/` contains classified labels for test data meant for submission
- `scripts/` contains scripts used to execute individual classifiers - learn from training data and classify test data
- `src/` contains source files of the package
    - `adaboost.jl` - the adaboost classifier
    - `dataset.jl` - structure and helper functions used to represent and manipulate input data
    - `io.jl` - functions to transform input `.csv` files into inner representation
    - `logistic.jl` - the logistic regression classifier
    - `svm.jl` - the support vector machines classifier
    - `titanic.jl` - package definition
- `test/` - contains unit tests for selected functions
