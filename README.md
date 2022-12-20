# Effort estimation for software development using predictive models

At some point in the life, every software developer finds themselves scratching their head when asked to provide the estimation of time/efforts to be spent during the development of a software. Effort, including developer hours/months, though countable is not tangible, and thus is dependent on guess work or prior knowledge.

Given the complexity, system constraints, required reliability, developer experience, etc., this projects' goal is to estimate the effort in the development of a software product.

In software development, effort estimation is a parameter that determines the realistic effort required by a team to develop, maintain or scale a software application. It is represented in hours or months, and is useful in planning a project and allocating budgets in the early stages of the projects.

## Why?

The lack of having a good effort estimate can result in issues such as wrong budget allocation, unrealistic expectations, rejection of bid for project ownership, substandard services, poor team morale, and continuous delays in delivering satisfactory updates to the customer.

Based on the available literature, majority of the previous attempts have converted the existing datasets to an ordinal data structure. The problem statement requires the magnitude of effort and not just the relative scale, thus, these approaches do not help a lot, and are therefore not optimal. Our approach takes into account each of the variables present in its raw form without manipulating any data, so that we can get accurate and pertinent results.

## Background study:

[1] A.J. Albrecht and J.E. Gaffney. “Software Function, Source Lines of Code, and Development Effort Prediction: A Software Science Validation”. In: IEEE Transactions on Software Engineering SE-9.6 (1983), pp. 639–648. [DOI: 10.1109/TSE.1983.235271](https://ieeexplore.ieee.org/document/1703110).

[2] K. Srinivasan and D. Fisher. “Machine learning approaches to estimating software development effort”. In: IEEE Transactions on Software Engineering 21.2 (1995), pp. 126–137. [DOI: 10. 1109/32.345828](https://ieeexplore.ieee.org/document/345828).

[3] Martin Shepperd and Chris Schofield. “Estimating software project effort using analogies”. In: Software Engineering, IEEE Transactions on 23 (Dec. 1997), pp. 736–743. [DOI: 10.1109/32. 637387](https://ieeexplore.ieee.org/document/637387).

[4] Ekrem Kocaguneli, Tim Menzies, and Jacky W. Keung. “On the Value of Ensemble Effort Estimation”. In: IEEE Transactions on Software Engineering 38.6 (2012), pp. 1403–1416. [DOI: 10.1109/TSE.2011.111](https://ieeexplore.ieee.org/document/6081882).

## Datasets

We are using two datasets here, both made available through Promise repository. Datasets can be found [here](https://github.ncsu.edu/agupta57/engr-ALDA-Fall2022-P25-Effort_estimation_for_software_development_using_predictive_models/tree/main/datasets).

## Performance measures

We are using four performance measures for this project:

* RMSE (root mean square error): This is computed by taking the square root of the mean of the squared difference between actual and predicted values.
* MAE (mean absolute error): This is computed by taking the mean of the absolute difference between actual and predicted values.
* MMRE (mean magnitude of relative error): This is computed by taking the mean of the absolute relative error between actual and predicted values.
* R<sup>2</sup> (coefficient of determination): It represents the proportion of variance (of y) that has been explained by the independent variables in the model.

More details on these performance measures can be found [here](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).

## Installation guide

### Prerequisites

1) [Python](https://www.python.org/): 3.10.6
2) [Matplotlib](https://matplotlib.org/): 3.5.3
3) [Numpy](https://numpy.org/doc/stable/index.html): 1.23.2
4) [Pandas](https://pandas.pydata.org/): 1.4.4
5) [Scikit-learn](https://scikit-learn.org/stable/): 1.1.2
6) [Scipy](https://scipy.org/): 1.7.3
7) [Seaborn](https://seaborn.pydata.org/): 0.12.0
8) [Tensorflow](https://www.tensorflow.org/): 2.10.0
9) [xgboost](https://xgboost.readthedocs.io/en/stable/): 1.7.1

### How to install and run

Note: These steps are tested on linux

1. Clone the repo `git clone https://github.ncsu.edu/agupta57/engr-ALDA-Fall2022-P25-Effort_estimation_for_software_development_using_predictive_models`.
2. Create a virtual environment by using `python3 -m venv <environment-name>` (can be done anywhere).
3. Activate the environment using `source <environment-name>/bin/activate` for linux and `<environment-name>\Scripts\activate.bat` for Windows. For the rest of the OS please check here https://docs.python.org/3/library/venv.html. Ensure that python and pip both point to the path provided by virtual environment.
4. Install the requirements `pip install -r requirements.txt`.
5. Once this is done get into the folder `src/cocomo81/` or `src/Desharnais` depending on which dataset you want to use.
6. Run the code using `python cocomo81.py` or `python desharnais.py`.
7. Let the code run and capture the results.

### Results

All the results from local run can be found [here](https://github.ncsu.edu/agupta57/engr-ALDA-Fall2022-P25-Effort_estimation_for_software_development_using_predictive_models/tree/main/results).

## Method and Experiment Setup

1. Load datasets using pandas. Parse and remove non-numeric values. Perform Exploratory Data Analysis (EDA) and analyze [box](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html), [scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html), [histogram plots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) and [correlation matrix](https://seaborn.pydata.org/generated/seaborn.heatmap.html).
2. Perform preprocessing steps like normalization, [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), and splitting the datasets into folds to apply cross validation. This experiment uses a 10-fold approach, where testing is (identifier mod 10 = i - 1, i ϵ {1, 10}).
3. Applied individual models include the following:
    - [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    - [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
    - [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
    - [KNN Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
    - [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
    - [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    - [Support Vector Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
4. Used bagging and boosting (Adaboost) techniques to form ensemble:
    - [Bagging Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
    - [AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
5. Applied RandomForest, XGBoost, stacking and voting approach for ensemble of models:
    - [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    - [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
    - [Stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
    - [Voting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
6. Applied NN approaches:
    - [Multi-layer Perceptron Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
    - [Custom Tensorflow ANN](https://www.tensorflow.org/api_docs/python/tf/keras)
8. Compared the performance using aforementioned performance measures.


## Directory structure

    .
    ├── .vscode
    |   ├── settings.json                   # Workspace settings for vscode
    ├── datasets
    |   ├── cocomo81.csv                    # Dataset for cocomo81
    |   ├── desharnais.csv                  # Dataset for desharnais
    |   ├── README.md
    ├── presentation                        # Contains presentation for the project
    ├── results                             # Folder containing all the results
    |   ├── cocomo                          # Folder containing all results for cocomo
    |   ├── README.md
    ├── src
    |   ├── Desharnais
    |   |   ├── desharnais.py               # The source code to analyse desharnais dataset
    |   |   ├── README.md
    |   ├── cocomo81
    |   |   ├── cocomo81.py                 # The source code to analyse cocomo81 dataset
    |   |   ├── README.md
    |   ├── README.md
    ├── README.md
    └── requirements.txt                    # All the depdencies to run the code

## Conclusion

| Desharnais | Cocomo81 |
| ------------------------------------------ | ---      |
| No non-numeric data observed. | No non-numeric data observed. |
| Five of the eleven features have high correlation with the target. | Only one feature has high correlation with target. |
| Independent variables have varying distributions, target variable has a right skewed normal distribution. | Independent variables not normally distributed. |
| KNN regression using 5 principal components and 6 nearest neighbours gives the best results - low error measures and highest R2 score. | Decision tree with boosting (ensemble) gives the best results across all performance measures. Has the lowest error measures and highest R2 score. Unlike other models, which perform better on some measures, DT performs better on all. |
| Using 6 and 7 principal components gives good consistent results on linear regularized models such as Lasso and Ridge regression. | Applying PCA leads to worse results except when applied with ANN. |

## Future scope

* Currently the main drawback is in the number of datasets that are present with regards to effort estimation.
* Even in the datasets that are present, the number of samples are quite low to actually innovate an effective Neural Network approach.
* Since the data samples are very less, we need to run an extremely high amount of epochs to actually the fit the model, which results in overfitting.
* Thus, more real world data needs to be captured to perform a more effective and relevant analysis.
* Also, here the process of effort estimation is ending with the initial product development, and can be further extended to include maintenance.

## Support

For any queries, please shoot us an email at any of the following:

* agupta57@ncsu.edu
* apartha4@ncsu.edu
* lsangar@ncus.edu
