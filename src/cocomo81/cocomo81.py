import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sklearn.metrics as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
import xgboost as xg
import random
import os
import tensorflow as tf


class Cocomo:
    def __init__(self):
        self.cols = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap',
                     'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced', 'loc', 'actual']
        self.df = self.readFromCsv(r'../../datasets/cocomo81.csv')
        self.covarianceMatrix = None
        self.eigenVals = None
        self.eigenVecs = None
        self.pcaDataset = None
        self.folds = {}
        self.stats = {}
        self.estimators = {}
        self.printDatasetHead(3)
        self.findShape()
        self.checkNull()
        self.describeDataset()
        self.misc()
        self.visualizeBoxPlot()
        self.visualizeHistogram()
        self.visualizeScatterPlot()
        self.visualizeOutputDependency()
        self.printCorrelationMatrix()
        self.visualizeDistribution()
        self.normalise()
        self.pcaTransformData()
        self.splitDataIntoFolds()
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runLinearRegression(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runRidge(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runLasso(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runKNN(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runElasticNet(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runDecisionTree(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runSVR(type)
        for type in ['Default', 'Bagging', 'Boosting']:
            self.runMLP(type)
        self.runRandomForest()
        self.runXgBoost()
        self.formEstimatorList()
        self.runStackedEnsemble()
        self.runVotingEnsemble()
        self.runAnn()
        self.visualizeResults()

    def readFromCsv(self, filePath):
        return pd.read_csv(filePath, names=self.cols)

    def printDatasetHead(self, num):
        print('\nDataset head:\n', self.df.head(num))

    def findShape(self):
        print('\nDataset shape: ', self.df.shape)

    def checkNull(self):
        print('\nAny null entry: ', self.df.isnull().values.any())

    def describeDataset(self):
        print('\nDataset info:\n')
        print(self.df.info())
        print('\nDataset describe:\n', self.df.describe())

    def misc(self):
        print(self.df.actual.value_counts())

    def visualizeBoxPlot(self):
        for col in self.cols:
            if col != 'actual':
                plt.boxplot(self.df[col], vert=False)
                plt.title(col)
                label = 'Min: {}, Mean: {:.2f}, Max: {}'.format(
                        self.df[col].min(), self.df[col].mean(), self.df[col].max())
                plt.xlabel(label)
                plt.show()

    def visualizeHistogram(self):
        for col in self.cols:
            if col != 'actual':
                plt.hist(self.df[col], bins=10, edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('frequency')
                plt.show()

    def visualizeScatterPlot(self):
        print('Choose 2 column number to view scatter:')
        for i in range(self.df.shape[1]):
            print(i, ':', self.df.columns[i], end='  \t')
        print()
        ch = 'y'
        while ch == 'y' or ch == 'Y':
            print('\nEnter column1 number:')
            col1 = int(input())
            print('\nEnter column2 number:')
            col2 = int(input())
            plt.scatter(self.df.iloc[:, col1], self.df.iloc[:, col2])
            plt.xlabel(self.df.columns[col1])
            plt.ylabel(self.df.columns[col2])
            plt.show()
            print(
                '\nWant to see another scatter plot? Enter y for Yes or any other key for no:')
            ch = input()

    def visualizeOutputDependency(self):
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(30, 30), gridspec_kw={
            'left': 0.07, 'bottom': 0.1, 'right': 0.95, 'top': 0.95, 'wspace': 0.3, 'hspace': 0.4})
        i = 0
        j = 0
        for k in range(16):
            axes[i][j].scatter(self.df.iloc[:, 16], self.df.iloc[:, k])
            axes[i][j].set_xlabel(self.df.columns[16])
            axes[i][j].set_ylabel(self.df.columns[k])
            j += 1
            if j > 3:
                j = 0
                i += 1
        plt.show()

    def printCorrelationMatrix(self):
        # print('\nCorrelation matrix:\n', self.df.corr())
        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(self.df.corr(), annot=True, linewidths=.5, fmt='.2f')
        plt.show()

    def visualizeDistribution(self):
        for col in self.cols:
            if col != 'actual':
                sns.displot(self.df, kind='kde', x=col, cut=0)
                plt.show()

    def normalise(self):
        for col in self.cols:
            if col != 'actual':
                max = self.df[col].max()
                min = self.df[col].min()
                self.df[col] = self.df[col].apply(
                    lambda x: ((x - min)/(max - min)))

    def pcaTransformData(self, ifAnn=0):
        # PCA giving worse results for everything except ANN
        if ifAnn:
            pca = PCA(n_components=9)
            pca.fit(self.df.drop('actual', axis=1))
            self.pcaDataset = pd.DataFrame(pca.transform(
                self.df.drop('actual', axis=1))[:, 0:9])
        # xAxis = range(1, 17)
        # plt.plot(xAxis, pca.explained_variance_ratio_, 'r', linewidth=2)
        # plt.title('Bar graph and Scree Plot')
        # plt.xlabel('Principal Components')
        # plt.ylabel('Explained variance')
        # plt.bar(xAxis, pca.explained_variance_ratio_)
        # x = np.arange(len(pca.explained_variance_ratio_))
        # plt.xticks(x)
        # plt.show()
        else:
            self.pcaDataset = self.df.drop('actual', axis=1)

    def splitDataIntoFolds(self):
        for i in range(10):
            self.folds[i] = {
                'train': [],
                'test': []
            }
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        counter = 0
        for train_index, test_index in kf.split(self.pcaDataset):
            self.folds[counter % 10]['train'].append(train_index)
            self.folds[counter % 10]['test'].append(test_index)
            counter += 1

    def trainTestSplit(self, counter):
        trainX = list(self.pcaDataset.values[self.folds[counter]['train']][0])
        trainY = list(
            self.df['actual'].values[self.folds[counter]['train']][0])
        testX = list(self.pcaDataset.values[self.folds[counter]['test']][0])
        testY = list(self.df['actual'].values[self.folds[counter]['test']][0])
        return (trainX, trainY, testX, testY)

    def printStats(self, predictions, actual, model):
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        mmre = np.mean(np.abs(np.array(actual) -
                              np.array(predictions)) / np.array(actual))
        r2 = sm.r2_score(actual, predictions)
        print('RMSE: ', rmse)
        print('Mean Absolute error: ', mae)
        print('MMRE: ', mmre)
        print('R2 score: ', r2)
        self.stats[model] = {
            'rmse': rmse,
            'mae': mae,
            'mmre': mmre,
            'r2': r2
        }

    def runLinearRegression(self, type='Default'):
        print('\nLinear {} Regression stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = LinearRegression()
            if type == 'Bagging':
                regressor = BaggingRegressor(base_estimator=LinearRegression(
                ), n_estimators=2000, random_state=0, n_jobs=-1, max_samples=55, max_features=10, bootstrap_features=True, oob_score=True)
            elif type == 'Boosting':
                regressor = AdaBoostRegressor(base_estimator=LinearRegression(
                ), n_estimators=2000, random_state=0, learning_rate=0.1)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'lr-' + type[:3])

    def runRidge(self, type='Default'):
        print('\nRidge {} Regression Stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = RidgeCV(cv=10, alphas=[1e-3, 1e-2, 1e-1, 1])
            if type == 'Bagging':
                regressor = BaggingRegressor(base_estimator=RidgeCV(cv=10, alphas=[
                    1e-3, 1e-2, 1e-1, 1]), n_estimators=2000, random_state=0, n_jobs=-1, max_samples=55, max_features=10, bootstrap_features=True, oob_score=True)
            elif type == 'Boosting':
                regressor = AdaBoostRegressor(base_estimator=RidgeCV(cv=10, alphas=[
                    1e-3, 1e-2, 1e-1, 1]), n_estimators=2000, random_state=0, learning_rate=0.1)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'r-' + type[:3])

    def runLasso(self, type='Default'):
        print('\nLasso {} Regression Stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = LassoCV(cv=10, random_state=0)
            if type == 'Bagging':
                regressor = BaggingRegressor(base_estimator=LassoCV(
                    cv=10, random_state=0), n_estimators=40, random_state=0, n_jobs=-1)
            elif type == 'Boosting':
                regressor = AdaBoostRegressor(base_estimator=LassoCV(
                    cv=10, random_state=0), n_estimators=2000, random_state=0, learning_rate=0.1)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'l-' + type[:3])

    def runKNN(self, type='Default'):
        print('\nKNN {} Regression Stats:\n'.format(type))
        for k in range(1, 10):
            predictions = []
            actual = []
            testCounter = 0
            while testCounter < 10:
                trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
                regressor = KNeighborsRegressor(n_neighbors=k)
                if type == 'Bagging':
                    regressor = BaggingRegressor(base_estimator=KNeighborsRegressor(
                        n_neighbors=k), n_estimators=2000, random_state=0, n_jobs=-1, max_samples=55, max_features=10, bootstrap_features=True, oob_score=True)
                elif type == 'Boosting':
                    regressor = AdaBoostRegressor(base_estimator=KNeighborsRegressor(
                        n_neighbors=k), n_estimators=2000, random_state=0, learning_rate=0.1)
                regressor.fit(trainX, trainY)
                predictions += list(regressor.predict(testX))
                actual += testY
                testCounter += 1
            print('\nFor k = ', k)
            self.printStats(predictions, actual, str(k) + '-knn-' + type[:3])

    def runElasticNet(self, type='Default'):
        print('\nElasticNet {} Regression Stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = ElasticNetCV(cv=10, random_state=0)
            if type == 'Bagging':
                regressor = BaggingRegressor(base_estimator=ElasticNetCV(
                    cv=10, random_state=0), n_estimators=40, random_state=0, n_jobs=-1)
            elif type == 'Boosting':
                regressor = AdaBoostRegressor(base_estimator=ElasticNetCV(
                    cv=10, random_state=0), n_estimators=2000, random_state=0, learning_rate=0.1)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'en-' + type[:3])

    def runDecisionTree(self, type='Default'):
        print('\nDecisionTree {} Regression Stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = tree.DecisionTreeRegressor(random_state=0)
            if type == 'Bagging':
                regressor = BaggingRegressor(base_estimator=tree.DecisionTreeRegressor(
                    random_state=0), n_estimators=2000, random_state=0, n_jobs=-1, max_samples=55, max_features=10, bootstrap_features=True, oob_score=True)
            elif type == 'Boosting':
                regressor = AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(
                    random_state=0), n_estimators=2000, random_state=0, learning_rate=0.1)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'dt-' + type[:3])

    def runSVR(self, type='Default'):
        print('\nSVM {} Regression Stats:\n'.format(type))
        param_grid = [
            {'C': [0.1, 0.2, 0.3, 1, 5, 10, 20, 100,
                   200, 1000], 'kernel': ['linear']},
            {'C': [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000], 'gamma': [
                0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3], 'kernel': ['rbf']},
            {'C': [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000], 'degree': [1, 2, 3, 4, 5], 'coef0':[
                0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10], 'kernel': ['poly']},
            {'C': [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000], 'gamma': [
                0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3], 'coef0':[
                0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10], 'kernel': ['sigmoid']}
        ]
        for i in param_grid:
            testCounter = 0
            predictions = []
            actual = []
            while testCounter < 10:
                trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
                regressor = SVR()
                grid = GridSearchCV(estimator=regressor, param_grid=i, cv=10)
                if type == 'Bagging':
                    grid = BaggingRegressor(base_estimator=GridSearchCV(estimator=regressor, param_grid=i, cv=10), n_estimators=10,
                                            random_state=0, n_jobs=-1, max_samples=55, max_features=10, bootstrap_features=True, oob_score=True)
                elif type == 'Boosting':
                    grid = AdaBoostRegressor(base_estimator=GridSearchCV(
                        estimator=regressor, param_grid=i, cv=10), n_estimators=10, random_state=0, learning_rate=0.1)
                grid.fit(trainX, trainY)
                predictions += list(grid.predict(testX))
                actual += testY
                testCounter += 1
            print('\nKernel: ', i['kernel'])
            self.printStats(predictions, actual,
                            i['kernel'][0][:3] + '-svm-' + type[:3])

    @ignore_warnings(category=ConvergenceWarning)
    def runMLP(self, type='Default'):
        print('\nMLP {} Regression stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = MLPRegressor(random_state=0, max_iter=10)
            if type == 'Bagging':
                regressor = BaggingRegressor(base_estimator=MLPRegressor(random_state=0, max_iter=10), n_estimators=2000,
                                             random_state=0, n_jobs=-1, max_samples=55, max_features=10, bootstrap_features=True, oob_score=True)
            elif type == 'Boosting':
                regressor = AdaBoostRegressor(base_estimator=MLPRegressor(
                    random_state=0, max_iter=10), n_estimators=2000, random_state=0, learning_rate=0.1)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'mlp-' + type[:3])

    def runRandomForest(self):
        print('\nRandom Forest Regression Stats:\n')
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = RandomForestRegressor(
                n_estimators=2000, random_state=0, max_features=0.3)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'rf')

    def runXgBoost(self):
        print('\nXGBoost Regression Stats:\n')
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            trainDMatrix = xg.DMatrix(data=trainX, label=trainY)
            testDMatrix = xg.DMatrix(data=testX, label=testY)
            params = {
                'booster': 'gblinear',
                'objective': 'reg:squarederror'
            }
            regressor = xg.train(
                params=params, dtrain=trainDMatrix, num_boost_round=10)
            predictions += list(regressor.predict(testDMatrix))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'xgb')

    def formEstimatorList(self):
        self.estimators = [
            ('linear', LinearRegression()),
            ('ridge', RidgeCV(cv=10, alphas=[1e-3, 1e-2, 1e-1, 1])),
            ('lasso', LassoCV(cv=10, random_state=0)),
            ('elasticNet', ElasticNetCV(cv=10, random_state=0)),
            ('mlp', MLPRegressor(random_state=0, max_iter=10)),
            ('randomForest', RandomForestRegressor(
                n_estimators=2000, random_state=0, max_features=0.3)),
            ('knn', KNeighborsRegressor(n_neighbors=2))
        ]

    @ignore_warnings(category=ConvergenceWarning)
    def runStackedEnsemble(self):
        print('\nStacked Regression Stats:\n'.format(type))
        for estimator in self.estimators:
            predictions = []
            actual = []
            testCounter = 0
            while testCounter < 10:
                trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
                regressor = StackingRegressor(
                    estimators=self.estimators, final_estimator=estimator[1])
                regressor.fit(trainX, trainY)
                predictions += list(regressor.predict(testX))
                actual += testY
                testCounter += 1
            print('\nFor estimator = ', estimator)
            self.printStats(predictions, actual, estimator[0][:3] + '-stacked')

    @ignore_warnings(category=ConvergenceWarning)
    def runVotingEnsemble(self):
        print('\nVoting Regression Stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = VotingRegressor(estimators=self.estimators)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1
        self.printStats(predictions, actual, 'voting')

    def runAnn(self):
        print('\nANN Stats:\n')
        os.environ['PYTHONHASHSEED'] = str(10)
        random.seed(10)
        np.random.seed(10)
        tf.random.set_seed(10)
        predictions = []
        actual = []
        testCounter = 0
        self.pcaTransformData(ifAnn=1)
        self.splitDataIntoFolds()
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            trainX = np.array(trainX)
            trainY = np.array(trainY)
            testX = np.array(testX)
            testY = np.array(testY)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(
                64, input_shape=(trainX.shape[1],), activation="tanh"))
            model.add(tf.keras.layers.Dense(32, activation="relu"))
            model.add(tf.keras.layers.Dense(1, activation="linear"))
            model.compile(loss='mse', optimizer="adam")
            model.fit(trainX, trainY, epochs=600, batch_size=10)
            predY = model.predict(testX)
            predictions += list(predY)
            actual += list(testY)
            tf.keras.backend.clear_session()
            del model
            os.environ['PYTHONHASHSEED'] = str(10)
            random.seed(10)
            np.random.seed(10)
            tf.random.set_seed(10)
            testCounter += 1
        self.printStats(predictions, actual, 'ANN')

    def visualizeResults(self):
        stats = ['rmse', 'mae', 'mmre', 'r2']
        for stat in stats:
            defModels = []
            bagModels = []
            bosModels = []
            ensModels = []
            for model in self.stats.keys():
                if '-Def' in model:
                    defModels.append(model)
                elif '-Bag' in model:
                    bagModels.append(model)
                elif '-Boo' in model:
                    bosModels.append(model)
                else:
                    ensModels.append(model)
            models = defModels + bagModels + bosModels + ensModels
            xTicks = np.arange(len(models))
            values = []
            for model in models:
                values.append(self.stats[model][stat])
                print(model, self.stats[model][stat])
            if stat == 'rmse':
                plt.ylim([800, 2500])
            elif stat == 'mae':
                plt.ylim([300, 1800])
            plt.bar(models, values, label=stat, color='b')
            plt.xticks(xTicks, models, rotation=90)
            plt.xlabel('Models')
            plt.ylabel(stat)
            plt.title('model v/s ' + stat)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    Cocomo()
