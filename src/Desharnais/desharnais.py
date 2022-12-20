import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn.metrics as sm
import warnings
from matplotlib.pyplot import figure
warnings.filterwarnings("ignore")


def readCsv(path):
    # Read csv into pandas dataframe
    columns = ['Proj', 'TeamExp', 'ManagerExp', 'YearEnd', 'Len', 'Effort', 'Transac', 'Entities', 'PtsNonAdjust',
               'Adjust', 'PtsAjust', 'Lang']
    df = pd.read_csv(path, names=columns)

    # Rearrange column order to make Effort as last dataframe column
    columns[columns.index('Effort')] = columns[columns.index('Lang')]
    columns[len(columns) - 1] = 'Effort'
    df = df[columns]
    print(df.head(1))
    return df

#######################################################################################################################
def checkNullFill(df):
    # Check for null values
    print("\nDoes the data contain null values:", df.isnull().values.any(), end="\n\n")

    # Convert object type to numeric type
    df['TeamExp'] = pd.to_numeric(df['TeamExp'], errors='coerce')
    df['ManagerExp'] = pd.to_numeric(df['ManagerExp'], errors='coerce')
    df.fillna(0, inplace=True)


def describeData(df):
    cols = df.columns
    desc = df.describe()
    for each in cols:
        eachColStats = pd.DataFrame(desc[each]).transpose()
        print(f"Attribute {each} statistics:\n{eachColStats} \n\n")
    print("\nDatatype of each column:\n")
    print(df.info())
#######################################################################################################################



#######################################################################################################################
def boxPlot(df, cols):
    for i in range(0, len(cols), 3):
        df[cols[i:i + 3]].plot.box(subplots=True)
        plt.tight_layout()
        plt.show()


def barPlot(df, cols):
    for i in range(0, len(cols), 4):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3))

        ax1.hist(df[cols[i]], bins=10, edgecolor="black")
        ax1.set_xlabel(cols[i])
        ax1.set_ylabel('Frequency')

        ax2.hist(df[cols[i + 1]], bins=10, edgecolor="black")
        ax2.set_xlabel(cols[i + 1])
        ax2.set_ylabel('Frequency')

        ax3.hist(df[cols[i + 2]], bins=10, edgecolor="black")
        ax3.set_xlabel(cols[i + 2])
        ax3.set_ylabel('Frequency')

        ax4.hist(df[cols[i + 3]], bins=10, edgecolor="black")
        ax4.set_xlabel(cols[i + 3])
        ax4.set_ylabel('Frequency')

        fig.tight_layout(pad=2.0)
        plt.show()


def scatterPlotVsEffort(df):
    effort = 11
    for j in range(0, df.shape[1], 4):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
        axes[0].scatter(df.iloc[:, effort], df.iloc[:, j], color='red')
        axes[0].set_xlabel(df.columns[effort])
        axes[0].set_ylabel(df.columns[j])

        axes[1].scatter(df.iloc[:, effort], df.iloc[:, j + 1], color='red')
        axes[1].set_xlabel(df.columns[effort])
        axes[1].set_ylabel(df.columns[j + 1])

        axes[2].scatter(df.iloc[:, effort], df.iloc[:, j + 2], color='red')
        axes[2].set_xlabel(df.columns[effort])
        axes[2].set_ylabel(df.columns[j + 2])
        fig.tight_layout()

        axes[3].scatter(df.iloc[:, effort], df.iloc[:, j + 3], color='red')
        axes[3].set_xlabel(df.columns[effort])
        axes[3].set_ylabel(df.columns[j + 3])
        fig.tight_layout()
        plt.show()


def scatterAmongAll(df):
    print("Want to see all scatter plots between every attribute ?")
    ans = input()
    if ans.lower() == "yes" or ans.lower() == "y":
        print("\nOptions :\n1. View each pair plot in detail \n2. View zoomed out pairwise plot")
        ans2 = int(input())
        if ans2 == 1:
            for i in range(0, df.shape[1]):
                for j in range(0, df.shape[1], 3):
                    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
                    axes[0].scatter(df.iloc[:, i], df.iloc[:, j])
                    axes[0].set_xlabel(df.columns[i])
                    axes[0].set_ylabel(df.columns[j])

                    axes[1].scatter(df.iloc[:, i], df.iloc[:, j + 1])
                    axes[1].set_xlabel(df.columns[i])
                    axes[1].set_ylabel(df.columns[j + 1])

                    axes[2].scatter(df.iloc[:, i], df.iloc[:, j + 2])
                    axes[2].set_xlabel(df.columns[i])
                    axes[2].set_ylabel(df.columns[j + 2])
                    fig.tight_layout()
                    plt.show()
        elif ans2 == 2:
            sns.pairplot(df)
            
            
def plotMetrics(model, metric, res, color):
    typeModel = list(res.keys())
    metricValues = list(res.values())

    plt.figure(figsize=(9, 5))
    low = min(metricValues)
    high = max(metricValues)
    title = model+" "+metric.upper()
    if metric not in ["r2", "mmre"] : plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.1*(high-low))])
    plt.bar(range(len(res)), metricValues, tick_label = typeModel, color = color, edgecolor = "black", width = 0.7)
    plt.title(title)
    plt.show()
    
    
def plotMetricsDriver(forPlotMetrics):
    modelNames = ["LinearRegression","LassoRegression","RidgeRegression","KNNRegression"]
    metrics = ["r2", "rmse", "mae", "mmre"]
    colors = ['orange', 'yellow', 'green', 'cyan']
    for i in range(len(modelNames)):
        for eachMetric in metrics:
            res = {}
            for eachType, eachTypeVal in forPlotMetrics.items():
                res[eachType] = eachTypeVal[modelNames[i]][eachMetric]
            plotMetrics(modelNames[i], eachMetric, res, colors[i])
            
            
def barPlotResults(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
        bars.append(bar[0])
    if legend:
        ax.legend(bars, data.keys(),loc=(1.04, 0.72))
        
        
def barPlotsResultsDriver(forPlotMetrics):
    arr = {}
    metrics = ["r2", "rmse", "mae", "mmre"]
    for i in range(len(metrics)):
        buf = {
            "LinearRegression":[],
            "LassoRegression" :[],
            "RidgeRegression" : [],
            "KNNRegression": []
        }
    
        for eachKey, valDict in forPlotMetrics.items():
            buf["LinearRegression"].append(valDict["LinearRegression"][metrics[i]])
            buf["LassoRegression"].append(valDict["LassoRegression"][metrics[i]])
            buf["RidgeRegression"].append(valDict["RidgeRegression"][metrics[i]])
            buf["KNNRegression"].append(valDict["KNNRegression"][metrics[i]])
        
        arr[metrics[i]] = buf
    
    for eachKey, eachVal in arr.items():
        fig, ax = plt.subplots()
        barPlotResults(ax, eachVal, total_width=.8, single_width=.9)
        plt.title(eachKey.upper())
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        x_label = f"{' '*4}all_features{' '*7}normalized{' '*8}select_features{' '*9}PCA"
        plt.xlabel(x_label, horizontalalignment='left', x=0.01)
        plt.show()
#######################################################################################################################




#######################################################################################################################
def correlationData(df):
    pd.set_option('display.expand_frame_repr', False)
    print(f"Correlation matrix:\n{df.corr().round(decimals=1)}")

    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.2f')
    plt.show()


def effortDistribution(df):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.distplot(df['Effort'], fit=norm)

    plt.subplot(1, 2, 2)
    res = stats.probplot(df['Effort'], plot=plt)
    plt.subplots_adjust(wspace=0.4)
    plt.show()

#######################################################################################################################



#######################################################################################################################
def normalize(df):
    x = df.values
    y = df['Effort']
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled)
    df_norm.columns = df.columns
    if 'Proj' in df_norm.columns: df_norm = df_norm.drop('Proj', axis=1)
    df_norm = df_norm.drop('Effort', axis=1)
    return df_norm, y

#######################################################################################################################

def LinReg(x_train, y_train, x_test):
    model_LR = LinearRegression()
    model_LR.fit(x_train, y_train)
    y_test_pred = model_LR.predict(x_test)
    return y_test_pred


def LassoReg(x_train, y_train, x_test):
    alphas = [0.01, 0.1, 1, 10, 100]
    cv_mse = []
    for each in alphas:
        model_Lasso = Lasso(alpha=each)
        ten_fold_mse = TenFold(model_Lasso, x_train, y_train)
        cv_mse.append(ten_fold_mse)

    ind = cv_mse.index(min(cv_mse))
    model_Lasso = Lasso(alpha=alphas[ind])
    model_Lasso.fit(x_train, y_train)
    y_test_pred = model_Lasso.predict(x_test)
    return y_test_pred


def LassoRegCV(x_train, y_train, x_test):
    model_lassocv = LassoCV(alphas=None, cv=10, max_iter=100000)
    model_lassocv.fit(x_train, y_train)
    y_test_pred = model_lassocv.predict(x_test)
    return y_test_pred


def RidgeReg(x_train, y_train, x_test):
    alphas = [0.01, 0.1, 1, 10, 100]
    cv_mse = []
    for each in alphas:
        model_ridge = Ridge(alpha=each)
        ten_fold_mse = TenFold(model_ridge, x_train, y_train)
        cv_mse.append(ten_fold_mse)

    ind = cv_mse.index(min(cv_mse))
    model_ridge = Ridge(alpha=alphas[ind])
    model_ridge.fit(x_train, y_train)
    y_test_pred = model_ridge.predict(x_test)
    return y_test_pred


def knnReg(x_train, y_train, x_test, k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    ten_fold_mse = TenFold(model, x_train, y_train)
    return y_test_pred


def mmre(y_actual, y_pred):
    if isinstance(y_actual, np.ndarray):
        y_actual = np.array(y_actual.tolist())
    else:
        y_actual = np.array(y_actual)
    y_subtracted = abs(np.subtract(y_actual, y_pred)) / y_actual

    mmreSum = 0
    for i in range(len(y_actual)):
        if y_actual[i] != 0:
            mmreSum += abs(y_actual[i] - y_pred[i]) / y_actual[i]
        else:
            mmreSum += abs(y_actual[i] - y_pred[i])
    mmre = mmreSum / len(y_actual)
    return mmre


def KNNBest(x_train, y_train, x_test, k=12):
    knn_k = []
    cv_mse = []

    for k in range(2, k + 1, 2):
        model = KNeighborsRegressor(n_neighbors=k)
        ten_fold_mse = TenFold(model, x_train, y_train)
        cv_mse.append(ten_fold_mse)
        knn_k.append(k)

    minCvRmse = min(cv_mse)
    minK = knn_k[cv_mse.index(minCvRmse)]
    model = KNeighborsRegressor(n_neighbors=minK)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)

    return y_test_pred


def pca_calc(comp, x, y):
    res = {}
    pca = PCA(n_components=comp)
    pca.fit(x)
    num_Of_PCA_Comps = ['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4', 'Comp_5', 'Comp_6', 'Comp_7', 'Comp_8', 'Comp_9',
                        'Comp_10']
    df_pca = pd.DataFrame(pca.transform(x), columns=num_Of_PCA_Comps[:comp])

    y_pca = y
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(df_pca, y_pca, test_size=0.25, random_state=30)

    res['LinearRegression'] = LinReg(x_train_pca, y_train_pca, x_test_pca)
    res['LassoRegression'] = LassoReg(x_train_pca, y_train_pca, x_test_pca)
    res['RidgeRegression'] = RidgeReg(x_train_pca, y_train_pca, x_test_pca)
    res['LassoCVRegression'] = LassoRegCV(x_train_pca, y_train_pca, x_test_pca)
    res['KNNRegression'] = KNNBest(x_train_pca, y_train_pca, x_test_pca)

    return res, y_test_pca
#######################################################################################################################



#######################################################################################################################
def TenFold(model, X, y):
    cv = KFold(n_splits=10, random_state=30, shuffle=True)
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    return math.sqrt(sum(-1 * score) / len(X))


def calcMetrics(ytest, ypred):
    res = {}
    res['r2'] = round(r2_score(ytest, ypred), 4)
    res['rmse'] = round(math.sqrt(mean_squared_error(ytest, ypred)), 4)
    res['mae'] = round(mean_absolute_error(ytest, ypred), 4)
    res['mmre'] = round(mmre(ytest, ypred), 5)
    return res


def printMetrics(modelDetails, res, misc=""):
    print(f"\nFor model {modelDetails} {misc}:")
    for key, value in res.items():
        print(f"{key} : {value}")
        
        
def pcaPeakRecalc(resultMetrics, newResults):
    for model, vals in resultMetrics.items():
        for eachMetric, val in resultMetrics[model].items():
            if eachMetric == "r2":
                if resultMetrics[model]["r2"] < newResults[model]["r2"]:
                    resultMetrics[model]["r2"] = newResults[model]["r2"]
            else:
                if resultMetrics[model][eachMetric] > newResults[model][eachMetric]:
                    resultMetrics[model][eachMetric] = newResults[model][eachMetric]
#######################################################################################################################




#######################################################################################################################
def allDataModels(df):
    X = df.loc[:, df.columns != 'Effort']
    Y = df['Effort']
    resultMetrics = {}
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=30)

    models = {
        "LinearRegression": LinReg,
        "LassoRegression": LassoReg,
        "RidgeRegression": RidgeReg,
        "KNNRegression": KNNBest
    }

    for modelName, model in models.items():
        y_test_pred = models[modelName](x_train, y_train, x_test)
        resMetrics = calcMetrics(y_test, y_test_pred)
        modelStr, modelStrDetails = modelName, "with all data and no normalization"
        printMetrics(modelStr, resMetrics, modelStrDetails)
        resultMetrics[modelName] = resMetrics
    return resultMetrics


def allFeatNormalized(df):
    x, y = normalize(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)
    resultMetrics = {}
    
    models = {
        "LinearRegression": LinReg,
        "LassoRegression": LassoReg,
        "RidgeRegression": RidgeReg,
        "KNNRegression": KNNBest
    }

    for modelName, model in models.items():
        y_test_pred = models[modelName](x_train, y_train, x_test)
        resMetrics = calcMetrics(y_test, y_test_pred)
        modelStr, modelStrDetails = modelName, "with all features and normalization"
        printMetrics(modelStr, resMetrics, modelStrDetails)
        resultMetrics[modelName] = resMetrics
    return resultMetrics


def selectFeats(df):
    max_corr_features = ['Len', 'Transac', 'Entities', 'PtsNonAdjust', 'PtsAjust', 'Effort']
    xfeats, y = normalize(df[max_corr_features])
    x_train, x_test, y_train, y_test = train_test_split(xfeats, y, test_size=0.25, random_state=30)
    resultMetrics = {}
    
    models = {
        "LinearRegression": LinReg,
        "LassoRegression": LassoReg,
        "RidgeRegression": RidgeReg,
        "KNNRegression": KNNBest
    }

    for modelName, model in models.items():
        y_test_pred = models[modelName](x_train, y_train, x_test)
        resMetrics = calcMetrics(y_test, y_test_pred)
        modelStr, modelStrDetails = modelName, "with select features and normalization"
        printMetrics(modelStr, resMetrics, modelStrDetails)
        resultMetrics[modelName] = resMetrics
    return resultMetrics


def pcaPlot(df, components):
    pca = PCA(n_components=10)
    df = df.loc[:, df.columns != 'Effort']
    pca_fit = pca.fit(df)

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Variance')
    plt.show()


def pcaTrain(comps, df):
    x, y = normalize(df)
    resultMetrics = {}
    result, y_test = pca_calc(comps, x, y)
    for model, modelPred in result.items():
        resMetrics = calcMetrics(y_test, modelPred)
        modelStrDetails = "with normalization and PCA components for training"
        printMetrics(model, resMetrics, modelStrDetails)
        resultMetrics[model] = resMetrics
    return resultMetrics
#######################################################################################################

def visualization(df):
    cols = df.columns
    boxPlot(df, cols)
    barPlot(df, cols)
    scatterPlotVsEffort(df)
    scatterAmongAll(df)
    correlationData(df)
    effortDistribution(df)


def models(df):
    forPlotMetrics = {}
    forPlotMetrics['all_features'] = allDataModels(df)
    forPlotMetrics['all_feats_normalized'] = allFeatNormalized(df)
    forPlotMetrics['select_features'] = selectFeats(df)
    pcaPlot(df, df.shape[1])
    
    pcaPeakResults = {}
    pcaPeakResults = pcaTrain(3, df)

    for i in range(3, 10):
        print(f"\n*******Using {i} principal components:*******")
        newRes = pcaTrain(i, df)
        pcaPeakRecalc(pcaPeakResults, newRes)
    
    forPlotMetrics['PCA'] = pcaPeakResults
    plotMetricsDriver(forPlotMetrics)
    barPlotsResultsDriver(forPlotMetrics)
#######################################################################################################


if __name__ == "__main__":
    df = readCsv(r"../../datasets/desharnais.csv")
    checkNullFill(df)
    describeData(df)
    visualization(df)
    models(df)
