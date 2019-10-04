import os

import pandas as pd
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as pp
import matplotlib.backends.backend_pdf as pdf
import matplotlib.gridspec as gs

import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def calcOutlierFracs(data, thresh=3.29):
    dataM=np.nanmean(data)
    dataStd=np.nanstd(data)
    dataAbsZS=np.abs((data-dataM)/dataStd)
    return np.sum(dataAbsZS>thresh)/dataAbsZS.size
    

def makeTrainValSets(df, catCol, nSplits=4):
    splitFrac=1/nSplits
    cats=df[catCol].unique()
    df=df.set_index(catCol)

    valSet=dict()
    fitSet=dict()

    nRows=dict()
    randInds=dict()
    for cat in cats:
        nRows[cat]=len(df.loc[cat])
        randInds[cat]=np.random.permutation(np.arange(nRows[cat]))

        valSet[cat]=[]
        fitSet[cat]=[]


    for i in np.arange(nSplits):
        startF = i * splitFrac
        endF = (i + 1) * splitFrac
        for cat in cats:
            startN=int(startF*nRows[cat])
            endN=int(endF*nRows[cat])
            valInds=randInds[cat][startN:endN]
            fitInds=list(set(randInds[cat])-set(valInds))
            valSet[cat].append(df.loc[cat].iloc[valInds].reset_index())
            fitSet[cat].append(df.loc[cat].iloc[fitInds].reset_index())

    return {"val": valSet, "fit": fitSet}


def trainTest(df, model, nSplits=4):
    dataSets = makeTrainValSets(df, "category", nSplits=nSplits)
    for i in np.arange(nSplits):
        fitSet = dataSets["fit"]["success"][i]
        
        failFitSet=dataSets["fit"]["fail"][i]
        failFitSet=failFitSet.sample(n=len(fitSet))
        fitSet=fitSet.append(failFitSet)
        fitData=fitSet.drop(["appID", "category", "refTS"], 1)
        fitDCols=fitData.columns
        fitCatCol="category"
        model.fit(fitData, fitSet["category"])

        testSet = dataSets["val"]["fail"][i]
        testSet = testSet.append(dataSets["val"]["success"][i])
        pred = model.predict(testSet[fitDCols])
        testSet["predict"] = pred
        print(testSet.groupby(["category", "predict"]).size())
        
    return (fitDCols, fitCatCol)

def makeModelCurves(df, model, nSplits=4):
    dataSets = makeTrainValSets(df, "category", nSplits=nSplits)
    for i in np.arange(nSplits):
        fitSet = dataSets["fit"]["fail"][i]
        fitSet = fitSet.append(dataSets["fit"]["success"][i])
        fitData=fitSet.drop(["appID", "category", "refTS"], 1)
        fitDCols=fitData.columns
        fitCatCol="category"
        model.fit(fitData, fitSet["category"])

        testSet = dataSets["val"]["fail"][i]
        testSet = testSet.append(dataSets["val"]["success"][i])
        
        pred = model.predict_proba(testSet[fitDCols])
        testSet["predict"] = pred[:,1]
        
        threshArr=np.r_[0.01:1:0.01]
        falsePos=[]
        trueNeg=[]
        truePos=[]
        falseNeg=[]
        catSizes=testSet.groupby(["category"]).size()
        for thresh in threshArr:
            testSet["predCat"]=testSet["predict"]>thresh
            
            catPredSizes=testSet.groupby(["category", "predCat"]).size()
            
            try:
                falsePos.append(catPredSizes.loc["fail", True])
            except KeyError:
                falsePos.append(0)
            try:
                trueNeg.append(catPredSizes.loc["fail", False])
            except KeyError:
                trueNeg.append(0)
            try:
                truePos.append(catPredSizes.loc["success", True])
            except KeyError:
                truePos.append(0)
            try:
                falseNeg.append(catPredSizes.loc["success", False])
            except KeyError:
                falseNeg.append(0)
        falsePos=np.array(falsePos)
        trueNeg=np.array(trueNeg)
        truePos=np.array(truePos)
        falseNeg=np.array(falseNeg)
    return pd.DataFrame({"falsePos": falsePos, "trueNeg": trueNeg, 
            "truePos": truePos, "falseNeg": falseNeg,
            "percision": truePos/(truePos+falsePos),
            "recall": truePos/(truePos+falseNeg),
            "truePosFrac": truePos/(truePos+falseNeg),
            "falsePosFrac": falsePos/(falsePos+trueNeg),
            "thresh": threshArr})

    
appFeatureDir="appFeatures"

app90Days=pd.read_json(os.path.join(appFeatureDir, "90Days.json"))

app180Days=pd.read_json(os.path.join(appFeatureDir, "180Days.json"))

app300Days=pd.read_json(os.path.join(appFeatureDir, "300Days.json"))

rfC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight="balanced_subsample")
gbC = GradientBoostingClassifier(loss="deviance", learning_rate=0.1, n_estimators=100, subsample=1.0, 
       criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
       max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, max_features=None, verbose=0, 
       presort="auto", validation_fraction=0.1, n_iter_no_change=None)

logC = LogisticRegression(penalty="elasticnet", class_weight="balanced", 
                          solver="saga", l1_ratio=0.1, max_iter=10000)

logCV=LogisticRegressionCV(Cs=20, fit_intercept=True, penalty="l2", 
                            max_iter=3000, l1_ratios=np.r_[0:1.01:0.1])
#solver="saga",

# logCParamDict={"C": [0.01, 0.1, 1, 10, 100, 1000], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
#                "class_weight": ["balanced"], "dual": [False], "fit_intercept": [True, False],
#                "intercept_scaling": [0.1, 0.5, 1, 5], "max_iter": [10000], "multi_class": ["warn"],
#                "n_jobs": [None], "penalty": ["elasticnet"], "random_state": [None], "solver": ["saga"],
#                "tol": [0.0001], "verbose":[0], "warm_start": [False]}


dataSizes=app90Days.groupby(["category"]).size().reset_index()
dataSizes.rename(columns={0: "size"}, inplace=True)
pages=pdf.PdfPages("plots/dataSizes.pdf")
fig=pp.figure(figsize=(3, 3))
ax=fig.add_subplot(111)
ax.bar([0, 1], dataSizes["size"], color=["m", "c"])
ax.set_xticks([0, 1])
ax.set_xticklabels(dataSizes["category"])
ax.set_xlabel("")
ax.set_ylabel("# of games")
pages.savefig(fig)
pp.close(fig)
pages.close()

model=logCV
dataSets = makeTrainValSets(app90Days, "category", nSplits=4)

for i in np.arange(4):
    fitSet = dataSets["fit"]["success"][i]
    
    failFitSet=dataSets["fit"]["fail"][i]
    failFitSet=failFitSet.sample(n=len(fitSet))
    fitSet=fitSet.append(failFitSet)
    fitData=fitSet.drop(["appID", "category", "refTS"], 1)
    fitDCols=fitData.columns
    fitCatCol="category"
    model.fit(fitData, fitSet["category"])

    testSet = dataSets["val"]["fail"][i]
    testSet = testSet.append(dataSets["val"]["success"][i])
    pred = model.predict(testSet[fitDCols])
    
    scores=model.predict_proba(testSet[fitDCols])
    scoreDF=pd.DataFrame({"score": scores[:,1], "category": testSet["category"]})
    
    testSet["predict"] = pred
    print(testSet.groupby(["category", "predict"]).size())


pages = pdf.PdfPages("plots/logCV_scores.pdf")
fig = pp.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.hist(scoreDF[scoreDF["category"]=="fail"].score, color="b", alpha=0.3)
ax.hist(scoreDF[scoreDF["category"]=="success"].score, color="r", alpha=0.3)
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_xlabel("false positive rate")
# ax.set_ylabel("true positive rate")
# fig.suptitle("ROC")
pages.savefig(fig)
pp.close(fig)
pages.close()



modelCs=makeModelCurves(app90Days, gbC, nSplits=4)
pages = pdf.PdfPages("plots/gbC_PvR.pdf")
fig = pp.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
sns.lineplot(modelCs["recall"], modelCs["percision"], color="r", ax=ax, lw=7)
ax.axhline(0.2, lw=3, color="blue")
ax.axvline(0.5, lw=2, color="m")
ax.axhline(0.5, lw=2, color="m")
# sns.lineplot(modelCs["thresh"], modelCs["recall"], color="r", ax=ax, lw=2)
# sns.lineplot(modelCs["thresh"], 
#              2*modelCs["percision"]*modelCs["recall"]/(modelCs["percision"]+modelCs["recall"]),
#              ax=ax)
# sns.lineplot([0, 1], [0, 1], color="grey", ax=ax)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("recall")
ax.set_ylabel("percision")
pages.savefig(fig)
pp.close(fig)
pages.close()

(fitDCols, fitCatCol)=trainTest(app90Days, rfC, nSplits=4)

with open("../apiServer/modelData/model.pkl", "wb") as fh:
    pickle.dump({"model": rfC, "featureCols": list(fitDCols)}, fh)