import os

import pandas as pd
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

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
#         print(testSet.iloc[0])
        
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
#     print(testSet.iloc[0])
    return (pd.DataFrame({"falsePos": falsePos, "trueNeg": trueNeg, 
            "truePos": truePos, "falseNeg": falseNeg,
            "percision": truePos/(truePos+falsePos),
            "recall": truePos/(truePos+falseNeg),
            "truePosFrac": truePos/(truePos+falseNeg),
            "falsePosFrac": falsePos/(falsePos+trueNeg),
            "thresh": threshArr}), testSet, fitDCols)

    
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


modelCs, testSet, fitDCols=makeModelCurves(app90Days, gbC, nSplits=4)

testSetByCat=testSet.set_index("category")
failGProbas=np.array(testSetByCat.loc["fail", "predict"])
sucGProbas=np.array(testSetByCat.loc["success", "predict"])

featureWeights=pd.DataFrame({"weight": gbC.feature_importances_,
                             "feature": fitDCols})
featureWeights.to_csv("plots/featureWs.csv")

binCenters=np.r_[0:1.001:0.01]
sucFracs=[]
for bC in binCenters:
    bCL=bC-0.05
    if bCL<0:
        bCL=0
    bCH=bC+0.05
    if bCH>1:
        bCH=1
    failCount=np.sum((failGProbas>bCL) & (failGProbas<bCH))
    sucCount=np.sum((sucGProbas>bCL) & (sucGProbas<bCH))
    sucFracs.append(sucCount/(sucCount+failCount))
sucFracs=np.array(sucFracs)

print(roc_auc_score(testSet["category"], testSet["predict"]))

with open("../apiServer/modelData/gbCModel.pkl", "wb") as fh:
    pickle.dump({"model": gbC, "featureCols": list(fitDCols), "testData": testSet,
                 "successFracs": sucFracs, "scoreBinCenters": binCenters}, fh)
    
    
    

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

pages = pdf.PdfPages("plots/gbC_F1.pdf")
fig = pp.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
sns.lineplot(modelCs["thresh"], modelCs["recall"], color="r", ax=ax, lw=5)
sns.lineplot(modelCs["thresh"], modelCs["percision"], color="b", ax=ax, lw=5)
sns.lineplot(modelCs["thresh"], 
             2*modelCs["percision"]*modelCs["recall"]/(modelCs["percision"]+modelCs["recall"]),
             color="m", ax=ax, lw=5)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("thresh")
ax.set_ylabel("score")
pages.savefig(fig)
pp.close(fig)
pages.close()

pages = pdf.PdfPages("plots/gbC_ROC.pdf")
fig = pp.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
sns.lineplot(modelCs["falsePos"]/(modelCs["falsePos"]+modelCs["trueNeg"]),
             modelCs["truePos"]/(modelCs["truePos"]+modelCs["falseNeg"]),
             color="r", ax=ax, lw=5)
sns.lineplot([0, 1], [0, 1], color="b", ax=ax, lw=3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("true positive fraction")
ax.set_ylabel("false positive fraction")
pages.savefig(fig)
pp.close(fig)
pages.close()

pages = pdf.PdfPages("plots/gbC_scores.pdf")
fig = pp.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.hist(failGProbas, bins, color="b", alpha=0.8)
ax.hist(sucGProbas, bins, color="r", alpha=0.8)
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
ax.set_xlabel("model score")
ax.set_ylabel("# of games")
# fig.suptitle("ROC")
pages.savefig(fig)
pp.close(fig)
pages.close()


# (fitDCols, fitCatCol)=trainTest(app90Days, rfC, nSplits=4)

# with open("../apiServer/modelData/model.pkl", "wb") as fh:
#     pickle.dump({"model": rfC, "featureCols": list(fitDCols)}, fh)