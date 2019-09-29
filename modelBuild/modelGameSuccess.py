import os

import pandas as pd
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as pp
import matplotlib.backends.backend_pdf as pdf
import matplotlib.gridspec as gs

import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
        fitSet = dataSets["fit"]["fail"][i]
        fitSet = fitSet.append(dataSets["fit"]["success"][i])
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

def makeROC(df, model, nSplits=4):
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
        truePos=[]
        catSizes=testSet.groupby(["category"]).size()
        for thresh in threshArr:
            testSet["predCat"]=testSet["predict"]>thresh
            
            catPredSizes=testSet.groupby(["category", "predCat"]).size()
            
            try:
                falsePos.append(catPredSizes.loc["fail", True]/np.float(catSizes.loc["fail"]))
            except KeyError:
                falsePos.append(0)
            try:
                truePos.append(catPredSizes.loc["success", True]/catSizes.loc["success"])
            except KeyError:
                truePos.append(0)
        
    return np.vstack([falsePos, truePos])

    
appFeatureDir="appFeatures"

app90Days=pd.read_json(os.path.join(appFeatureDir, "90Days.json"))

app180Days=pd.read_json(os.path.join(appFeatureDir, "180Days.json"))

app300Days=pd.read_json(os.path.join(appFeatureDir, "300Days.json"))

rfC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight="balanced_subsample")
logC = LogisticRegression(class_weight="balanced", solver="lbfgs")

roc=makeROC(app90Days, logC, nSplits=4)
pages = pdf.PdfPages("plots/logC_ROC.pdf")
fig = pp.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
sns.lineplot(roc[0,:], roc[1,:], color="b", ax=ax)
sns.lineplot([0, 1], [0, 1], color="grey", ax=ax)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
fig.suptitle("ROC")
pages.savefig(fig)
pp.close(fig)
pages.close()

(fitDCols, fitCatCol)=trainTest(app90Days, rfC, nSplits=4)

with open("../apiServer/modelData/model.pkl", "wb") as fh:
    pickle.dump({"model": rfC, "featureCols": list(fitDCols)}, fh)