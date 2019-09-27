import os

import pandas as pd
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

    
appFeatureDir="appFeatures"

app90Days=pd.read_json(os.path.join(appFeatureDir, "90Days.json"))

app180Days=pd.read_json(os.path.join(appFeatureDir, "180Days.json"))

app300Days=pd.read_json(os.path.join(appFeatureDir, "300Days.json"))

rfC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight="balanced_subsample")
logC = LogisticRegression(class_weight="balanced", solver="lbfgs")

(fitDCols, fitCatCol)=trainTest(app90Days, rfC, nSplits=4)

with open("../apiServer/modelData/model.pkl", "wb") as fh:
    pickle.dump({"model": rfC, "featureCols": list(fitDCols)}, fh)