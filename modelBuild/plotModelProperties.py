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

with open("../apiServer/modelData/gbCModel.pkl", "rb") as fh:
    modelData=pickle.load(fh)
    
gbC=modelData["model"]
fitDCols=modelData["featureCols"]
testSet=modelData["testData"]
sucFracs=modelData["successFracs"]
binCenters=modelData["scoreBinCenters"]

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
