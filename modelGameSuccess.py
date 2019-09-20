import urllib
import copy
import datetime

import pandas as pd
import json
import numpy as np

import matplotlib;
matplotlib.use("pdf");

import matplotlib.pyplot as pp;
import matplotlib.backends.backend_pdf as pdf;
import matplotlib.gridspec as gs;

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

matplotlib.rcParams['pdf.fonttype'] = 42;
matplotlib.rcParams['ps.fonttype'] = 42;
    

appAges=pd.read_json("appAttrs/appAges.json", encoding="utf-8")

failSet=appAges.set_index(["status"]).loc["active"].reset_index()
failSet["fail"]=failSet["Now-R"]>(365*1.5)
failSet=failSet.set_index(["fail"]).loc[True].reset_index().drop(["fail"], 1)

successSet=appAges.set_index(["status"]).loc["finished"].reset_index()
successSet["success"]=(successSet["R-FN"]>90) & (successSet["R-FN"]<(365*1.5))
successSet=successSet.set_index(["success"]).loc[True].reset_index().drop(["success"], 1)

appNIStats=pd.read_json("appAttrs/appNewsStats.json", encoding="utf-8")

failSet=failSet.merge(appNIStats[["nNews", "mean", "std", "max", "appID"]], on="appID")
failSet["max"]=list(map(lambda m1, m2: np.max([m1, m2]), failSet["max"], failSet["now-LN"]))

successSet=successSet.merge(appNIStats[["nNews", "mean", "std", "max", "appID"]])

workingSet=failSet.append(successSet)

clf=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(workingSet[["mean", "nNews", "max"]], workingSet["status"])
