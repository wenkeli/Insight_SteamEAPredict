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

matplotlib.rcParams['pdf.fonttype'] = 42;
matplotlib.rcParams['ps.fonttype'] = 42;


# def estimateAge(releaseTS, firstNewsTS):
#     if(firstNewsTS>releaseTS):
#         return releaseTS
#     else:
#         return firstNewsTS
    

appListFN="appAttrs/appTableClean.json"

appList=pd.read_json(appListFN, encoding="utf-8")

nSPDay=3600*24

nowTS=datetime.datetime(2019, 9, 21).timestamp()
appList["Now-R"]=(nowTS-appList["rTimeS"])/nSPDay

appList=appList[["Now-R", "appID"]].copy()
appList.to_json("appAttrs/appAges.json")
