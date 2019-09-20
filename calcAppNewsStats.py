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


def calcIntervals(df, col):
    if(len(df)<2):
        return pd.DataFrame()
    
    df=df.sort_values(col)
    intervals=np.diff(df[col])
    return pd.DataFrame({col+"Int": list(intervals)})

appListFN="appAttrs/appTable.json"
allNewsFN="appNews/allNews.json"

appList=pd.read_json(appListFN, encoding="utf-8")

appNews=pd.read_json(allNewsFN, encoding="utf-8")
appNews["timeS"]=list(map(lambda d: d.timestamp(), appNews["date"]))
appNews.rename(columns={"appid": "appID"}, inplace=True)

appNews=appNews.merge(appList[["appID", "rTimeS", "status"]])

compAppNews=appNews.set_index(["status"]).loc["finished"].reset_index()
compAppNews["preRel"]=(compAppNews["timeS"]-compAppNews["rTimeS"])<(-3600*24*30)
compAppNews=compAppNews.set_index(["preRel"]).loc[True].reset_index()
compAppNews.drop(["preRel"], 1, inplace=True)

appNews=appNews.set_index(["status"]).loc["active"].reset_index()
appNews=appNews.append(compAppNews)
appNews.reset_index(inplace=True)
appNews.drop(["index"], 1, inplace=True)

appNews.to_json("appNews/filteredAppNews.json")

nNewsPerApp=appNews.groupby("appID").size().reset_index()
nNewsPerApp.rename(columns={0: "nNews"}, inplace=True)

nNewsPerApp.to_json("appAttrs/appNewsCount.json")

appNInts=appNews.groupby(["appID"]).apply(calcIntervals, "timeS").reset_index()
appNIStats=appNInts.groupby(["appID"]).describe()["timeSInt"].reset_index()

appNInts.to_json("appNews/appNewsIntervals.json")

appNIStats.to_json("appAttrs/appNewsIntervalStats.json")

appList=appList.merge(nNewsPerApp)
appList=appList.merge(appNIStats)
appList["mean"]=appList["mean"]/3600/24
appList["std"]=appList["std"]/3600/24
appList["max"]=appList["max"]/3600/24

appList.to_json("appAttrs/appNewsStats.json")

appByStat=appList.set_index(["status"])
pages=pdf.PdfPages("plots/appNews_distributions.pdf")
fig=pp.figure(figsize=(5, 3))
ax=fig.add_subplot(111)
plotD=np.array(appByStat.loc["active", "mean"])
plotD=plotD[(plotD<200) & (plotD>-50)]
sns.distplot(plotD, norm_hist=True, color="r", kde=False, ax=ax)
plotD=np.array(appByStat.loc["finished", "mean"])
plotD=plotD[(plotD<200) & (plotD>-50)]
sns.distplot(plotD, norm_hist=True, color="b", kde=False, ax=ax)
ax.set_xlabel("days")
fig.suptitle("mean update interval")
pages.savefig(fig)
pp.close(fig)

fig=pp.figure(figsize=(5, 3))
ax=fig.add_subplot(111)
plotD=np.array(appByStat.loc["active", "std"])
plotD=plotD[(plotD<200) & (plotD>-50)]
sns.distplot(plotD, norm_hist=True, color="r", kde=False, ax=ax)
plotD=np.array(appByStat.loc["finished", "std"])
plotD=plotD[(plotD<200) & (plotD>-50)]
sns.distplot(plotD, norm_hist=True, color="b", kde=False, ax=ax)
ax.set_xlabel("days")
fig.suptitle("std update interval")
pages.savefig(fig)
pp.close(fig)

fig=pp.figure(figsize=(5, 3))
ax=fig.add_subplot(111)
plotD=np.array(appByStat.loc["active", "max"])
plotD=plotD[(plotD<500) & (plotD>-50)]
sns.distplot(plotD, norm_hist=True, color="r", kde=False, ax=ax)
plotD=np.array(appByStat.loc["finished", "max"])
plotD=plotD[(plotD<500) & (plotD>-50)]
sns.distplot(plotD, norm_hist=True, color="b", kde=False, ax=ax)
ax.set_xlabel("days")
fig.suptitle("max update interval")
pages.savefig(fig)
pp.close(fig)
pages.close()
