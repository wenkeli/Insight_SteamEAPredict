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
    

appListFN="appAttrs/appTable.json"
allNewsFN="appNews/allNews.json"

appList=pd.read_json(appListFN, encoding="utf-8")

appNews=pd.read_json(allNewsFN, encoding="utf-8")
appNews["timeS"]=list(map(lambda d: d.timestamp(), appNews["date"]))

appFNTS=appNews.groupby(["appid"]).min()["timeS"].reset_index()
appFNTS.rename(columns={"appid": "appID", "timeS": "FNTS"}, inplace=True)

appLNTS=appNews.groupby(["appid"]).max()["timeS"].reset_index()
appLNTS.rename(columns={"appid": "appID", "timeS": "LNTS"}, inplace=True)

appList=appList.merge(appFNTS)
appList=appList.merge(appLNTS)

nSPDay=3600*24
appList["R-FN"]=(appList["rTimeS"]-appList["FNTS"])/nSPDay

nowTS=datetime.datetime.utcnow().timestamp()
appList["Now-R"]=(nowTS-appList["rTimeS"])/nSPDay
appList["now-LN"]=(nowTS-appList["LNTS"])/nSPDay

appList.to_json("appAttrs/appAges.json")


appList=pd.read_json("appAttrs/appAges.json", encoding="utf-8")
plotDir="plots"

appsByStat=appList.set_index(["status"])

pages=pdf.PdfPages("plots/appAge_distributions.pdf")
fig=pp.figure(figsize=(5, 3))
ax=fig.add_subplot(111)
plotD=np.array(appsByStat.loc["active", "Now-R"])
plotD=plotD[(plotD<3000) & (plotD>-50)]
sns.distplot(plotD, hist=True, color="r", kde=False, ax=ax)
ax.set_xlabel("days")
fig.suptitle("age of Early Access games")
pages.savefig(fig)
pp.close(fig)
fig=pp.figure(figsize=(5, 3))
ax=fig.add_subplot(111)
plotD=np.array(appsByStat.loc["finished", "R-FN"])
plotD=plotD[(plotD<2500) & (plotD>-50)]
sns.distplot(plotD, hist=True, color="b", kde=False, ax=ax)
ax.set_xlabel("days")
fig.suptitle("finished games: first news to release")
pages.savefig(fig)
pp.close(fig)
fig=pp.figure(figsize=(5, 3))
ax=fig.add_subplot(111)
plotD=np.array(appsByStat.loc["active", "now-LN"])
plotD=plotD[(plotD<1500) & (plotD>-50)]
sns.distplot(plotD, hist=True, color="r", kde=False, ax=ax)
ax.set_xlabel("days")
fig.suptitle("early access games: age of last news")
pages.savefig(fig)
pp.close(fig)
pages.close()