import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as pp
import matplotlib.backends.backend_pdf as pdf
import matplotlib.gridspec as gs

import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


appTableFN="appAttrs/appTableClean.json"
appNewsAgesFN="appAttrs/appNewsAges.json"
appAgesFN="appAttrs/appAges.json"

appTable=pd.read_json(appTableFN, encoding="utf-8")
appNewsAges=pd.read_json(appNewsAgesFN, encoding="utf-8")
appAges=pd.read_json(appAgesFN, encoding="utf-8")

appNewsAges=appNewsAges.merge(appTable[["status", "appID"]])
compAppNAges=appNewsAges.set_index(["status"]).loc["finished"].reset_index()

appAges=appAges.merge(appTable[["status", "appID", "rTimeS"]])
actAppAges=appAges.set_index(["status"]).loc["active"].reset_index()

cARFN=np.sort(np.array(compAppNAges["R-FN"]))
cARFN=cARFN[cARFN>90]
cARFNAppN=np.arange(cARFN.size)
cARFNAppF=cARFNAppN/cARFN.size

aANR=np.sort(np.array(actAppAges["Now-R"]))[::-1]
aANR=aANR[aANR>0]
aANRAppN=np.arange(aANR.size)
aANRAppF=aANRAppN/aANR.size

bins=np.r_[90:2001:100]
aANRHist, _=np.histogram(aANR, bins)
aANRHist=aANRHist/aANR.size

cARFNHist, _=np.histogram(cARFN, bins)
cARFNHist=cARFNHist/cARFN.size

pages = pdf.PdfPages("plots/appDevAges_hist.pdf")
fig = pp.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
ax.bar(bins[:-1], aANRHist, width=90, alpha=0.7, lw=0, color="c", align="edge")
ax.bar(bins[:-1], cARFNHist, width=90, alpha=0.7, lw=0, color="r", align="edge")
# sns.distplot(cARFN, bins=bins, color="r", norm_hist=False, kde=False, ax=ax, hist_kws={"alpha": 0.5})
# sns.lineplot(aANR, aANRAppN, color="r", ax=ax)
ax.axvline(590, lw=2, color="white")
ax.set_xlim([90, 2000])
ax.set_xlabel("age (days)")
ax.set_ylabel("fraction of games")
pages.savefig(fig)
pp.close(fig)
pages.close()

pages = pdf.PdfPages("plots/actApp_nowR_hist.pdf")
fig = pp.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
sns.distplot(aANR, bins=bins, color="c", norm_hist=True, kde=False, ax=ax)
# sns.lineplot(aANR, aANRAppN, color="r", ax=ax)
# ax.axvline(600, lw=2, color="white")
ax.set_xlim([0, 1500])
ax.set_xlabel("days")
ax.set_ylabel("# of games")
pages.savefig(fig)
pp.close(fig)
pages.close()


actApps=actAppAges[actAppAges["Now-R"]>600][["appID", "rTimeS"]].copy()
actApps["category"]="fail"
actApps.rename(columns={"rTimeS": "refTS"}, inplace=True)

compApps=compAppNAges[(compAppNAges["R-FN"]>90) & (compAppNAges["R-FN"]<600)][
    ["appID", "FNTS"]].copy()
compApps["category"]="success"
compApps.rename(columns={"FNTS": "refTS"}, inplace=True)

filteredApps=actApps.append(compApps)
filteredApps=filteredApps.reset_index().drop(["index"], 1)
filteredApps.to_json("appAttrs/selApps.json")