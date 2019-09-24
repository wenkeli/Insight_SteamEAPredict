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
aANR=aANR[aANR>90]
aANRAppN=np.arange(aANR.size)
aANRAppF=aANRAppN/aANR.size

pages = pdf.PdfPages("plots/compApp_RFN.pdf")
fig = pp.figure(figsize=(5, 7))
ax = fig.add_subplot(111)
sns.lineplot(cARFN, cARFNAppN, color="b", ax=ax)
sns.lineplot(aANR, aANRAppN, color="r", ax=ax)
ax.set_xlim([90, 1500])
ax.set_xlabel("days")
ax.set_ylabel("# games")
fig.suptitle("dev age of completed games")
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