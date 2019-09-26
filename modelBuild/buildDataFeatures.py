import pandas as pd
import json
import numpy as np
import os

def formatDFTS(df, tsCol, endT):
    tsArr=np.sort(np.hstack([np.array(df[tsCol]), endT]))
    tsDiff=np.diff(tsArr)
    return pd.DataFrame({tsCol: [tsArr], tsCol+"Diff": [tsDiff]})

def calcParams(aT, aNT, aNTCol, aRT, aRTCol, startS, endS):
    aNT=aNT[(aNT[aNTCol]>=startS) & (aNT[aNTCol]<endS)]
    aRT=aRT[(aRT[aRTCol]>=startS) & (aRT[aRTCol]<endS)]

    nNews=aNT.groupby("appID").size()
    nNews=nNews.reset_index().rename(columns={0: "nNews"})
    newsTS=aNT.groupby("appID").apply(formatDFTS, aNTCol, endS)
    newsTS["maxUpI"]=list(map(lambda a: np.max(a), newsTS[aNTCol+"Diff"]))
    newsTS.reset_index(inplace=True)

    aT=aT.merge(nNews, how="left")
    aT=aT.merge(newsTS[["appID", "maxUpI"]], how="left")
    aT["0N"]=np.isnan(aT["nNews"])

    aT=aT.set_index(["0N"])
    aT.loc[True, "nNews"]=0
    aT.loc[True, "maxUpI"]=endS
    aT.reset_index(inplace=True)
    aT.drop(["0N"], 1, inplace=True)

    nRevs=aRT.groupby("appID").size()
    nRevs=nRevs.reset_index().rename(columns={0: "nRevs"})

    nPNRevs=aRT.groupby(["votedUp", "appID"]).size()
    nPRevs=nPNRevs.loc[True].reset_index().rename(columns={0: "nPRevs"})
    nNRevs=nPNRevs.loc[False].reset_index().rename(columns={0: "nNRevs"})

    nVotes=aRT.groupby(["appID"]).sum().reset_index()
    nVotes=nVotes[["appID", "votesUp", "votesFunny", "nComments", "devRes"]]

    nPNVotes=aRT.groupby(["votedUp", "appID"]).sum()
    nPVotes=nPNVotes.loc[True][["votesUp", "votesFunny", "nComments", "devRes"]].reset_index()
    nPVotes.rename(columns={"votesUp": "votesUpP", "votesFunny": "votesFunnyP",
                            "nComments": "nCommentsP", "devRes": "devResP"}, inplace=True)
    nNVotes=nPNVotes.loc[False][["votesUp", "votesFunny", "nComments", "devRes"]].reset_index()
    nNVotes.rename(columns={"votesUp": "votesUpN", "votesFunny": "votesFunnyN",
                            "nComments": "nCommentsN", "devRes": "devResN"}, inplace=True)

    aT=aT.merge(nRevs, how="left")
    aT=aT.merge(nVotes, how="left")
    aT["0R"]=np.isnan(aT["nRevs"])
    aT=aT.set_index(["0R"])
    aT.loc[True, "nRevs"]=0
    aT.loc[True, "votesUp"]=0
    aT.loc[True, "votesFunny"]=0
    aT.loc[True, "nComments"]=0
    aT.loc[True, "devRes"]=0
    aT.reset_index(inplace=True)
    aT.drop(["0R"], 1, inplace=True)

    aT=aT.merge(nPRevs, how="left")
    aT=aT.merge(nPVotes, how="left")
    aT["0R"] = np.isnan(aT["nPRevs"])
    aT=aT.set_index(["0R"])
    aT.loc[True, "nPRevs"]=0
    aT.loc[True, "votesUpP"]=0
    aT.loc[True, "votesFunnyP"]=0
    aT.loc[True, "nCommentsP"]=0
    aT.loc[True, "devResP"]=0
    aT.reset_index(inplace=True)
    aT.drop(["0R"], 1, inplace=True)

    aT=aT.merge(nNRevs, how="left")
    aT=aT.merge(nNVotes, how="left")
    aT["0R"] = np.isnan(aT["nNRevs"])
    aT=aT.set_index(["0R"])
    aT.loc[True, "nNRevs"]=0
    aT.loc[True, "votesUpN"]=0
    aT.loc[True, "votesFunnyN"]=0
    aT.loc[True, "nCommentsN"]=0
    aT.loc[True, "devResN"]=0
    aT.reset_index(inplace=True)
    aT.drop(["0R"], 1, inplace=True)

    return aT


nSPerDay=3600*24

appAttrDir="appAttrs"
appNewsDir="appNews"
appRevsDir="appReviews"
appFeatureDir="appFeatures"

selApps=pd.read_json(os.path.join(appAttrDir, "selAppsProcessed.json"), encoding="utf-8")
appNews=pd.read_json(os.path.join(appNewsDir, "appNewsProcessed.json"), encoding="utf-8")
appRevs=pd.read_json(os.path.join(appRevsDir, "appReviewsProcessed.json"), encoding="utf-8")


features90Days=calcParams(selApps, appNews, "t-ref", appRevs, "tC-ref", 0, 90*nSPerDay)
features90Days.to_json(os.path.join(appFeatureDir, "90Days.json"))

features180Days=calcParams(selApps, appNews, "t-ref", appRevs, "tC-ref", 0, 180*nSPerDay)
features180Days.to_json(os.path.join(appFeatureDir, "180Days.json"))

features300Days=calcParams(selApps, appNews, "t-ref", appRevs, "tC-ref", 0, 300*nSPerDay)
features300Days.to_json(os.path.join(appFeatureDir, "300Days.json"))