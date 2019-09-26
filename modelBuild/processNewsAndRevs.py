import pandas as pd
import numpy as np
import os

def replaceFreeWith0(s):
    try:
        if s.lower()=="free":
            return "$0"
    except:
        return "$0"
    return s

appAttrDir="appAttrs"
appNewsDir="appNews"
appRevsDir="appReviews"

appTable=pd.read_json(os.path.join(appAttrDir, "appTableClean.json"), encoding="utf-8")
selApps=pd.read_json(os.path.join(appAttrDir, "selApps.json"), encoding="utf-8")


selApps=selApps.merge(appTable[["appID", "Price"]])
del(appTable)
selApps["Price"]=list(map(replaceFreeWith0, selApps["Price"]))
selApps["Price"]=list(map(lambda p: np.float(p[1:]), selApps["Price"]))
selApps.to_json(os.path.join(appAttrDir, "selAppsProcessed.json"))

appNews=pd.read_json(os.path.join(appNewsDir, "allNews.json"))
appNews["date"]=list(map(lambda d: d.timestamp(), appNews["date"]))
appNews=appNews[["appid", "date"]].copy()
appNews.rename(columns={"appid": "appID"}, inplace=True)
appNews=appNews.merge(selApps[["appID", "refTS"]])

appNews["t-ref"]=appNews["date"]-appNews["refTS"]
appNews.to_json(os.path.join(appNewsDir, "appNewsProcessed.json"))



appRevs=pd.read_json(os.path.join(appRevsDir, "allReviewsCleaned.json"), encoding="utf-8")
appRevs=appRevs.set_index(["written_during_early_access"]).loc[True].reset_index()
appRevs=appRevs.merge(selApps[["appID", "refTS"]])
appRevs["tC-ref"]=appRevs["tsCreated"]-appRevs["refTS"]
appRevs["tU-ref"]=appRevs["tsUpdated"]-appRevs["refTS"]
appRevs["tDR-tC"]=appRevs["tsDevRes"]-appRevs["tsCreated"]

appRevs.rename(columns={
    "voted_up": "votedUp", "comment_count": "nComments", "votes_up": "votesUp",
    "votes_funny": "votesFunny"}, inplace=True)
appRevs["votedDown"]=~appRevs["votedUp"]
appRevs["devRes"]=list(map(lambda t: ~np.isnan(t), appRevs["tsDevRes"]))
appRevs.to_json(os.path.join(appRevsDir, "appReviewsProcessed.json"))