import json
import os

import datetime

import pandas as pd
import numpy as np

def formatTS(ts):
    if(isinstance(ts, pd._libs.tslibs.timestamps.Timestamp)):
        return ts.timestamp()
    return ts

appRevDir="appReviews"

AppAttrDir="appAttrs"

fileNames=os.listdir(appRevDir)

allReviews=pd.DataFrame()
appRevStats=pd.DataFrame()
for fileName in fileNames:
    print(fileName)
    fPath=os.path.join(appRevDir, fileName)

    try:
        with open(fPath, "r") as fh:
            data=json.load(fh)
        appID=data["appID"]
        nPos=data["nPositive"]
        nNeg=data["nNegative"]

        revStat=pd.DataFrame({"appID": [appID], "nPositive": [nPos], "nNegative": [nNeg]})
        appRevStats=appRevStats.append(revStat)

        reviews=pd.DataFrame(data["reviews"])
        reviews["authorNGOwned"]=list(map(lambda a: a["num_games_owned"], reviews["author"]))
        reviews["authorNReviews"] = list(map(lambda a: a["num_reviews"], reviews["author"]))
        reviews["authorID"] = list(map(lambda a: a["steamid"], reviews["author"]))
        reviews["authorPT"] = list(map(lambda a: a["playtime_forever"], reviews["author"]))
        reviews["authorLPTS"] = list(map(lambda a: a["last_played"], reviews["author"]))
        reviews["appID"]=appID
        reviews.drop(["recommendationid", "author", "weighted_vote_score", "review"], 1, inplace=True)
        allReviews=allReviews.append(reviews)

    except:
        continue

allReviews.reset_index(inplace=True)
allReviews.drop(["index"], 1, inplace=True)

appRevStats.reset_index(inplace=True)
appRevStats.drop(["index"], 1, inplace=True)

allReviewsFN="appReviews/allReviews.json"
appRevStatsFN="appAttrs/appRevVoteStat.json"

allReviews.to_json(allReviewsFN)
appRevStats.to_json(appRevStatsFN)

allReviews["tsCreated"]=list(
    map(formatTS, allReviews["timestamp_created"]))
allReviews["tsUpdated"]=list(
    map(formatTS, allReviews["timestamp_updated"]))
allReviews["tsDevRes"]=list(
    map(formatTS, allReviews["timestamp_dev_responded"])
)

allReviews.drop(
    ["timestamp_created", "timestamp_updated", "developer_response",
     "timestamp_dev_responded", "language"], 1, inplace=True)

allReviews.to_json("appReviews/allReviewsCleaned.json")
