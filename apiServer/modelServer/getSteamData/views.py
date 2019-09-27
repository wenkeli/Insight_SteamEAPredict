from django.shortcuts import render

# Create your views here.
import requests
import json
import time
import datetime
import dateutil.parser as dateParser

import numpy as np
import pandas as pd


def getWebAPIDataJson(url, params={}, tryCount=10, sleepBaseT=1, sleepRandMF=1):
    for i in np.arange(tryCount):
        print(i)
        response=requests.get(url=url, params=params)
        if(response.status_code==200):
            return json.loads(response.content.decode("utf8"))
        else:
            time.sleep(sleepBaseT+np.random.random()*sleepRandMF)
    return False


def getAppDetails(appID):
    url="https://store.steampowered.com/api/appdetails"
    header={"appids": appID}
    return getWebAPIDataJson(url, params=header)
            

def readAppNews(appID):
    url="http://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"
    header={"appid": appID, "count": "10000", "maxlength": 1000, "format": "json"}
    return getWebAPIDataJson(url, params=header)


def readAppReviews(appID):
    url="https://store.steampowered.com/appreviews/"
    header={"num_per_page": 100, "purchase_type": "all", "json": 1, "filter": "recent"}
    appURL=url+str(appID)
    
    cursor=""
    reviewList=[]

    i=0
    while(True):
        if(len(cursor)>0):
            header["cursor"]=cursor
        result=getWebAPIDataJson(appURL, header)
        if(not result):
            return False
        
        reviewList.extend(result["reviews"])
        print(result["cursor"])
        cursor=result["cursor"]
        i=i+1
        print(i)
        if(int(result["query_summary"]["num_reviews"])<int(header["num_per_page"])):
            break;
    return reviewList


def getAppData(appID):
    appDetails=getAppDetails(appID)[str(appID)]["data"]
    appRelTS=dateParser.parse(appDetails["release_date"]["date"]).timestamp()
    try:
        appPrice=appDetails["price_overview"]["final"]/100
    except KeyError:
        appPrice=0
    
    appTable=pd.DataFrame({"appID": [appID], "Price": [appPrice], "refTS": [appRelTS]})
    
    appNews=readAppNews(appID)
    if(len(appNews["appnews"]["newsitems"])>0):
        appNews=pd.DataFrame(appNews["appnews"]["newsitems"])
        appNews=appNews[["gid", "title", "author", "feedlabel", "feedname", "appid", "date"]].copy()
        appNews=appNews.groupby(["appid"]).get_group(appID)
        appNews.rename(columns={"appid": "appID"}, inplace=True)
#         appNews["date"]=list(map(lambda d: d.timestamp(), appNews["date"]))
        appNews=appNews[["appID", "date"]].copy()
        appNews["t-ref"]=appNews["date"]-appRelTS
    else:
        appNews=pd.DataFrame()
    
    appRevs=readAppReviews(appID)
    if(len(appRevs)>0):
        appRevs=pd.DataFrame(appRevs)
        appRevs["authorNGOwned"]=list(map(lambda a: a["num_games_owned"], appRevs["author"]))
        appRevs["authorNReviews"] = list(map(lambda a: a["num_reviews"], appRevs["author"]))
        appRevs["authorID"] = list(map(lambda a: a["steamid"], appRevs["author"]))
        appRevs["authorPT"] = list(map(lambda a: a["playtime_forever"], appRevs["author"]))
        appRevs["authorLPTS"] = list(map(lambda a: a["last_played"], appRevs["author"]))
        appRevs["appID"]=appID
        appRevs.drop(["recommendationid", "author", "weighted_vote_score", "review"], 1, inplace=True)
        appRevs=appRevs.groupby(["written_during_early_access"]).get_group(True)
        appRevs.rename(columns={
            "timestamp_created": "tsCreated", "timestamp_updated": "tsUpdated",
            'voted_up': "votedUp", 'votes_up': "votesUp", 'votes_funny': "votesFunny",
            'comment_count': "nComments"}, inplace=True)
        appRevs["tC-ref"]=appRevs["tsCreated"]-appRelTS
        appRevs["tU-ref"]=appRevs["tsUpdated"]-appRelTS
        if("timestamp_dev_responded" in appRevs.columns):
            appRevs.rename(columns={"timestamp_dev_responded": "tsDevRes"}, inplace=True)
            appRevs["tDR-tC"]=appRevs["tsDevRes"]-appRevs["tsCreated"]
        else:
            appRevs["tsDevRes"]=np.nan
            appRevs["tDR-tC"]=np.nan
        appRevs["devRes"]=list(map(lambda r: ~np.isnan(r), appRevs["tsDevRes"]))
    else:
        appRevs=pd.DataFrame()
    
    
    return {"appTable": appTable, "appNews": appNews, "appRevs": appRevs}


def calcAppFeatures(appTable, appNews, appRevs, startS, endS):
    def formatDFTS(df, tsCol, endT):
        tsArr=np.sort(np.hstack([np.array(df[tsCol]), endT]))
        tsDiff=np.diff(tsArr)
        return pd.DataFrame({tsCol: [tsArr], tsCol+"Diff": [tsDiff]})
    appID=appTable.iloc[0]["appID"]
    aNTCol="t-ref"
    aRTCol="tC-ref"
    if(len(appNews)>0):
        appNews=appNews[(appNews[aNTCol]>=startS) & (appNews[aNTCol]<endS)]
        if(len(appNews)>0):
            nNews=appNews.groupby("appID").size()
            nNews=nNews.reset_index().rename(columns={0: "nNews"})
            newsTS=appNews.groupby("appID").apply(formatDFTS, aNTCol, endS)
            newsTS["maxUpI"]=list(map(lambda a: np.max(a), newsTS[aNTCol+"Diff"]))
            newsTS.reset_index(inplace=True)
            
            appTable=appTable.merge(nNews, how="left")
            appTable=appTable.merge(newsTS[["appID", "maxUpI"]], how="left")
            appTable["0N"]=np.isnan(appTable["nNews"])
            
            appTable=appTable.set_index(["0N"])
            try:
                appTable.loc[True, "nNews"]=0
                appTable.loc[True, "maxUpI"]=endS
            except KeyError:
                pass
            appTable.reset_index(inplace=True)
            appTable.drop(["0N"], 1, inplace=True)
            appTable=appTable.groupby("appID").get_group(appID)
    else:
        appTable["nNews"]=0
        appTable["maxUpI"]=endS
    
    if(len(appRevs)>0):
        appRevs=appRevs[(appRevs[aRTCol]>=startS) & (appRevs[aRTCol]<endS)]
        if(len(appRevs)>0):
            nRevs=appRevs.groupby("appID").size()
            nRevs=nRevs.reset_index().rename(columns={0: "nRevs"})
            nVotes=appRevs.groupby(["appID"]).sum().reset_index()
            nVotes=nVotes[["appID", "votesUp", "votesFunny", "nComments", "devRes"]]
            
            appTable=appTable.merge(nRevs, how="left")
            appTable=appTable.merge(nVotes, how="left")
            appTable["0R"]=np.isnan(appTable["nRevs"])
            appTable=appTable.set_index(["0R"])
            
            nPNRevs=appRevs.groupby(["votedUp", "appID"]).size()
            nPNVotes=appRevs.groupby(["votedUp", "appID"]).sum()
            try:
                appTable.loc[True, "nRevs"]=0
                appTable.loc[True, "votesUp"]=0
                appTable.loc[True, "votesFunny"]=0
                appTable.loc[True, "nComments"]=0
                appTable.loc[True, "devRes"]=0
            except KeyError:
                pass
            appTable.reset_index(inplace=True)
            appTable.drop(["0R"], 1, inplace=True)            
            
            try:
                nPRevs=nPNRevs.loc[True].reset_index().rename(columns={0: "nPRevs"})
                nPVotes=nPNVotes.loc[True][["votesUp", "votesFunny", "nComments", "devRes"]].reset_index()
                nPVotes.rename(columns={"votesUp": "votesUpP", "votesFunny": "votesFunnyP",
                                        "nComments": "nCommentsP", "devRes": "devResP"}, inplace=True)
                appTable=appTable.merge(nPRevs, how="left")
                appTable=appTable.merge(nPVotes, how="left")
                appTable["0R"] = np.isnan(appTable["nPRevs"])
                appTable=appTable.set_index(["0R"])
                try:
                    appTable.loc[True, "nPRevs"]=0
                    appTable.loc[True, "votesUpP"]=0
                    appTable.loc[True, "votesFunnyP"]=0
                    appTable.loc[True, "nCommentsP"]=0
                    appTable.loc[True, "devResP"]=0
                except KeyError:
                    pass
                appTable.reset_index(inplace=True)
                appTable.drop(["0R"], 1, inplace=True)
            except KeyError:
                appTable["nPRevs"]=0
                appTable["votesUpP"]=0
                appTable["votesFunnyP"]=0
                appTable["nCommentsP"]=0
                appTable["devResP"]=0                        
            
            try:
                nNRevs=nPNRevs.loc[False].reset_index().rename(columns={0: "nNRevs"})
                nNVotes=nPNVotes.loc[False][["votesUp", "votesFunny", "nComments", "devRes"]].reset_index()
                nNVotes.rename(columns={"votesUp": "votesUpN", "votesFunny": "votesFunnyN",
                                        "nComments": "nCommentsN", "devRes": "devResN"}, inplace=True)
                appTable=appTable.merge(nNRevs, how="left")
                appTable=appTable.merge(nNVotes, how="left")
                appTable["0R"] = np.isnan(appTable["nNRevs"])
                appTable=appTable.set_index(["0R"])
                try:
                    appTable.loc[True, "nNRevs"]=0
                    appTable.loc[True, "votesUpN"]=0
                    appTable.loc[True, "votesFunnyN"]=0
                    appTable.loc[True, "nCommentsN"]=0
                    appTable.loc[True, "devResN"]=0
                except KeyError:
                    pass
                appTable.reset_index(inplace=True)
                appTable.drop(["0R"], 1, inplace=True)
            except KeyError:
                appTable["nNRevs"]=0
                appTable["votesUpN"]=0
                appTable["votesFunnyN"]=0
                appTable["nCommentsN"]=0
                appTable["devResN"]=0
        
    else:
        appTable["nRevs"]=0
        appTable["votesUp"]=0
        appTable["votesFunny"]=0
        appTable["nComments"]=0
        appTable["devRes"]=0

        appTable["nPRevs"]=0
        appTable["votesUpP"]=0
        appTable["votesFunnyP"]=0
        appTable["nCommentsP"]=0
        appTable["devResP"]=0
    
        appTable["nNRevs"]=0
        appTable["votesUpN"]=0
        appTable["votesFunnyN"]=0
        appTable["nCommentsN"]=0
        appTable["devResN"]=0
        
    appTable=appTable.groupby("appID").get_group(appID)
    return appTable


    