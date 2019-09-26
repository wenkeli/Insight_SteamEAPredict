import json
import time
import os
import urllib
import copy



import getWebAPIData as gwad

import pandas as pd

import numpy as np

def readReviewBlock(url, header):
    for i in np.arange(10):
        try:
            result=gwad.getWebAPIDataJson(url, params=header)
            print(i)
            return result
        except:
            time.sleep(0.25+np.random.random()/2)
            continue
    return False

def readAppReviews(url, header, appID):
    appURL=url+str(appID)
    appHeader=copy.copy(header)
    cursor=""
    reviewList=[]
    nPosRevs=0
    nNegRevs=0
    i=0
    prevBlock=[]
    while(True):
        if(len(cursor)>0):
            appHeader["cursor"]=cursor
        result=readReviewBlock(appURL, appHeader)
        if(not result):
            return False
        
        if(len(cursor)<=0):
            nPosRevs=int(result["query_summary"]["total_positive"])
            nNegRevs=int(result["query_summary"]["total_negative"])
        
        reviewList.extend(result["reviews"])
        print(result["cursor"])
        cursor=result["cursor"]
        i=i+1
        print(i)
        if(int(result["query_summary"]["num_reviews"])<int(header["num_per_page"])):
            break;
    return {"reviews": reviewList, "nPositive": nPosRevs, "nNegative": nNegRevs}
        

apiURL="https://store.steampowered.com/appreviews/"

maxNPPage=100

apiHeader={"num_per_page": maxNPPage, "purchase_type": "all", "json": 1, "filter": "recent"}

appListFN="appAttrs/appTable.json"

appRevDir="appReviews"

appList=pd.read_json(appListFN, encoding="utf-8")

doneList=[]
idsToProc=set(appList["appID"].unique())-set(doneList)

doneList=[]
for appID in idsToProc:
    print(appID)
    result=readAppReviews(apiURL, apiHeader, appID)
    if(not result):
        break
    doneList.append(appID)
    
    result["appID"]=int(appID)
    savePath=os.path.join(appRevDir, str(appID)+".json")
    
    with open(savePath, "w") as fh:
        json.dump(result, fh)
        
        