import json
import time
import os



import getWebAPIData as gwad

import pandas as pd

import numpy as np

def readAppNews(url, header, appID):
    header["appid"]=appID
    for i in np.arange(20):
        try:
            result=gwad.getWebAPIDataJson(url, params=header)
            print(i)
            return result
        except:
            time.sleep(0.25+np.random.random()/2)
            continue
    return False

apiURL="http://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"

apiHeader={"appid": "0", "count": "10000", "maxlength": 1000, "format": "json"}

appListFN="appAttrs/appTableClean.json"

appNewsDir="appNews"

appList=pd.read_json(appListFN, encoding="utf-8")

# allNews=pd.read_json("appNews/allNews.json", encoding="utf-8")

# idsToProc=set(appList["appID"].unique())-set(allNews["appid"].unique())
idsToProc=set(appList["appID"].unique())

doneList=[]
for appID in idsToProc:
    print(appID)
    result=readAppNews(apiURL, apiHeader, appID)
    if(not result):
        print("api error")
        break
    doneList.append(appID)
    
    savePath=os.path.join(appNewsDir, str(appID)+"_RD.json")
    
    with open(savePath, "w") as fh:
        json.dump(result, fh)
    