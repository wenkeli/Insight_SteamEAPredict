import json
import datetime
import os

import pandas as pd
import numpy as np

appNewsDir="appNews"

fileNames=os.listdir(appNewsDir)

allNews=pd.DataFrame()
for fileName in fileNames:
    print(fileName)
    fPath=os.path.join(appNewsDir, fileName)
    
    try:
        with open(fPath, "r") as fh:
            data=json.load(fh)
        data=pd.DataFrame(data["appnews"]["newsitems"])
        data=data[["gid", "title", "author", "feedlabel", "feedname", "appid", "date"]]
        allNews=allNews.append(data)
    except:
        continue

allNewsFN="appNews/allNews.json"

allNews.drop_duplicates(subset=["gid"], inplace=True)
allNews.drop(["gid"], 1, inplace=True)
allNews.reset_index(inplace=True)
allNews.drop(["index"], 1, inplace=True)
allNews.to_json(allNewsFN)