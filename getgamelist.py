import re

import urllib
import requests
import json

import copy
import os
import pandas as pd

import getWebAPIData as gwad


def extractAppIDFromFile(fName, reExp):
    result=[]
    with open(fName, "r") as fh:
        for line in fh:
            reMatch=reExp.match(line)
            if(reMatch is not None):
                result.append(reMatch.groups()[0])
                
    return result
                
        

alphaNumRE=re.compile("[^a-zA-Z0-9_]+", flags=re.UNICODE)

appIDHtmlRE=re.compile(".*<a href=/app/([0-9]+)>.*", flags=re.UNICODE)

steamSpyURL="https://steamspy.com/api.php"

# steamListURL="http://api.steampowered.com/ISteamApps/GetAppList/v2/"

# steamEACSV="steamSpy_EA_list.csv"
# steamEXEACSV="steamSpy_EX_EA_List.csv"

# appList=pd.DataFrame(gwad.getWebAPIDataJson(url=steamListURL)["applist"]["apps"])
# appList["nameAN"]=list(map(lambda name: str.lower(alphaNumRE.sub("", name)), appList["name"]))
# 
# exEAList=pd.read_csv(steamEXEACSV, encoding="utf8")
# exEAList["nameAN"]=list(map(lambda name: str.lower(alphaNumRE.sub("", name)), exEAList["Game"]))
# 
# eaList=pd.read_csv(steamEACSV, encoding="utf8")
# eaList["nameAN"]=list(map(lambda name: str.lower(alphaNumRE.sub("", name)), eaList["Game"]))

steamEAHtm="steamSpy_EA_List.html"
steamEXEAHtm="steamSpy_EX_EA_List.html"

eaAppIDs=extractAppIDFromFile(steamEAHtm, appIDHtmlRE)
exEAAppIDs=extractAppIDFromFile(steamEXEAHtm, appIDHtmlRE)

isEXEA=[True]*len(exEAAppIDs)
isEXEA.extend([False]*len(eaAppIDs))

appIDs=copy.copy(exEAAppIDs)
appIDs.extend(eaAppIDs)

appIDs=pd.DataFrame({"appID": appIDs, "isEXEA": isEXEA})