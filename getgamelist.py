import re
import datetime
import dateutil

import urllib
import requests
import json

import copy
import os
import pandas as pd
import numpy as np

import pickle as pkl

import getWebAPIData as gwad


def extractAppIDFromFile(fName, appIDRE, nRE):
    result={"appID": [], "N": [], "name": []}
    with open(fName, "r") as fh:
        for line in fh:
            nMatch=nRE.match(line)
            if(nMatch is not None):
                result["N"].append(int(nMatch.groups()[0]))
            else:
                idMatch=appIDRE.match(line)
                if(idMatch is not None):
                    result["name"].append(idMatch.groups()[0])
                    result["appID"].append(idMatch.groups()[1])
    return result

def parseReleaseDate(rd):
    try:
        return dateutil.parser.parse(rd).timestamp()
    except:
        return -1
        

alphaNumRE=re.compile("[^a-zA-Z0-9_]+", flags=re.UNICODE)

appIDHtmlRE=re.compile(".*data-order=\"(.+)\">.*<a href=/app/([0-9]+)>.*", flags=re.UNICODE)
appNHtmlRE=re.compile(".*<td>([0-9]+)</td>.*")

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

steamEA="appAttrs/steamSpy_EA_List"
steamEXEA="appAttrs/steamSpy_EX_EA_List"

appListFN="appAttrs/appTable.json"

eaAppIDs=pd.DataFrame(extractAppIDFromFile(steamEA+".html", appIDHtmlRE, appNHtmlRE))
exEAAppIDs=pd.DataFrame(extractAppIDFromFile(steamEXEA+".html", appIDHtmlRE, appNHtmlRE))

exEAAppIDs["status"]="finished"
eaAppIDs["status"]="active"

eaAppDetails=pd.read_csv(steamEA+".csv", encoding="utf-8")
eaAppDetails.rename(columns={"#": "N"}, inplace=True)
exEAAppDetails=pd.read_csv(steamEXEA+".csv", encoding="utf-8")
exEAAppDetails.rename(columns={"#": "N"}, inplace=True)

eaApp=eaAppIDs.merge(eaAppDetails, on="N")
exEAApp=exEAAppIDs.merge(exEAAppDetails, on="N")

appList=eaApp.append(exEAApp)

appList["nameMatch"]=list(map(lambda n1, n2: n1==n2, appList["name"], appList["Game"]))
appList.drop(["N", "Players", "Score rank(Userscore / Metascore)"], 1, inplace=True)

appList["rTimeS"]=list(map(parseReleaseDate, appList["Release date"]))
appList["hasRelease"]=list(map(lambda ts: ts>0, appList["rTimeS"]))
appList=appList.groupby("hasRelease").get_group(True).copy()
appList.reset_index(inplace=True)
appList.drop(["hasRelease", "index"], 1, inplace=True)

appList.to_json(appListFN)