from django.shortcuts import render
from django.http import HttpResponse

import json

from loadModel.apps import clModel, featList, successFracs
from getSteamData.views import getAppData, calcAppFeatures

# Create your views here.


def predictSuccess(request):
    appID=int(request.GET["appid"])
    
    appData=getAppData(appID)
    appFeatures=calcAppFeatures(
        appData["appTable"], appData["appNews"], appData["appRevs"], 0, 90*3600*24)
    
    predLabel=clModel.predict(appFeatures[featList])[0]
    predScore=clModel.predict_proba(appFeatures[featList])[0][1]
#     print(successFracs[successFracs["binC"]>predScore].iloc[0])
    predProb=successFracs[successFracs["binC"]>predScore].iloc[0]["sucFracs"]
    predProb=int(predProb*100)
    
    result={"predictCat": predLabel, "successProb": predProb}
    print(result)
    
    return HttpResponse(json.dumps(result))
