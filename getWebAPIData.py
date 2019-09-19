import urllib
import requests
import json


def getWebAPIDataJson(url, params={}):
    response=requests.get(url=url, params=params)
    if(response.status_code==200):
        return json.loads(response.content.decode("utf8"))
    else:
        raise requests.exceptions.RequestException
    