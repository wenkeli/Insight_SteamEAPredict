from django.apps import AppConfig

import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class LoadmodelConfig(AppConfig):
# class LoadmodelConfig:
    name = 'loadModel'
    
    def ready(self):
        global clModel
        global featList
        global successFracs
        
        modelFN="../modelData/gbCModel.pkl"
        
        with open(modelFN, "rb") as fh:
            modelData=pickle.load(fh)
        clModel=modelData["model"]
        featList=modelData["featureCols"]
        successFracs=pd.DataFrame({"binC": modelData["scoreBinCenters"],
                                   "sucFracs": modelData["successFracs"]})
        successFracs.sort_values("binC", inplace=True)
        
        print("model loaded")
