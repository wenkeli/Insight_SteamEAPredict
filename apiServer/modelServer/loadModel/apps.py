from django.apps import AppConfig

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class LoadmodelConfig(AppConfig):
# class LoadmodelConfig:
    name = 'loadModel'
    
    def ready(self):
        global clModel
        global featList
        
        modelFN="../modelData/model.pkl"
        
        with open(modelFN, "rb") as fh:
            modelData=pickle.load(fh)
        clModel=modelData["model"]
        featList=modelData["featureCols"]
        
        print("model loaded")
