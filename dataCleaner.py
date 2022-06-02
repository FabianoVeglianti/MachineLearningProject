import numpy as np
import pandas as pd


class dataCleaner:

    def __init__(self):
        self._featuresToBeMantainedList = []
        self._featuresToBeOneHotEncoded = []
        self._emp_length_mapDict = None
        self._target_mapDict = None

    def addFeatureToBeMantained(self, featureName):
        self._featuresToBeMantainedList.append(featureName)

    def set_target_mapDict(self, dict):
        self._target_mapDict = dict

    def encode_target(self, data):
        data.replace(self._target_mapDict, inplace=True)
        data.loan_status.astype(float)

    def removePercentageSign(self, data, featureList):
        for feature in featureList:
            data[feature]=data[feature].str.strip('%').astype(float)

    def convertFromDateToInt(self, data, format, featureList):
        for feature in featureList:
            data[feature] = pd.to_datetime(data[feature], format=format)
            data[feature] = pd.DatetimeIndex(data[feature]).astype(np.int64)*1e-9

    def set_emp_length_mapDict(self, dict):
        self._emp_length_mapDict = dict

    def encode_emp_length(self, data):
        data.replace(self._emp_length_mapDict, inplace=True)

    def addFeatureToBeOneHotEncoded(self, featureName):
        self._featuresToBeOneHotEncoded.append(featureName)

    def applyOneHotEncoding(self, data):
        for feature in self._featuresToBeOneHotEncoded:
            dummies = pd.get_dummies(data[feature])
            dummies.drop(dummies.columns[-1],axis=1,inplace=True) #rimuovo una variabile dummy per ogni feature categorica in modo da non avere collinearità
            data.drop(feature, axis=1, inplace=True) 
            data = pd.concat([data, dummies], axis=1)
        return data #il passaggio di parametro per il tipo Dataframe è per valore, non per riferimento, dunque occorre restituire data per vedere gli effetti

    def convertToFloat(self, data):
        for col in data.columns:
            data[col].astype(float)

