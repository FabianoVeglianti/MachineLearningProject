import numpy as np
import pandas as pd
from pyparsing import one_of


class dataCleaner:

    def __init__(self):
        self._featuresToBeMantainedList = []
        self._featuresToBeOneHotEncoded = []
        self._featureNeedToRemovePercentageList = []
        self._featureToBeConvertedFromDateToIntList = []
        self._emp_length_mapDict = None
        self._target_mapDict = None
        self._dateFormat = None
        self._oneHotEncoder = None
        self._imputer = None
        self._scaler = None
        self._featureSelector = None


    def addFeatureToBeMantained(self, featureName):
        self._featuresToBeMantainedList.append(featureName)

    def set_target_mapDict(self, dict):
        self._target_mapDict = dict

    def encode_target(self, data):
        data.replace(self._target_mapDict, inplace=True)
        data.loan_status = data.loan_status.astype(float)

    def removePercentageSign(self, data, featureList):
        for feature in featureList:
            data[feature]=data[feature].str.strip('%').astype(float)
            if feature not in self._featureNeedToRemovePercentageList:
                self._featureNeedToRemovePercentageList.append(feature)

    def convertFromDateToInt(self, data, format, featureList):
        if self._dateFormat is None:
            self._dateFormat = format
        for feature in featureList:
            data[feature] = pd.to_datetime(data[feature], format=format)
            data[feature] = pd.DatetimeIndex(data[feature]).astype(np.int64)*1e-9
            if feature not in self._featureToBeConvertedFromDateToIntList:
                self._featureToBeConvertedFromDateToIntList.append(feature)

    def set_emp_length_mapDict(self, dict):
        self._emp_length_mapDict = dict

    def encode_emp_length(self, data):
        data.replace(self._emp_length_mapDict, inplace=True)

    def addFeatureToBeOneHotEncoded(self, featureName):
        self._featuresToBeOneHotEncoded.append(featureName)

    def setOneHotEncoder(self, encoder):
        self._oneHotEncoder = encoder

    def applyOneHotEncoding(self, data):
        transformed = self._oneHotEncoder.transform(data)
        data = pd.DataFrame(transformed.toarray(), columns=self._oneHotEncoder.get_feature_names(), index = data.index)
        return data

    def convertToFloat(self, data):
        data = data.astype(float)
        return data

    def set_imputer(self, imputer):
        self._imputer = imputer

    def fillNaN(self, data):
        data[:] = self._imputer.transform(data)
        return data

    def set_scaler(self, scaler):
        self._scaler = scaler

    def scaleData(self, data):
        data = pd.DataFrame(self._scaler.transform(data), columns=data.columns, index=data.index)
        return data

    def set_featureSelector(self, featureSelector):
        self._featureSelector = featureSelector

    def applyFeatureSelection(self, data_x):
        data_x = self._featureSelector.transform(data_x)
        print(data_x.shape)
        return data_x

    """
    cleanData è un metodo pensato per essere usato sul test set.
    Richiede che gli attributi della classe siano stati già settati in precedenza durante il processamento del training set.
    """
    def cleanData(self, data_x, data_y):
        self.encode_target(data_y)

        numericalFeatureList = [feature for feature in self._featuresToBeMantainedList if feature not in self._featuresToBeOneHotEncoded]
        data_x_categorical = data_x.loc[:, self._featuresToBeOneHotEncoded]
        data_x_numerical = data_x.loc[:, numericalFeatureList]

        self.removePercentageSign(data_x_numerical, self._featureNeedToRemovePercentageList)
        self.convertFromDateToInt(data_x_numerical, self._dateFormat, self._featureToBeConvertedFromDateToIntList)
        self.encode_emp_length(data_x_numerical)
        data_x_numerical = self.convertToFloat(data_x_numerical)
        data_x_numerical = self.scaleData(data_x_numerical) 
        #TODO Controllare se scalare anche le categoriche in accordo a quello che facciamo sul training
        print(data_x_numerical.mean())

        data_x_categorical = self.applyOneHotEncoding(data_x_categorical)
        

        data_x = pd.concat([data_x_numerical, data_x_categorical], axis=1)

        self.fillNaN(data_x)
        #data_x = self.applyFeatureSelection(data_x)
        return data_x, data_y


