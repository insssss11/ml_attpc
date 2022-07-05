from .IDataReader import IDataReader
import numpy as np
from math import log
from typing import Dict


def GetBiasInit(y_target):
   nFal, nPos = np.bincount(y_target.astype(np.uint8))
   return log(nPos/nFal)

class OutputDataReader(IDataReader):
    def __init__(self, *outputs):
        super().__init__()
        self.nReactionTypes = None
        self._dataMapFull = {}
        self._ioSpecMapFull = None
        self._lossMapFull = {
            "flg0" : "binary_crossentropy",
            "flg1" : "binary_crossentropy",
            "flg2" : "binary_crossentropy",
            "flg3" : "binary_crossentropy",
            "Ek" : "mse", 
            "mom" : "mse",
            "x" : "mse",
            "y" : "mse",
            "z" : "mse",
            "trkLen" : "mse",
            "theta" : "mse"}
        self._metricMapFull = {
            "flg0" : "acc",
            "flg1" : "recall",
            "flg2" : "recall",
            "flg3" :"recall",
            "Ek" : "mae",
            "mom" : "mae",
            "x" : "mae",
            "y" : "mae",
            "z" : "mae",
            "trkLen" : "mae",
            "theta" : "mae"}
        self._selectedOutputs = {
            "flg0" : True,
            "flg1" : True,
            "flg2" : True,
            "flg3" : True,
            "Ek" : True,
            "mom" : False,
            "x" : True,
            "y" : True,
            "z" : True,
            "trkLen" : True,
            "theta" : True}
        if outputs:
            self.EnableOnly(*outputs)

    def EnableOnly(self, *outputs):
        self._selectedOutputs = {
            "flg0" : False,
            "flg1" : False,
            "flg2" : False,
            "flg3" : False,
            "Ek" : False,
            "mom" : False,
            "x" : False,
            "y" : False,
            "z" : False,
            "trkLen" : False,
            "theta" : False}
        for output in outputs:
            try:
                self._selectedOutputs[output] = True
            except KeyError:
                print("Invalid output name : ", output)

    def GetIoSpecMapFull(self):
        return self._ioSpecMapFull

    def GetMetricMapFull(self):
        return self._metricMapFull

    def GetLossMapFull(self):
        return self._lossMapFull

    def GetSelectedOutputs(self):
        return self._selectedOutputs

    def _InitIOSpecMap(self) -> None:
        self._ioSpecMap = {}
        self.nReactionTypes = self._info["nReactionTypes"]
        self._ioSpecMapFull = {
            "flg0" : (1, np.float32, "sigmoid"),
            "flg1" : (1, np.float32, "sigmoid"),
            "flg2" : (1, np.float32, "sigmoid"),
            "flg3" : (1, np.float32, "sigmoid"),
            "Ek" : ((1), np.float32, None, None), 
            "mom" : ((3), np.float32, None, None),
            "x" : ((1), np.float32, None, None),
            "y" : ((1), np.float32, None, None),
            "z" : ((1), np.float32, None, None),
            "trkLen" : ((1), np.float32, None, None),
            "theta" : ((1), np.float32, None, None)}
        for name, isSelected in self._selectedOutputs.items():
            if isSelected:
                self._ioSpecMap[name] = self._ioSpecMapFull[name]

    def _InitDataMap(self) -> None:
        self.nEvents = self._info["nEvents"]
        
        self._dataMapFull.clear()        
        for key, val in self._ioSpecMapFull.items():
            try:
                self._dataMapFull[key] = np.zeros((self.nEvents, *val[0]), dtype=val[1])
            except TypeError:
                self._dataMapFull[key] = np.zeros((self.nEvents, val[0]), dtype=val[1])        
        for key, val in self._dataMapFull.items():
            if self._selectedOutputs[key]:
                self._dataMap[key] = self._dataMapFull[key]

    def _ReadDataLine(self, dataLine, curEvtId) -> None:
        splitedData = [val for val in dataLine.split() if val]
        try:
            self._dataMapFull["flg0"][curEvtId] = np.float32(splitedData[0])
            self._dataMapFull["flg1"][curEvtId] = np.float32(splitedData[1])
            self._dataMapFull["flg2"][curEvtId] = np.float32(splitedData[2])
            self._dataMapFull["flg3"][curEvtId] = np.float32(splitedData[3])

            self._dataMapFull["Ek"][curEvtId] = np.float32(splitedData[self.nReactionTypes])
            self._dataMapFull["mom"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 1 : self.nReactionTypes + 4])
            self._dataMapFull["x"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 4])
            self._dataMapFull["y"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 5])
            self._dataMapFull["z"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 6])
            self._dataMapFull["theta"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 7])
            self._dataMapFull["trkLen"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 8])
        except Exception as e:
            raise Exception("Failed to read data line %d : "%(curEvtId), e)
        if curEvtId + 1 == self.nEvents:
            for name, spec in self._ioSpecMap.items():
                try:
                    if spec[2] == "sigmoid":
                        self._ioSpecMap[name] = (*spec, GetBiasInit(self._dataMapFull[name].flatten()))
                    else:
                        self._ioSpecMap[name] = (*spec, None)
                except Exception as e:
                    print(e)

    def GetLossMap(self) -> Dict[str, str]:
        lossMap = {}
        for outputName, isSelected in self._selectedOutputs.items():
            if isSelected:
                lossMap[outputName] = self._lossMapFull[outputName]
        return lossMap

    def GetMetricMap(self) -> Dict[str, str]:
        metricMap = {}
        for outputName, isSelected in self._selectedOutputs.items():
            if isSelected:
                metricMap[outputName] = self._metricMapFull[outputName]
        return metricMap