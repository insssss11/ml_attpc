from abc import abstractmethod
from io import TextIOWrapper
from typing import Tuple
import numpy as np


# This class provides interface to load a training data file
# LoadData() returns two dictionaries.
# The first one is for initializing input or output layer of DNN and the other is a data map containing training data.

# A possible format of data file(line starting with '#' is ignored)
# Header is first two lines of parameters containing how to interprete data and loaded as dictionary as self._info
# [parName1]  [parName2]  ... [parNameN]
# [parValue1] [parValue2] ... [parValueN]

# An user must write a derived class implementing InitDataMap() and InitIOSpecMap() to fill self._dataMap and self._ioSpecMap
class IDataReader:
    def __init__(self):
        self.Reset()
    
    def Reset(self):
        self._info = {}
        self._dataLines = []
        self._ioSpecMap = {} # (name, (shape, dtype, ...))
        self._dataMap = {} # (name, NDarray)
        self.nEvents = 0        

    def GetInfo(self) -> dict:
        return self._info
    
    def LoadData(self, fileName) -> Tuple[dict, dict]:
        self.Reset()
        try:
            txtLines = self.__ReadFile(fileName)
            self.__InitInfo(txtLines[:2])
            self._InitIOSpecMap()
            self._InitDataMap()
            self.__FillDataMap(txtLines[2:])
        except Exception as e:
            print("Error occured in " + self.__class__.__name__ + ".LoadData():\n\t", e)
        finally:
            return self._ioSpecMap, self._dataMap

    def __FillDataMap(self, dataLines) -> None:
        if not self.nEvents == len(dataLines):
            raise Exception("The number of events on the header is not the same with the actual event numbers.")
        for curEvtId in range(self.nEvents):
            self._ReadDataLine(dataLines[curEvtId], curEvtId)

    # This method must fill a line of self._dataMap
    @abstractmethod
    def _ReadDataLine(self, dataLine, curEvtId) -> None:
        pass    

    # This method must initialize self._ioSpecMap.
    @abstractmethod
    def _InitIOSpecMap(self) -> None:
        pass

    def __OpenDataFile(self, fileName) -> TextIOWrapper:
        try:
            file = open(fileName)
        except IOError as e:
            raise Exception("OpenDataFile() : ", e)
        return file

    def __ReadFile(self, fileName) -> list:
        try:
            file = self.__OpenDataFile(fileName)
            txtLines = file.readlines()
            # Remove comments and empty lines
            txtLines = [line[:line.find("#")].strip() for line in txtLines if line[:line.find("#")].strip()]
        except IOError as e:
            raise Exception("Failed to open " + fileName + ".")
        else:
            file.close()
            return txtLines

    # This method initializes self._info.
    def __InitInfo(self, infoLines) -> None:
        if len(infoLines) < 2:
            raise Exception("Failed to initialize data information.")
        keys = [key for key in infoLines[0].split() if len(key) > 0]
        vals = [int(val) for val in infoLines[1].split() if len(val) > 0]
        if not len(keys) == len(vals):
            raise Exception("The numbers of parameter names and values are inconsistent")
        self._info = dict(zip(keys, vals))

    # This method initializes self._dataMap.
    def _InitDataMap(self) -> None:
        self._dataMap.clear()
        self.nEvents = self._info["nEvents"]
        for key, val in self._ioSpecMap.items():
            try:
                self._dataMap[key] = np.zeros((self.nEvents, *val[0]), dtype=val[1])
            except TypeError:
                self._dataMap[key] = np.zeros((self.nEvents, val[0]), dtype=val[1])