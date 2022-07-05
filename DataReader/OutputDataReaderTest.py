from .OutputDataReader import OutputDataReader, GetBiasInit
import numpy as np
from typing import Dict

class OutputDataReaderTest(OutputDataReader):
    def __init__(self, *outputs):
        super().__init__()

        self._lossMapFull["Ebeam"] = "mse"
        self._lossMapFull["Egm"] = "mse"
        self._lossMapFull["thetaGm"] = "mse"
        self._lossMapFull["phiGm"] = "mse"

        self._metricMapFull["Ebeam"] = "mae"
        self._metricMapFull["Egm"] = "mae"
        self._metricMapFull["thetaGm"] = "mae"
        self._metricMapFull["phiGm"] = "mae"

        self._selectedOutputs["Ebeam"] = True
        self._selectedOutputs["Egm"] = True
        self._selectedOutputs["thetaGm"] = True
        self._selectedOutputs["phiGm"] = True
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
            "theta" : False,
            "Ebeam" : False,
            "Egm" : False,
            "thetaGm" : False,
            "phiGm" : False}

        for output in outputs:
            try:
                self._selectedOutputs[output] = True
            except KeyError:
                print("Invalid output name : ", output)


    def _InitIOSpecMap(self) -> None:
        self._ioSpecMap = {}
        self.nReactionTypes = self._info["nReactionTypes"]
        self._ioSpecMapFull = {
            "mom" : ((3), np.float32, None, None),
            "x" : ((1), np.float32, None, None),
            "y" : ((1), np.float32, None, None),
            "z" : ((1), np.float32, None, None),
            "Ek" : ((1), np.float32, None, None),
            "Ebeam" : ((1), np.float32, None, None),
            "Egm" : ((1), np.float32, None, None),
            "trkLen" : ((1), np.float32, None, None),
            "theta" : ((1), np.float32, None, None),
            "thetaGm" : ((1), np.float32, None, None),
            "phiGm" : ((1), np.float32, None, None),
            "flg0" : (1, np.float32, "sigmoid"),
            "flg1" : (1, np.float32, "sigmoid"),
            "flg2" : (1, np.float32, "sigmoid"),
            "flg3" : (1, np.float32, "sigmoid")}
        for name, isSelected in self._selectedOutputs.items():
            if isSelected:
                self._ioSpecMap[name] = self._ioSpecMapFull[name]

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

            self._dataMapFull["Ebeam"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 9])
            self._dataMapFull["Egm"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 10])
            self._dataMapFull["thetaGm"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 11])
            self._dataMapFull["phiGm"][curEvtId] = np.float32(splitedData[self.nReactionTypes + 12])

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