from .IDataReader import IDataReader
import numpy as np

class InputDataReader(IDataReader):
    def _InitIOSpecMap(self) -> None:
        self._ioSpecMap.clear()
        self.__nPadX, self.__nPadY = self._info["inputX"], self._info["inputY"]
        self._ioSpecMap["pad"] = ((self.__nPadX, self.__nPadY, 2), np.float32)

    def _ReadDataLine(self, dataLine, curEvtId) -> None:
        nonZeroPairs = [value[1:] for value in dataLine.split(")") if value]
        data = np.zeros((self.__nPadX*self.__nPadY, 2), dtype=np.float32)
        for nonZeroPair in nonZeroPairs:
            nonZeroPair = nonZeroPair.split(",")
            data[int(nonZeroPair[0])][0] = np.float32(nonZeroPair[1])
            data[int(nonZeroPair[0])][1] = np.float32(nonZeroPair[2])
        self._dataMap["pad"][curEvtId] = data.reshape((self.__nPadX, self.__nPadY, 2))        
