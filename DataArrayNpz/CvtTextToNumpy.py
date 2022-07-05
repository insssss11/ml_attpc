import numpy as np

from Conditions import FlagCut
from DataReader import InputDataReader, OutputDataReader, OutputDataReaderTest
from ReshapePadData import ReshapePadData

class CvtTextToNumpy:
    def __init__(self):
        self.__conditions = []

    def AddCondition(self, condiName : str, inputCondition=None, outputCondition=None):
        self.__conditions.append((condiName, (inputCondition, outputCondition)))

    def __AddFilteredIdx(self, data : dict):
        for conditionName, (inputCondition, outputCondition) in self.__conditions:
            data[conditionName] = np.full((data["nEvents"]), False, np.bool8)         

        for idx in range(data["nEvents"]):
            for conditionName, (inputCondition, outputCondition) in self.__conditions:
                satisfied1 = True if inputCondition == None else inputCondition.Eval(idx, data["inputData"])
                satisfied2 = True if outputCondition == None else outputCondition.Eval(idx, data["outputData"])
                data[conditionName][idx] = satisfied1 and satisfied2        

    def SaveAsNpz(self, dstName, inputTxtFile, outputTxtFile, compressed=False, reshapePad=None):
        try:
            data= {
                "inputSpecs" : None, "inputInfo" : None, "inputData" : None,
                "outputSpecs" : None, "outputInfo" : None, "outputData" : None,
                "losses" : None, "metrics" : None,
                "nEvents" : 0
            }
            inputReader, outputReader = InputDataReader(), OutputDataReaderTest()
            (data["inputSpecs"], data["inputData"]) = inputReader.LoadData(inputTxtFile)
            (data["outputSpecs"], data["outputData"]) = outputReader.LoadData(outputTxtFile)
            
            data["inputInfo"], data["outputInfo"] = inputReader.GetInfo(), outputReader.GetInfo()
            data["losses"], data["metrics"] = outputReader.GetLossMapFull(), outputReader.GetMetricMapFull()
            
            assert inputReader.nEvents == outputReader.nEvents
            nEvents = data["nEvents"] = outputReader.nEvents
            
            idxRnd = np.random.permutation(nEvents)
            
            for inputName, dat in data["inputData"].items():
                data["inputData"][inputName] = dat[idxRnd]
            
            if not reshapePad is None:
                data["inputData"]["pad"] = reshapePad.ReshapePadData(data["inputData"]["pad"])
                data["inputSpecs"]["pad"] = ((reshapePad.newShape[0], reshapePad.newShape[1], 2), np.float32)                
                (data["inputInfo"]["inputX"], data["inputInfo"]["inputY"]) = reshapePad.newShape
                data["inputInfo"]["inputSize"] = reshapePad.newShape[0]*reshapePad.newShape[1]
            for outputName, dat in data["outputData"].items():
                data["outputData"][outputName] = dat[idxRnd]

            self.__AddFilteredIdx(data)

            if compressed:
                np.savez_compressed(file=dstName, **data)
            else:
                np.savez(file=dstName, **data)
        except Exception as e:
            print("Error occured in SaveAsNpz : ", e)
            raise e
