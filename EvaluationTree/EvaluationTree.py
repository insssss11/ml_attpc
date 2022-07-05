import numpy as np
from array import array
import ROOT
from keras.models import Model
from ModelBuilder import utils

class EvaluationTree:
    def __init__(self, model : Model, outputInfo : dict, outputSpec : dict, selectedOutput = None, nEvents = None):
        self.__pOutputSpec, self.__tOutputSpec = {}, outputSpec
        try:
            if selectedOutput == None:
                self.__pOutputSpec = self.__tOutputSpec
            else:
                for name, isSelected in selectedOutput.items():
                    if isSelected:
                        self.__pOutputSpec[name] = self.__tOutputSpec[name]
            if nEvents == None: 
                self.__nEvents = outputInfo["nEvents"]
            else:
                self.__nEvents = nEvents
        except KeyError as e:
            print("Error contructing EvaluationTree instance : ", e)
            raise e
        self.__branchMap = {}
        self.__model = model
        self.__curEvtId = 0
        self.__progBarLen = 100

    def MakeEvaluationTree(self, testInput : dict, testOutput : dict, predict_output = None) -> ROOT.TTree:
        self.__CheckSizes(testInput, testOutput)
        self.__curEvtId = 0
        tree = ROOT.TTree("tree", "tree for evaluation of compiled keras Model")
        self.__InitTargetBranches(tree)
        self.__InitPredBranches(tree)
        if predict_output == None:
            predict_output = self.__model.predict(testInput)
        print("Generating tree for evaluation ...")
        while self.__curEvtId < self.__nEvents:
            self.__InterpretPrediction(predict_output)
            self.__InterpretTarget(testOutput)
            if self.__curEvtId%(self.__nEvents//self.__progBarLen) == 0:
                self.__PrintProgressBar()
            self.__curEvtId += 1
            tree.Fill()
        print("\nGeneration done")
        return tree

    def __CheckSizes(self, testInput : dict, testOutput : dict):
        for input in testInput.values():
            if self.__nEvents != input.shape[0]:
                raise Exception("Size mismatch in input(" + str(self.__nEvents) + " and " + str(input.shape[0]) + ")")
        for output in testOutput.values():
            if self.__nEvents != output.shape[0]:
                raise Exception("Size mismatch in output(" + str(self.__nEvents) + " and " + str(output.shape[0]) + ")")        

    def __PrintProgressBar(self):
        idx = self.__progBarLen*self.__curEvtId//self.__nEvents
        if idx == 0:
            print("\r|>" + "".join(["*" for _ in range(self.__progBarLen - 1)]) + "|", end="")
        else:
            lStr = "".join(["=" for _ in range(idx)])
            rStr = "".join(["*" for _ in range(self.__progBarLen - idx - 1)])
            print("\r|" + lStr + ">" + rStr + "|", end="")

    def __InitTargetBranches(self, tree : ROOT.TTree) -> None:        
        for name, spec in self.__tOutputSpec.items():
            bName = name[:4]
            bdsc = ""
            dtype = None
            if spec[0] > 1:
                bdsc += "[%d]"%(spec[0])
            if spec[2] == "sigmoid":
                dtype = np.float32
                bdsc += "/F"          
            elif spec[2] == None:
                dtype = np.float32
                bdsc += "/F"
            # [variable name] = (target data, prediction data)
            self.__branchMap[name] = (np.zeros(spec[0], dtype=dtype), np.zeros(spec[0], dtype=dtype))
            tree.Branch(bName + "_t", self.__branchMap[name][0], bName + "_t" + bdsc)
            
    def __InitPredBranches(self, tree : ROOT.TTree):
        for name, spec in self.__tOutputSpec.items():
            bName = name[:4]
            bdsc = ""
            if spec[0] > 1:
                bdsc += "[%d]"%(spec[0])
            if spec[2] == "sigmoid":
                bdsc += "/F"          
            elif spec[2] == None:
                bdsc += "/F"
            tree.Branch(bName + "_p", self.__branchMap[name][1], bName + "_p" + bdsc)        

    def __InterpretTarget(self, target : dict):
        if len(self.__tOutputSpec) != len(target):
            raise Exception("Faied to interpret tartget data : output specifications and target data size mismatch")      
        try:
            for name, data in target.items():
                for idx in range(self.__tOutputSpec[name][0]):
                        self.__branchMap[name][0][idx] = data[self.__curEvtId][idx]
        except ValueError:
            raise Exception("Faied to interpret target data of event " + str(self.__curEvtId))
        except KeyError:
            raise Exception("Key error ocurred intepreting targets.")

    def __InterpretPrediction(self, prediction : list):
        names = list(self.__pOutputSpec.keys())
        specs = list(self.__pOutputSpec.values())
        if len(self.__pOutputSpec) > 1 and len(self.__pOutputSpec) != len(prediction):
            raise Exception("Faied to interpret prediction data : output specifications and prediction data size mismatch, got " ,len(self.__pOutputSpec), "and", len(prediction))
        try:
            if len(self.__pOutputSpec) == 1:
                for idx in range(specs[0][0]):
                    self.__branchMap[names[0]][1][idx] = prediction[self.__curEvtId][idx]
            else:
                for i in range(len(prediction)):
                    for idx in range(specs[i][0]):
                            self.__branchMap[names[i]][1][idx] = prediction[i][self.__curEvtId][idx]
        except ValueError:
            raise Exception("Faied to interpret prediction data of event " + str(self.__curEvtId))
        except KeyError as e:
            raise Exception("Key error ocurred intepreting predictions.", e.args)
        except Exception as e:
            raise e
