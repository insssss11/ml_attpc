from __future__ import annotations

from keras import Model
from keras.layers import Input, Layer
import numpy as np

class ModelAssyBuilder:
    def __init__(self, inputShape, outputNames, inputName="input"):
        self.__inputShape = inputShape
        self.__inputName = inputName
        self.__outputNames = outputNames
        
        self.Reset()

    def Reset(self):
        self.__inputLayer = None

        self.__outputLayers = {}
        for outputName in self.__outputNames:
            self.__outputLayers[outputName] = None
        self.__layerNum = 0

    def AddModel(self, model : Model, outputNames):
        if type(outputNames) is str:
            outputNames = [outputNames]

        if self.__inputShape != model.layers[0].input.shape[1:]:
            raise Exception("AddModel : input shape mismatch. Expected " + str(self.__inputShape) +
                ",but got " + str(model.layers[0].input_shape))

        if self.__inputLayer == None:
            self.__inputLayer = Input(shape=self.__inputShape, name=self.__inputName, dtype="float32")
       
        curLayer = self.__inputLayer
        nOutputs = len(outputNames)
        # to make names of hidden layers unique
        for layer in model.layers[1:-nOutputs]:
            layer._name = layer._name + "_" + str(self.__layerNum)
            self.__layerNum += 1
            curLayer = layer(curLayer)

        # check and fill output layers
        for layer in model.layers[-nOutputs:]:
            outputName = layer.name
            if not outputName in outputNames:
                raise Exception("AddModel : Invalid output layer ", outputName)
            elif self.__outputLayers[outputName] != None:
                raise Exception("AddModel : Output duplicated. Got ", outputName)
            self.__outputLayers[outputName] = layer(curLayer)
        
    def Build(self) -> Model:
        if self.__CheckReady():
            raise Exception("Build : Not all output layers added.")
        # this list contains output layers in the same order of output name list given at initializaton.
        orderedOutput = []
        for outputName in self.__outputNames:
            orderedOutput.append(self.__outputLayers[outputName])
        model = Model(self.__inputLayer, orderedOutput)
        self.Reset()
        return model

    def __CheckReady(self):
        for output in self.__outputLayers.values():
            if output == None:
                return False
        

