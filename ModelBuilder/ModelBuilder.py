from __future__ import annotations

from keras import models, Model
from keras import layers
from keras.layers import Input
import keras.initializers
import numpy as np

class ModelBuilder:
    def __init__(self):
        self.__inputLayers = []
        self.__inputSpecs = []

        self.__outputLayers = []
        self.__outputSpecs = []
        
        self._curLayer = None

    def Build(self) -> models.Model:
        model = None
        try:
            self.__ConstructOutput()
            model = models.Model(self.__inputLayers, self.__outputLayers)
        except Exception as e:
            print("Error in ModelBuilder.Build : ", e)
            raise e
        finally:
            self.Clear()
            return model

    def Clear(self):
        self.__inputLayers = []
        self.__inputSpecs = []
        self.__outputLayers = []
        self.__outputSpecs = []
        self._curLayer = None        

    # inputSpecMap : dict[name, (shape, dtype)]
    def AddInput(self, inputSpecMap : dict[str, (int, np.dtype)]) -> ModelBuilder:
        for name, (shape, type) in inputSpecMap.items():
            self.__inputSpecs.append((name, shape, type))
        return self

    def AddModel(self, model : Model):
        try:
            if self._curLayer == None:
                self.__inputLayers = model.input
                self._curLayer = model.output
            else:
                self._curLayer = model(self._curLayer)
        except Exception as e:
            print(e)
        finally:
            return self

    def AddDense(self, units : int, dropout = None, dropoutName = None, normalization = None) -> ModelBuilder:
        try:
            kerInit = self.__GetKernalInit("relu")
            if self._curLayer == None:
                self.__ConstructInput()
            if len(self._curLayer.get_shape().dims) > 2:
                self._curLayer = layers.Flatten()(self._curLayer)
            self._curLayer = layers.Dense(units, activation="relu", kernel_initializer=kerInit)(self._curLayer)
            if dropout != None:
                self._curLayer = layers.Dropout(dropout, name=dropoutName)(self._curLayer)
            if normalization:
                self._curLayer = layers.BatchNormalization()(self._curLayer)
        except Exception as e:
            print(e)
        finally:
            return self

    def AddConv2D(self, filter, kernelSize, padding="valid", poolingSize=None, poolingName=None, dropout = None, normalization = None) -> ModelBuilder:
        try:
            kerInit = self.__GetKernalInit("relu")
            if self._curLayer == None:
                self.__ConstructInput()
            if len(self._curLayer.get_shape().dims) < 3:
                raise Exception("Error : a Conv2D layer must be added to a layer with 2-dimensional output.")
            self._curLayer = layers.Conv2D(filter, kernelSize, padding=padding, activation="relu", kernel_initializer=kerInit)(self._curLayer)
            if poolingSize != None:
                self._curLayer = layers.MaxPooling2D(poolingSize, name=poolingName)(self._curLayer)
            if dropout != None:
                self._curLayer = layers.Dropout(dropout)(self._curLayer)
            if normalization:
                self._curLayer = layers.BatchNormalization()(self._curLayer)
        except Exception as e:
            print(e)
        return self

    # outputSpecMap : dict[name, (dimension, dtype, activation)]
    def AddOutput(self, outputSpecMap : dict[str, (int, np.dtype, str)]) -> ModelBuilder:
        for name, spec in outputSpecMap.items():
            self.__outputSpecs.append((name, spec))
        return self

    def __ConstructInput(self) -> None:
        nInputs = len(self.__inputSpecs)
        x = None
        if not nInputs > 0:
            raise Exception("No input set, add an input calling AddInput()")
        elif nInputs == 1:
            (inputName, shape, type) = self.__inputSpecs[0]
            try:
                if len(shape) == 2:
                    self.__inputLayers = Input(shape=(*shape, 1), name=inputName, dtype=type)
                else:
                    self.__inputLayers = Input(shape=shape, name=inputName, dtype=type)
            except TypeError:
                self.__inputLayers = Input(shape=(shape,), name=inputName, dtype=type)
            self._curLayer = self.__inputLayers
        else:
            for (inputName, shape, type) in self.__inputSpecs:
                x = Input(shape=(*shape,), name=inputName, dtype=type)
                if len(shape) > 1:
                    print("Warning : multiple inputs with more than 2 inputs. Concatenating after flattening...")
                    x = layers.Flatten()(x)
                self.__inputLayers.append(x)
            self._curLayer = layers.concatenate(self.__inputLayers)

    def __ConstructOutput(self) -> None:
        if not len(self.__outputSpecs) > 0:
            raise Exception("No output set, add an output calling AddOutput()")
        for outputName, spec in self.__outputSpecs:
            shape, dtype, acti, bias_init = spec[0], spec[1], spec[2], spec[3]
            kerInit = self.__GetKernalInit(acti)
            if bias_init != None:
                kerInit = keras.initializers.Constant(bias_init)
            self.__outputLayers.append(layers.Dense(shape, activation=acti, dtype=dtype, name=outputName,
                bias_initializer=kerInit)(self._curLayer))


    def __GetKernalInit(self, acti):
        if acti == None or acti == "sigmoid" or acti == "softmax":
            return "glorot_normal"
        elif acti == "relu":
            return "he_normal"
        else:
            raise Exception("Invalid activationi function name : " + acti)

