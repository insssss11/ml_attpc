import tensorflow as tf
from WeightedAverage import WeightedAverage

from keras import Model
from keras.layers import Input, Average

class EnsenbleBuilder:
    def __init__(self, inputShape, *outputs):
        self.__inputShape = inputShape
        if not outputs:
            raise Exception("The name of outputs must be more than one.")
        self.__nOutputLayers = len(outputs)
        self.__outputLayers = {}
        self.__averageOutputs = []
        self.__weightsMap = {}

        for output in outputs:
            self.__outputLayers[output] = []
            self.__weightsMap[output] = []
        
        self.__models = []

    def __CheckModelValid(self, model : Model):
        if self.__inputShape != model.layers[0].input_shape:
            return True

    def AddModel(self, model : Model, **karg):
        if not self.__CheckModelValid(model):
            raise Exception("Input mismatch : ", self.__inputShape, "!=",  model.layers[0].input_shape)
        self.__models.append(model)
        for output, weight in karg.items():
            self.__weightsMap[output].append(weight)


    def Build(self):
        try:
            inputTensor = Input(shape=self.__inputShape, name="pad", dtype="float32")
            for i in range(len(self.__models)):
                model = self.__models[i]
                outputTensor = inputTensor
                for layer in model.layers[1:-self.__nOutputLayers]:
                    name = layer.name
                    layer._name = name + str(i)
                    outputTensor = layer(outputTensor)
                for layer in model.layers[-self.__nOutputLayers:]:
                    name = layer.name
                    layer._name = name + str(i)
                    self.__outputLayers[name].append(layer(outputTensor))
            for name, outputTensors in self.__outputLayers.items():
                self.__averageOutputs.append(WeightedAverage(self.__weightsMap[name], name=name)(outputTensors))
            return Model(inputTensor, self.__averageOutputs)
        except Exception as e:
            print("Error ocurred in Build() : ", e)