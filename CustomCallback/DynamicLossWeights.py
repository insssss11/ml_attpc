import keras.backend as K
from keras.callbacks import Callback

class DynamicLossWeights(Callback):
    def __init__(self, outputNames, lossWeights, relativeWeights = None,
            updateFreqEpochs=10, verbose=0, weightLimit = None):
        if not updateFreqEpochs > 0:
            raise ValueError("updateFreqEpochs must be a positive integer. Got : " + str(updateFreqEpochs))
    
        if not lossWeights:
            raise ValueError("There must be one lossWeights at least.")
        elif not len(outputNames) == len(lossWeights):
            raise ValueError("The sizes of outputNames and lossWeights must be identical.")

        self.nOutputs = len(outputNames)

        if relativeWeights == None:
            self.relativeWeights = [1 for _ in range(self.nOutputs)]
        elif not len(outputNames) == len(relativeWeights):
            raise ValueError("The sizes of outputNames and relativeWeights must be identical.")
        else:
            self.relativeWeights = relativeWeights

        if weightLimit != None:
            if weightLimit <=0 :
                raise ValueError("The weightLimit must be a positive real number.")
            self.weightLimit = weightLimit
        else:
            self.weightLimit = 1e10

        self.lossWeights = lossWeights
        self.outputNames = outputNames
        self.updateFreqEphocs = updateFreqEpochs
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.updateFreqEphocs == 0:
            if self.verbose >= 1:
                print("Updating loss weights at epoch %6d"%epoch)
            
            lossValUnit = self.__GetLoss(self.outputNames[0], logs)
            
            for i in range(len(self.outputNames)):
                lossVal = self.__GetLoss(self.outputNames[i], logs)
                newLossWeight = self.relativeWeights[i]*(lossValUnit/lossVal)
                if newLossWeight < self.weightLimit:
                    K.set_value(self.lossWeights[i], newLossWeight)
                    if self.verbose == 2:
                        print("{0:<10}:{1:8.2f}".format(
                            self.outputNames[i], newLossWeight))
                else:
                    K.set_value(self.lossWeights[i], self.weightLimit)
                    if self.verbose == 2:
                        print("{0:<10}:{1:8.2f}(reached upper limit)".format(
                            self.outputNames[i], self.weightLimit))                    

    def __GetLoss(self, outputName, logs):
        if not outputName + "_loss"  in logs:
            raise AttributeError("No output named '" + outputName + "' in the model.")
        else:
            return logs[outputName + "_loss"]

