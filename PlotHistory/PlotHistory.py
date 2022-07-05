from typing import Tuple, Dict, List
from matplotlib import rcParams
import matplotlib.pyplot as plt

from keras import callbacks

class PlotHistory:
    def __init__(self, history : callbacks.History, unitFigPixelSize=(320., 240.)):
        self.__figNum = 0
        self.__historyDict = history.history
        self.__epochs = [i + 1 for i in history.epoch]
        self.__validationExists = "val_loss" in self.__historyDict
        self.__InitMetrics()
        self.__InitFigSize(unitFigPixelSize, rcParams["figure.dpi"])
        self.factor = 0.
        self.startEpoch = 0
        self.endEpoch = len(self.__epochs)
    
    def __InitMetrics(self) -> None:
        self.__metrics = {}
        for key in self.__historyDict.keys():
            if key.find("loss") != -1 or key.find("val") != -1:
                continue
            else:
                idx = key.find("_")
                self.__metrics[key[:idx]] = key[idx + 1:]

    def __InitFigSize(self, unitFigPixelSize, dpi : float):
        self.__unitFigInchSize = (unitFigPixelSize[0]/dpi, unitFigPixelSize[1]/dpi)

    def Show(self):
        plt.show()

    def PlotLosses(self, names = None, shape = None, **karg) -> None:
        # if names is None:
        #     self.__PlotTraining("Loss", )
        if shape is None:
            self.__PlotFigByFig(names, "Loss")
        else:
            self.__PlotSubfigures(names, shape, "Loss", **karg)

    def PlotMetrics(self, names, shape = None, **karg) -> None:
        if shape is None:
            self.__PlotFigByFig(names, "Metric", **karg)
        else:
            self.__PlotSubfigures(names, shape, "Metric", **karg)

    # Exponential moving average
    def __SmoothCurve(self, curve):
        if not 0. <= self.factor < 1.:
            raise Exception("SetMAEFactor() : factor must be a positive float number smaller than one.")
        smoothedCurve = []
        for point in curve:
            if smoothedCurve:
                previous = smoothedCurve[-1]
                smoothedCurve.append(previous*self.factor + point*(1 - self.factor))
            else:
                smoothedCurve.append(point)
        return smoothedCurve

    def __PlotFigByFig(self, names, attName):
        self.__figNum += 1
        if type(names) is str:
            names = [names]
        _attName = None
        for name in names:
            try:
                if attName == "Metric":
                    if name:
                        _attName = self.__metrics[name]
                    elif "acc" in self.__metrics.values():
                        _attName = "acc"
                    elif "mae" in self.__metrics.values():
                        _attName = "mae"
                    else:
                        raise Exception("Invalid att name")
                else:
                    _attName = attName
                trainingName, validationName = self.__MakeNamePair(name, _attName)
                plt.figure(num=self.__figNum, figsize=self.__unitFigInchSize)
                self.__SetAxisLabels("Epochs", _attName)
                self.__PlotTraining(trainingName, _attName)
                if name:
                    name = " of " +  name
                if self.__validationExists:
                    plt.title("Training and validation " + _attName.lower() + name)
                    self.__PlotValidation(validationName, _attName)
                else:
                    plt.title("Training " + _attName.lower() + name)
                plt.legend(loc="upper right")
            except Exception as e:
                print("Error in PlotFigByFig : ", e)

    def __PlotSubfigures(self, names, shape, attName, suptitle=None):
        self.__figNum += 1
        figSize = (self.__unitFigInchSize[0]*shape[1], self.__unitFigInchSize[1]*shape[0])
        plt.figure(num=self.__figNum, figsize=figSize)
        if suptitle == None:
            if self.__validationExists:
                plt.suptitle("Training and validation " + attName.lower())
            else:
                plt.suptitle("Training " + attName.lower())
        else:
                plt.suptitle(suptitle)            

        _attName = None
        (rows, cols) = shape
        for c in range(cols):
            for r in range(rows):
                idx = c + cols*r
                try:
                    if attName == "Metric":
                        if names[idx]:
                            _attName = self.__metrics[names[idx]]
                        elif "acc" in self.__metrics.values():
                            _attName = "acc"
                        elif "mae" in self.__metrics.values():
                            _attName = "mae"
                        else:
                            raise Exception("Invalid att name")
                    else:
                        _attName = attName
                    trainingName, validationName = self.__MakeNamePair(names[idx], _attName)
                    if idx + 1 > rows*cols:
                        raise Exception("Error in PlotSubfigures : The number of graphes exceeds that of subfigures.")
                    plt.subplot(rows, cols, idx + 1)
                    plt.title(names[idx])
                    xlabel, ylabel = None, None
                    if r == rows - 1:
                        xlabel = "Epochs"
                    if c == 0:
                        ylabel = _attName
                    self.__SetAxisLabels(xlabel, ylabel)
                    self.__PlotTraining(trainingName, _attName)
                    if self.__validationExists:
                        self.__PlotValidation(validationName, _attName)
                    plt.legend(loc="upper right")
                except Exception as e:
                    print(e)
                except IndexError:
                    return
            
    def __PlotTraining(self, name, attName : str):
        try:
            xPoints = self.__epochs[self.startEpoch:self.endEpoch]
            yPoints = self.__SmoothCurve(self.__historyDict[name])[self.startEpoch:self.endEpoch]
            if "_" in attName:
                attName = attName[attName.rfind("_") + 1:]
            plt.plot(xPoints, yPoints, 'bo', label="Training " + attName.lower()) 
        except KeyError:
            raise Exception("Failed to find training " + attName.lower() + " of "  + name + " in history")

    def __PlotValidation(self, name, attName : str):
        try:
            xPoints = self.__epochs[self.startEpoch:self.endEpoch]
            yPoints = self.__SmoothCurve(self.__historyDict[name])[self.startEpoch:self.endEpoch]
            if "_" in attName:
                attName = attName[attName.rfind("_") + 1:]
            plt.plot(xPoints, yPoints, 'b', label="Validation " + attName.lower())
        except KeyError:
            raise Exception("Failed to find validation " + attName.lower() + " of "  + name + " in history")

    def __SetAxisLabels(self, xLabel=None, yLabel=None):
        if xLabel != None:  plt.xlabel(xLabel)
        if yLabel != None:  plt.ylabel(yLabel)

    def __MakeNamePair(self, name, attName : str) -> Tuple[str, str]:
        if not name:
            return attName.lower(), "val_" + attName.lower()

        if not name in self.__metrics:
            raise Exception("There is no output named " + name)
        else:
            training_name = name + "_" + attName.lower()
            validation_name = "val_" + name + "_" + attName.lower()
            return training_name, validation_name