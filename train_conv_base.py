from ModelBuilder import ModelBuilder, utils
import keras.backend as K

import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.models import load_model, Model
from keras.metrics import Recall
from keras import callbacks

import numpy as np

from CustomCallback import DynamicLossWeights
from EvaluationTree import EvaluationTree as evaTree

import numpy as np

import pickle
import sys

def DoIt(args):
    try:
        dataDir = args[1]
        outputDir = args[2]
        addtionalConv = int(args[3])
        depthStart = int(args[4])
        kernelSize = int(args[5])
        dnnUnit = int(args[6])
        outputs = args[7:]
        if not addtionalConv > 0:
            raise Exception("addtionalConv must be positive integer")
    except IndexError:
        print("Usage : python3 train_conv_base <dataDir> <outputDir> <addtionalConv> <start depth size> <kerner size> <dnn size> [outputs]")
        exit()
    except Exception as e:
        print(e)
        exit()
    
    # load input and output data for training
    trnNpz = np.load(dataDir + "/training.npz", allow_pickle=True)
    inputInfoTrn, outputInfoTrn = trnNpz["inputInfo"][()], trnNpz["outputInfo"][()]
    inputSpecsTrn, outputSpecsTrn = trnNpz["inputSpecs"][()], {}
    
    outputNames = list(trnNpz["outputSpecs"][()].keys())

    selectedOutputs = {}
    unnamedMetrics = []
    losses, metrics = {}, {}
        
    for outputName in outputNames:
        if outputName in outputs:
            selectedOutputs[outputName] = True
            outputSpecsTrn[outputName] = trnNpz["outputSpecs"][()][outputName]
            losses[outputName] = trnNpz["losses"][()][outputName]
            metrics[outputName] = trnNpz["metrics"][()][outputName]
            if trnNpz["metrics"][()][outputName] == "recall":
                unnamedMetrics.append((outputName, "recall"))
        else:
            selectedOutputs[outputName] = False

    inputDataTrn, outputDataTrn = {}, {}
    idxReg = trnNpz["reg"].astype("bool8")

    if not "flg0" in outputs:
        for inputName in inputSpecsTrn.keys():
            inputDataTrn[inputName] = trnNpz["inputData"][()][inputName][idxReg].copy()
        for outputName in outputSpecsTrn.keys():
            outputDataTrn[outputName] = trnNpz["outputData"][()][outputName][idxReg].copy()
    else:
        for inputName in inputSpecsTrn.keys():
            inputDataTrn[inputName] = trnNpz["inputData"][()][inputName].copy()
        for outputName in outputSpecsTrn.keys():
            outputDataTrn[outputName] = trnNpz["outputData"][()][outputName].copy()

    trnNpz = None

    # load test input and output data for both models(no shuffle)
    tstNpz = np.load(dataDir + "/test.npz", allow_pickle=True)

    inputInfoTst, outputInfoTst = tstNpz["inputInfo"][()], tstNpz["outputInfo"][()]
    inputSpecsTst, outputSpecsTst = tstNpz["inputSpecs"][()], tstNpz["outputSpecs"][()]
    inputDataTst, outputDataTst = tstNpz["inputData"][()], tstNpz["outputData"][()]
    nTestEvents = tstNpz["nEvents"][()]
    tstNpz = None

    # start of loop
    validationSplit = 0.2

    depth = depthStart
    pooling = 2
    print("****************************************************************************************************************")
    
    modelBuilder = ModelBuilder()
    modelBuilder.AddInput(inputSpecsTrn)
    modelBuilder.AddConv2D(depth, kernelSize=kernelSize, padding="SAME", poolingSize=None, normalization=True)
    modelBuilder.AddConv2D(depth, kernelSize=kernelSize, padding="SAME", poolingSize=pooling, normalization=True, dropout=0.5)

    modelBuilder.AddConv2D(2*depth, kernelSize=kernelSize, padding="SAME", poolingSize=None, normalization=True)
    modelBuilder.AddConv2D(2*depth, kernelSize=kernelSize, padding="SAME", poolingSize=None, normalization=True, dropout=0.5)
    modelBuilder.AddConv2D(2*depth, kernelSize=kernelSize, padding="SAME", poolingSize=pooling, normalization=True, dropout=0.5)

    # addtional conv layer

    for i in range(addtionalConv):
        if i == addtionalConv - 1:
            modelBuilder.AddConv2D(4*depth, kernelSize=kernelSize, padding="SAME", poolingSize=pooling, normalization=True)
        else:
            modelBuilder.AddConv2D(4*depth, kernelSize=kernelSize, padding="SAME", poolingSize=None, normalization=True)

    modelBuilder.AddDense(dnnUnit, dropout=0.5)
    modelBuilder.AddOutput(outputSpecsTrn)

    callbackReg = [
        callbacks.TensorBoard(outputDir),
        callbacks.ModelCheckpoint(filepath=outputDir + "/model_conv_base.h5", monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=10)]
    
    lossWeights = None
    if len(outputs) > 1:
        lossWeights = [K.variable(1.) for _ in outputs]
        dynamicLossWeights = DynamicLossWeights(outputs, lossWeights, verbose=2)
        callbackReg.append(dynamicLossWeights)

    print("Compile done. Staring training the model.")
    modelBase = modelBuilder.Build()
    for outputName, metricsName in unnamedMetrics:
        if metricsName.lower() == "recall":
            metrics[outputName] = Recall(name="recall")
    if lossWeights != None:
        modelBase.compile(optimizer="RMSprop", loss=losses, metrics=metrics, loss_weights=[val.numpy() for val in lossWeights])
    else:
        modelBase.compile(optimizer="RMSprop", loss=losses, metrics=metrics)
    
    # modelBase.optimizer.lr = 5e-4
    modelBase.summary()
    plot_model(modelBase, show_shapes=False, to_file=outputDir + "/convBase.png")
    historyReg = modelBase.fit(inputDataTrn, outputDataTrn, validation_split=validationSplit, 
                        shuffle=True,
                        batch_size=516,
                        epochs=400,
                        verbose=0, callbacks=callbackReg)

    layerIdx = 0
    for layer in modelBase.layers:
        if layer.name == "batch_normalization_4":
            idx = layerIdx
            break
        layerIdx += 1
    frontModel = Model(modelBase.input, modelBase.layers[idx].output)
    backModel = Model(modelBase.layers[idx + 1].input, modelBase.layers[-5].output)
    print("Training Done.")
    print("Saving training result...", end=" ")
    frontModel.save(outputDir + "/model_front.h5")
    backModel.save(outputDir + "/model_back.h5")
    frontModel.summary()
    backModel.summary()
    historyDict = {"history" : historyReg,
        "inputSpec" : inputSpecsTst, "outputSpec" : outputSpecsTst,
        "inputInfoTraining" : inputInfoTst, "outputInfoTraining" : outputInfoTst,
        "selectedOutputs": selectedOutputs}
    with open(outputDir+ '/history', 'wb') as historyFile:
        pickle.dump(historyDict, historyFile)        
    print("Done.")
    print("****************************************************************************************************************")

################################## main ####################################################
if "--cpu" in sys.argv:
    with tf.device("/cpu:0"):
        print("Running with CPU")
        DoIt(sys.argv)
else:
    DoIt(sys.argv)

    

