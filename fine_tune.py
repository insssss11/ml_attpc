from ModelBuilder import ModelBuilder, utils
import keras.backend as K

import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.metrics import Recall
from keras import callbacks
from keras import Model, models
from keras.layers import Dense, Input
import numpy as np
import ROOT

from CustomCallback import DynamicLossWeights
from EvaluationTree import EvaluationTree as evaTree

import numpy as np

import pickle
import sys

def DoIt(args):
    try:
        dataDir = args[1]
        modelFile1 = args[2]
        modelFile2 = args[3]
        outputDir = args[4]
        dnnUnit = int(args[5])
    except IndexError:
        print("Usage : python3 train_conv_base_reg <dataDir> <modelFile1> <modelFile2> <outputDir> <dnnUnit>")
        exit()
    models = []

    # load input and output data for training
    trnNpz = np.load(dataDir + "/training.npz", allow_pickle=True)
    inputInfoTrn, outputInfoTrn = trnNpz["inputInfo"][()], trnNpz["outputInfo"][()]
    inputSpecsTrn, outputSpecsTrn = trnNpz["inputSpecs"][()], {}

    outputNames = list(trnNpz["outputSpecs"][()].keys())

    nTraining = 5
    modelNames = ("xy", "z", "EkEbeam", "trkLen", "flg0")
    outputsList = (("x", "y"), ("z",), ("Ek", "Ebeam"), ("trkLen",), ("flg0",))
    monitorList = ("val_loss", "val_mae", "val_loss", "val_mae", "val_acc")
    dynamicList = (True, False, True, False, False)

    model1 = load_model(modelFile1)
    model1.trainable = False

    models = []

    epochs = 400
    validationSplit = 0.2
    for i in range(nTraining):
        outputs = outputsList[i]
        monitor = monitorList[i]
        dynamic = dynamicList[i]
        modelName = modelNames[i]

        selectedOutputs = {}
        unnamedMetrics = []
        losses, metrics = {}, {}
        inputDataTrn, outputDataTrn = {}, {}
        outputSpecsTrn = {}

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

        if monitor == "val_acc":
            for inputName in inputSpecsTrn.keys():
                inputDataTrn[inputName] = trnNpz["inputData"][()][inputName]
            for outputName in outputSpecsTrn.keys():
                outputDataTrn[outputName] = trnNpz["outputData"][()][outputName]           
        else:
            idxReg = trnNpz["reg"].astype("bool8")
            for inputName in inputSpecsTrn.keys():
                inputDataTrn[inputName] = trnNpz["inputData"][()][inputName][idxReg]
            for outputName in outputSpecsTrn.keys():
                outputDataTrn[outputName] = trnNpz["outputData"][()][outputName][idxReg]        

        # fine tuning step1 : Training onlye new dense layer.        
        model2 = load_model(modelFile2)
        model2.trainable = False

        builder = ModelBuilder()
        builder.AddModel(model1)
        builder.AddModel(model2)
        builder.AddDense(dnnUnit, dropout=0.5, dropoutName="dropout_last")
        builder.AddOutput(outputSpecsTrn)
        model = builder.Build()

        print("Compile done. Staring training the model.")
        for outputName, metricsName in unnamedMetrics:
            if metricsName.lower() == "recall":
                metrics[outputName] = Recall(name="recall")

        callback = [
            callbacks.TensorBoard(outputDir),
            callbacks.ModelCheckpoint(filepath=outputDir + "/model_chptr_" + modelName + ".h5" , monitor=monitor, save_best_only=True, verbose=0, save_weights_only=True),
            callbacks.EarlyStopping(monitor=monitor, patience=10)]
        if dynamic:
            lossWeights = [K.variable(1.) for _ in outputs]
            dynamicLossWeights = DynamicLossWeights(outputs, lossWeights, verbose=0)
            callback.append(dynamicLossWeights)
            model.compile(optimizer="RMSprop", loss=losses, metrics=metrics, loss_weights=[val.numpy() for val in lossWeights])
        else:
            model.compile(optimizer="RMSprop", loss=losses, metrics=metrics)
        model.optimizer.lr = 1e-5
        plot_model(model, show_shapes=False, to_file=outputDir + "/fig_" + modelName + ".png")
        print("First fit .............................")
        history = model.fit(inputDataTrn, outputDataTrn, validation_split=validationSplit, 
                            shuffle=True,
                            batch_size=512,
                            epochs=epochs,
                            verbose=0, callbacks=callback)

        # fine tuning step2 : Training dense and top convolution layers.
        model.load_weights(outputDir + "/model_chptr_" + modelName + ".h5")
        model2.trainable = True
        callback = [
            callbacks.TensorBoard(outputDir),
            callbacks.ModelCheckpoint(filepath=outputDir + "/model_chptr_" + modelName + ".h5" , monitor=monitor, save_best_only=True, verbose=0, save_weights_only=True),
            callbacks.EarlyStopping(monitor=monitor, patience=10)]
        if dynamic:
            lossWeights = [K.variable(1.) for _ in outputs]
            dynamicLossWeights = DynamicLossWeights(outputs, lossWeights, verbose=0)
            callback.append(dynamicLossWeights)
            model.compile(optimizer="RMSprop", loss=losses, metrics=metrics, loss_weights=[val.numpy() for val in lossWeights])
        else:
            model.compile(optimizer="RMSprop", loss=losses, metrics=metrics)
        model.optimizer.lr = 1e-5
        print("Second fit .............................")
        history = model.fit(inputDataTrn, outputDataTrn, validation_split=validationSplit, 
                            shuffle=True,
                            batch_size=512,
                            epochs=epochs,
                            verbose=0, callbacks=callback)
        
        # save the history
        historyDict = {"history" : history,
            "inputSpec" : inputSpecsTrn, "outputSpec" : outputSpecsTrn,
            "inputInfoTraining" : inputInfoTrn, "outputInfoTraining" : outputInfoTrn,
            "selectedOutputs": selectedOutputs}
        with open(outputDir+ '/history_' + modelName + "00", 'wb') as historyFile:
            pickle.dump(historyDict, historyFile)   
        
        # save fine tuned model
        print("Saving fine-tuned model :", outputDir + "/model_chptr_" + modelName + ".h5")
        model.load_weights(outputDir + "/model_chptr_" + modelName + ".h5")
        for i in range(len(model.layers)):
            if model.layers[i].name == "batch_normalization_4":
                inputLayer = Input(shape=model.layers[i].output_shape[1:])
                layer = inputLayer
                break
        for j in range(i + 1, len(model.layers)):
            if model.layers[j].name == "dropout_last":
                break
            else:
                layer = model.layers[j](layer)
        outputLayers = []
        for outputLayer in model.layers[j + 1:]:
            outputLayers.append(outputLayer(layer))
        model = Model(inputs=inputLayer, outputs=outputLayers, name="model_" + modelName)
        model.save(outputDir + "/model_chptr_" + modelName + ".h5")
        
        # save the history
        historyDict = {"history" : history,
            "inputSpec" : inputSpecsTrn, "outputSpec" : outputSpecsTrn,
            "inputInfoTraining" : inputInfoTrn, "outputInfoTraining" : outputInfoTrn,
            "selectedOutputs": selectedOutputs}
        with open(outputDir+ '/history_' + modelName + "01", 'wb') as historyFile:
            pickle.dump(historyDict, historyFile)    

        print("Done.")
        print("****************************************************************************************************************")

    # save final model assembly
    for modelName in modelNames:
        models.append(load_model(outputDir + "/model_chptr_" + modelName + ".h5"))

    outputLayers = []
    outputNames = ['x', 'y', 'z', 'Ebeam', 'Ek', 'trkLen', "flg0"]
    
    for model in models:
        outs = model(model1.output)
        try:
            for out in outs:
                outputLayers.append(out)
        except:
                outputLayers.append(outs)
    modelAssy = Model(inputs=model1.input, outputs=outputLayers)
    plot_model(modelAssy, outputDir + "/model_assy2.png", show_layer_names=True, expand_nested=True)
    print("Generating test data...", end=" ")

    # load test input and output data for both models(no shuffle)
    tstNpz = np.load(dataDir + "/test.npz", allow_pickle=True)

    inputInfoTst, outputInfoTst = tstNpz["inputInfo"][()], tstNpz["outputInfo"][()]
    inputSpecsTst, outputSpecsTst = tstNpz["inputSpecs"][()], tstNpz["outputSpecs"][()]
    inputDataTst, outputDataTst = tstNpz["inputData"][()], tstNpz["outputData"][()]
    nTestEvents = tstNpz["nEvents"][()]
    tstNpz = None
    e = evaTree(modelAssy, outputInfoTst, outputSpecsTst,{
            "mom" : False,
            "x" : True,
            "y" : True,
            "z" : True,
            "Ek" : True,
            "Ebeam" : True,
            "Egm" : False,
            "trkLen" : True,
            "theta" : False,
            "thetaGm" : False,
            "phiGm" : False,
            "flg0" : True,
            "flg1" : False,
            "flg2" : False,
            "flg3" : False})
    rootFile = ROOT.TFile(outputDir + "/eval_reg.root", "RECREATE")
    tree = e.MakeEvaluationTree(inputDataTst, outputDataTst)
    rootFile.Write()
    rootFile.Close()
    # modelAssy.save(outputDir + "/model.h5")
    print("Done.")

################################## main ####################################################
if "--cpu" in sys.argv:
    with tf.device("/cpu:0"):
        print("Running with CPU")
        DoIt(sys.argv)
else:
    DoIt(sys.argv)

    

