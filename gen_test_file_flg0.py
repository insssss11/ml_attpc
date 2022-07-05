


import sys
from pathlib import Path
import pickle
from typing import final

import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras import Model
import ROOT

from EvaluationTree import EvaluationTree as evaTree
from DataReader import InputDataReader, OutputDataReader

try:
    testDataDir = sys.argv[1]
    nPadX, nPadY = int(sys.argv[2]), int(sys.argv[3])
except IndexError:
    print("Usage : python3 gen_test_file_all.py <testDataDir> <nPadX> <nPadY>")
    exit()

paths = list(Path(".").rglob("model_front.h5"))

with tf.device("/cpu:0"):
    if testDataDir[-1] != "/":
        testDataDir += "/"
    inputDataReader, outputDataReader = InputDataReader(), OutputDataReader()

    # load test input and output data for both models(no shuffle)
    tstNpz = np.load(testDataDir + "test.npz", allow_pickle=True)

    inputInfoTst, outputInfoTst = tstNpz["inputInfo"][()], tstNpz["outputInfo"][()]
    inputSpecsTst, outputSpecsTst = tstNpz["inputSpecs"][()], tstNpz["outputSpecs"][()]
    inputDataTst, outputDataTst = tstNpz["inputData"][()], tstNpz["outputData"][()]
    nTestEvents = tstNpz["nEvents"][()]
    tstNpz = None

    for path in sorted(paths):
        path = path.as_posix()
        path = path[:path.rfind('/')]
        try:

            model = load_model(path + "/model_conv_base.h5")
            e = evaTree(model, outputInfoTst, outputSpecsTst, {
                    "flg0" : True,
                    "flg1" : False,
                    "flg2" : False,
                    "flg3" : False,
                    "Ek" : False,
                    "mom" : False,
                    "x" : False,
                    "y" : False,
                    "z" : False,
                    "trkLen" : False,
                    "theta" : False,
                    "Ebeam" : False,
                    "Egm" : False,
                    "thetaGm" : False,
                    "phiGm" : False})
            rootFile = ROOT.TFile(path + "/eval_flg0.root", "RECREATE")
            tree = e.MakeEvaluationTree(inputDataTst, outputDataTst)
            e = None
            rootFile.Write()
            rootFile.Close()
        except IOError:
            print("Some models are missing in the directory")
        except Exception as e:
            print(path, e)
            
