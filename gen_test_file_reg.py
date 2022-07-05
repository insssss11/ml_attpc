


import sys
from pathlib import Path
import pickle
from typing import final

import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import ROOT

from EvaluationTree import EvaluationTree as evaTree
from DataReader import InputDataReader, OutputDataReaderTest

try:
    testDataDir = sys.argv[1]
    nPadX, nPadY = int(sys.argv[2]), int(sys.argv[3])
except IndexError:
    print("Usage : python3 gen_test_file_all.py <testDataDir> <nPadX> <nPadY>")
    exit()

outputModels = (["model_chkptr_Ek.h5", ("Ek"),], ["model_chkptr_xy.h5", ("x", "y")], ["model_chkptr_z.h5", ("z")], ["model_chkptr_L.h5", ("trkLen")])
if testDataDir[-1] != "/":
    testDataDir += "/"

paths = list(Path(".").rglob("model_chkptr_reg.h5"))

def DoIt():
    inputDataReader, outputDataReader = InputDataReader(), OutputDataReaderTest()

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
            model = load_model(path + "/model_chkptr_reg.h5")
            print(path + "/model_chkptr_reg.h5")
            e = evaTree(model, outputInfoTst, outputSpecsTst, {
                    "flg0" : False,
                    "flg1" : False,
                    "flg2" : False,
                    "flg3" : False,
                    "Ek" : True,
                    "mom" : False,
                    "x" : True,
                    "y" : True,
                    "z" : True,
                    "trkLen" : True,
                    "theta" : False,
                    "Ebeam" : False,
                    "Egm" : False,
                    "thetaGm" : False,
                    "phiGm" : False})
            rootFile = ROOT.TFile(path + "/eval_reg.root", "RECREATE")
            tree = e.MakeEvaluationTree(inputDataTst, outputDataTst)
            e = None
            rootFile.Write()
            rootFile.Close()
        except Exception as e:
            print(path, e)

if "--gpu" in sys.argv:
    DoIt()            
else:
    with tf.device("/cpu:0"):
        DoIt()
