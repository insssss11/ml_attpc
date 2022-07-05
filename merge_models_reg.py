


import sys
from pathlib import Path
import pickle

import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from ModelBuilder.ModelAssyBuilder import ModelAssyBuilder

try:
    nPadX, nPadY = int(sys.argv[1]), int(sys.argv[2])
except IndexError:
    print("Usage : python3 merge_models_reg.py <nPadX> <nPadY>")
    exit()

outputModels = (["model_chkptr_Ek.h5", ("Ek"),], ["model_chkptr_xy.h5", ("x", "y")], ["model_chkptr_z.h5", ("z")], ["model_chkptr_L.h5", ("trkLen")])


paths = list(Path(".").rglob("saved_model.pb"))

def DoIt():
    assy = ModelAssyBuilder(inputShape=(nPadX, nPadY, 2), outputNames=("Ek", "x", "y", "z", "trkLen"), inputName="pad")
    for path in sorted(paths):
        path = path.as_posix()
        path = path[:path.rfind('/')]
        try:
            for i in range(len(outputModels)):
                # outputModels[i].append(load_model(path + "/" + outputModels[i][0]))
                model = load_model(path + "/" + outputModels[i][0])
                assy.AddModel(model, outputModels[i][1])
            model = assy.Build()
            model.save(path + "/model_chkptr_reg.h5")
            print("Merged and saved models in " + path)
        except IOError:
            print("Some models are missing in the directory ", path + "/" + outputModels[i][0])
        except Exception as e:
            print(path, e)
        finally:
            assy.Reset()    

if not "--gpu" in sys.argv:
    with tf.device("/cpu:0"):
        DoIt()
else:
    DoIt()
            
