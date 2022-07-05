from PlotHistory import PlotHistory

import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import pickle


with tf.device('/cpu:0'):
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print("Usage : python3 plot_training_history.py <model history save file1> ...")
        exit()
    for file in sys.argv[1:]:
        with open(file, 'rb') as historyFile:
            historyDict = pickle.load(historyFile)

        history = historyDict["history"]
        print(history)
        ph = PlotHistory(history, (550, 400))
        ph.factor = 0.5
        ph.startEpoch = 5
        ph.PlotLosses("")
        ph.PlotMetrics("")
        for name in historyDict["outputSpec"].keys():
            ph.PlotLosses(name)
            ph.PlotMetrics(name)
        ph.Show()

