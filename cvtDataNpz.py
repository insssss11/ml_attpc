import numpy as np

from Conditions import FlagCut
from DataArrayNpz import CvtTextToNumpy
from ReshapePadData import ReshapePadData, PadPlaneCutCircle, PadPlaneCutLine

planeCut1 = PadPlaneCutCircle((0, 100), 55)
planeCut2 = PadPlaneCutLine(0.5, -10)
planeCut3 = PadPlaneCutLine(0., 80., False)

fileNames = [
        ("./data_npz/40_40_3.6MeV/training.npz",
        "data_txt/40_40_3.6MeV/training_input.dat",
        "data_txt/40_40_3.6MeV/training_output.dat"),
        ("./data_npz/40_40_3.6MeV/test.npz",
        "data_txt/40_40_3.6MeV/test_input.dat",
        "data_txt/40_40_3.6MeV/test_output.dat")]

for npzFile, inputFile, outputFile in fileNames:
        ttn = CvtTextToNumpy()

        ttn.AddCondition("reg", outputCondition=FlagCut("flg0"))
        ttn.AddCondition("sec", outputCondition=(FlagCut("flg1") and FlagCut("flg2") and FlagCut("flg3")))

        print("Generating npz file '" + npzFile + "' ...")
        ttn.SaveAsNpz(npzFile, inputFile, outputFile)

del ttn

fileNames = [
        ("./data_npz/40_40_6.0MeV/training.npz",
        "data_txt/40_40_6.0MeV/training_input.dat",
        "data_txt/40_40_6.0MeV/training_output.dat"),
        ("./data_npz/40_40_6.0MeV/test.npz",
        "data_txt/40_40_6.0MeV/test_input.dat",
        "data_txt/40_40_6.0MeV/test_output.dat")]

for npzFile, inputFile, outputFile in fileNames:
        ttn = CvtTextToNumpy()

        ttn.AddCondition("reg", outputCondition=FlagCut("flg0"))
        ttn.AddCondition("sec", outputCondition=(FlagCut("flg1") and FlagCut("flg2") and FlagCut("flg3")))

        print("Generating npz file '" + npzFile + "' ...")
        ttn.SaveAsNpz(npzFile, inputFile, outputFile)

del ttn

planeCut1 = PadPlaneCutCircle((0, 100), 55)
planeCut2 = PadPlaneCutLine(0.5, -10)
planeCut3 = PadPlaneCutLine(0., 80., False)

reshapePad2X = ReshapePadData(100, 100, 80, 40)
reshapePad2X.SetPadPlaneCut(planeCut1 & planeCut2 & planeCut3)
reshapePad2X.SetNewShape((41, 41))

fileNames = [
        ("./data_npz/80_40_6.0MeV//training.npz",
        "data_txt/80_40_6.0MeV//training_input.dat",
        "data_txt/80_40_6.0MeV//training_output.dat"),
        ("./data_npz/80_40_6.0MeV//test.npz",
        "data_txt/80_40_6.0MeV//test_input.dat",
        "data_txt/80_40_6.0MeV//test_output.dat")]

for npzFile, inputFile, outputFile in fileNames:
        ttn = CvtTextToNumpy()

        ttn.AddCondition("reg", outputCondition=FlagCut("flg0"))
        ttn.AddCondition("sec", outputCondition=(FlagCut("flg1") and FlagCut("flg2") and FlagCut("flg3")))

        print("Generating npz file '" + npzFile + "' ...")
        ttn.SaveAsNpz(npzFile, inputFile, outputFile, reshapePad=reshapePad2X)

del ttn

reshapePad2Y = ReshapePadData(100, 100, 40, 80)
reshapePad2Y.SetPadPlaneCut(planeCut1 & planeCut2 & planeCut3)
reshapePad2Y.SetNewShape((41, 41))

fileNames = [
        ("./data_npz/40_80_6.0MeV/training.npz",
        "data_txt/40_80_6.0MeV/training_input.dat",
        "data_txt/40_80_6.0MeV/training_output.dat"),
        ("./data_npz/40_80_6.0MeV/test.npz",
        "data_txt/40_80_6.0MeV/test_input.dat",
        "data_txt/40_80_6.0MeV/test_output.dat")]

for npzFile, inputFile, outputFile in fileNames:
        ttn = CvtTextToNumpy()

        ttn.AddCondition("reg", outputCondition=FlagCut("flg0"))
        ttn.AddCondition("sec", outputCondition=(FlagCut("flg1") and FlagCut("flg2") and FlagCut("flg3")))

        print("Generating npz file '" + npzFile + "' ...")
        ttn.SaveAsNpz(npzFile, inputFile, outputFile, reshapePad=reshapePad2Y)

del ttn
