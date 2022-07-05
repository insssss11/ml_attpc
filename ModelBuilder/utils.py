from math import sqrt
import numpy as np
from pathlib import Path

mO = 14899.168636

def ResizeList(l, size : int, value = None):
    if l == None:
        return [value]*size
    elif len(l) > size:
        return l[:size]
    else:
        return tuple(l) + tuple([value]*(size - len(l)))

def GetKinE(mom, mass):
    pp = 0
    try:
        for m in mom:
            pp += m*m
    except TypeError:
        pp = mom*mom
    return sqrt(pp + mass*mass)

def Shuffle(n, input, output):
    idxRnd = np.random.permutation(n)
    inputShuffled, outputShuffled = {}, {}
    for name, data in input.items():
        inputShuffled[name] = data[idxRnd]
    for name, data in output.items():
        outputShuffled[name] = data[idxRnd]    
    return inputShuffled, outputShuffled

def Split(n, input, output, validationSplit):
    trainingInput, trainingOutput, validationInput, validationOutput = {}, {}, {}, {}
    
    for name, data in input.items():
        trainingInput[name] = data[int(n*validationSplit):]
        validationInput[name] = data[:int(n*validationSplit)]

    for name, data in output.items():
        trainingOutput[name] = data[int(n*validationSplit):]
        validationOutput[name] = data[:int(n*validationSplit)]
    return trainingInput, trainingOutput, validationInput, validationOutput

def GetDirRecursive(endWith = None):
    return list(map(str, Path(".").rglob("history")))
    
def DecodeDirName(dir):
    tokens1 = dir.split("/")
    tokens2 = tokens1[0].split("_")
    # depthFirst, kernerSize, dnnUnit, layers
    return int(tokens2[1][1:]), int(tokens2[2][1:]), int(tokens2[3][1:]), int(tokens1[1][-2:])