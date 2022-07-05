import numpy as np

from .PadPlaneCuts import PadPlaneCutBase

class ReshapePadData:
    def __init__(self, padPlaneX, padPlaneY, nPadX, nPadY):
        self.hdim = (padPlaneX/2, padPlaneY/2)
        self.shape = (nPadX, nPadY)
        self.cut = None
        self.newShape = None

    def SetPadPlaneCut(self, cut : PadPlaneCutBase):
        self.cut = cut

    def SetNewShape(self, newShape):
        self.newShape = newShape
        self.__CreatePadMap(self.newShape, self.cut)
    
    def ReshapePadData(self, padData : np.ndarray) -> np.ndarray:
        try:
            if self.cut == None or self.newShape == None:
                raise Exception("PadPlaneCut or newShape is not set.")
            nEvents = padData.shape[0]
            newPadData = np.zeros(shape=(nEvents, *self.newShape, *padData.shape[3:]), dtype=np.float32)
            padMap = self.__CreatePadMap(self.newShape, self.cut)
            print(newPadData.shape, padData.shape)
            for evt in range(nEvents):
                for xIdx in range(len(padMap)):
                    for yIdx in range(len(padMap[xIdx])):
                        if not padMap[xIdx][yIdx] is None:
                            newPadData[evt, xIdx, yIdx, :] = padData[evt, padMap[xIdx][yIdx][0], padMap[xIdx][yIdx][1], :].copy()

        except Exception as e:
            print("Error occured in RemapPadPlane(...) : ", e)
            raise 
        else:
            return newPadData

    def __CreatePadMap(self, newShape, cut : PadPlaneCutBase):
        try:
            newNpads = newShape[0]*newShape[1]
            padMap = [[None for _ in range(newShape[1])] for _ in range(newShape[0])]
            newPadIdx = 0
            for y in range(self.shape[1]):
                for x in range(self.shape[0]):
                    if newPadIdx == newNpads:
                        raise Exception("The number of pads of new padplane exceeds that of the given shape.")
                    xPos = (2*x + 1)*self.hdim[0]/self.shape[0]
                    yPos = (2*y + 1)*self.hdim[1]/self.shape[1]
                    if cut.Eval(xPos, yPos):
                        xIdx, yIdx = newPadIdx%newShape[0], newPadIdx//newShape[0]
                        padMap[xIdx][yIdx] = (x, y)
                        newPadIdx += 1
        except Exception as e:
            print("Error ocurred in CreatePadMap(...) :", e)
            raise e
        else:
            """
            for y in range(len(padMap[0])):
                for x in range(len(padMap)):
                    print(padMap[x][y], end=" ")
                print("\n")
            """
            print("Active Pad : ", newPadIdx)
            print("Inactive Pad : ", self.shape[1]*self.shape[0] - newPadIdx)
            return padMap