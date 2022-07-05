import math

class PadCenter:
    def __init__(self, padPlaneX, padPlaneY, nPadX, nPadY):
        self.hdim = (padPlaneX/2, padPlaneY/2)
        self.shape = (nPadX, nPadY)
    
    def GetCenterPos(self, xPadNum, yPadNum):
        posX = (2*xPadNum + 1)*self.hdim[0]/nPadX
        posY = (2*yPadNum + 1)*self.hdim[1]/nPadY
        return (posX, posY)

padPlaneX, padPlaneY = 100, 100
nPadX, nPadY = 80, 40

nActi, nInActi = 0, 0

padCenter = PadCenter(padPlaneX, padPlaneY, nPadX, nPadY)

for x in range(nPadX):
    for y in range(nPadY):
        (posX, posY) = padCenter.GetCenterPos(x, y)
        if (posX**2 + (posY - padPlaneY)**2 > 55*55) and (posX/2 - 10) < posY and posY < 80:
            nActi += 1
        else:
            nInActi += 1

print("(Total Pads, active pads, inactive pads) : %d, %d, %d"%(nPadX*nPadY, nActi, nInActi))