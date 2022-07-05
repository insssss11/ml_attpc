from Conditions import ConditionBase

class PadPlaneCutBase(ConditionBase):
    def __init__(self):
        super().__init__()
    
    def _DoEval(self, xPos, yPos, **kargs) -> bool:
        raise NotImplementedError

class PadPlaneCutCircle(PadPlaneCutBase):
    def __init__(self, centerPos, radius, outer=True):
        super().__init__()
        self.centerPos = centerPos
        self.radius = radius
        self.outer = outer

    def _DoEval(self, xPos, yPos, **kargs) -> bool:
        isSatisfied = (xPos - self.centerPos[0])**2 + (yPos - self.centerPos[1])**2 > self.radius**2
        if self.outer:
            return isSatisfied
        else:
            return not isSatisfied

class PadPlaneCutLine(PadPlaneCutBase):
    def __init__(self, slope, yInterc, upper=True):
        super().__init__()
        self.slope, self.yInterc = slope, yInterc
        self.upper = upper

    def _DoEval(self, xPos, yPos, **kargs) -> bool:
        isSatisfied = yPos > self.slope*xPos + self.yInterc
        if self.upper:
            return isSatisfied
        else:
            return not isSatisfied