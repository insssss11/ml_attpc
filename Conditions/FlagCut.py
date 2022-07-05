from Conditions import ConditionBase

class FlagCut(ConditionBase):
    def __init__(self, flgName):
        super().__init__()          
        self._flgName = flgName

    def _DoEval(self, idx, data, **kargs):
        return data[self._flgName][idx] > 0.5
