from __future__ import annotations

class ConditionBase:
    def __init__(self, invert = False):
        self.__inverted = invert

    def Eval(self, *args, **kargs) -> bool:
        if self.__inverted:
            return ~self._DoEval(*args, **kargs)
        else:
            return self._DoEval(*args, **kargs)

    def _DoEval(self, *args, **kargs) -> bool:
        raise NotImplementedError()

    def Invert(self, invert=True):
        self.__inverted = invert

    def __and__(self, condition : ConditionBase):
        return AndCondition(self, condition)

    def __or__(self, condition : ConditionBase):
        return OrCondition(self, condition)

class TrueCondition(ConditionBase):
    def __init__(self, **kargs):
        super().__init__(**kargs)
    
    def _DoEval(self) -> bool:
        return True

class AndCondition(ConditionBase):
    def __init__(self, *conditions : ConditionBase, **kargs):
        super().__init__(**kargs)
        if not conditions:
            raise Exception("AndCondition must be initialized with one or more derived classes of ConditionBase")            
        self.__conditions = conditions

    def _DoEval(self, *args, **kargs) -> bool:
        for condi in self.__conditions:
            if not condi.Eval(*args, **kargs):
                return False
        return True

class OrCondition(ConditionBase):
    def __init__(self, *conditions : ConditionBase, **kargs):
        super().__init__(**kargs)
        if not conditions:
            raise Exception("OrCondition must be initialized with one or more derived classes of ConditionBase")            
        self.__conditions = conditions

    def _DoEval(self, *args, **kargs) -> bool:
        for condi in self.__conditions:
            if condi.Eval(*args, **kargs):
                return True
        return False

