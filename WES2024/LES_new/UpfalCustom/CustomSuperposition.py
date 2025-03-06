from abc import ABC, abstractmethod

from mitwindfarm.Windfield import Windfield
from mitwindfarm.Wake import Wake
from WES2024.LES_new.UpfalCustom.CustomWindfield import AnalyticalSuperimposed as Superimposed


class Superposition(ABC):
    @abstractmethod
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        ...


class Linear(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="linear")


class Niayifar(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="niayifar")
    

class Quadratic(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="quadratic")


class Dominant(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="dominant")