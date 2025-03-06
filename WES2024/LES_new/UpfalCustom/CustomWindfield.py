import numpy as np
from numpy.typing import ArrayLike
from mitwindfarm import Superimposed
from mitwindfarm.Windfield import Windfield

class LogLaw(Windfield):
    def __init__(self, Ufric: float, z0: float, TIamb: float = 0.0, kappa:float = 0.41):
        self.Ufric = Ufric
        self.z0 = z0
        self.TIamb = TIamb
        self.kappa = kappa

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        u = (self.Ufric / self.kappa) * np.log(z / self.z0)
        u = np.nan_to_num(u)
        return u

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return self.TIamb * np.ones_like(x)
    


    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return np.zeros_like(x)
    

class AnalyticalSuperimposed(Superimposed):
    def RETI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        TI_base = self.base_windfield.TI(x, y, z)

        max_WATI = np.zeros_like(TI_base)
        for wake in self.wakes:
            max_WATI = np.maximum(wake.REWATI(x, y, z), max_WATI)

        TI_out = np.sqrt(TI_base**2 + max_WATI**2)

        return TI_out

    def line_wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns analytically line averaged rotor-equivalent wind speed at 
            specified coordinates using linear Niayifar superposition.

        Parameters:
        - x: x-coordinate.
        - y: y-coordinate.
        - z: z-coordinates.

        Returns:
        ArrayLike: Wind speed averaged along a lateral line of length 1, centered at (x, y).
        """
        base = self.base_windfield.wsp(x, y, z)
        deficits = []
        for wake in self.wakes:
            if self.method == "niayifar":
                deficits.append(wake.niayifar_line_deficit(x, y))
            else:
                deficits.append(wake.line_deficit(x, y))
                
        out = base - np.sum(deficits, axis=0)

        return out
