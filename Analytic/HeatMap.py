import math
import numpy as np
class HeatMap():
    def __init__(self):
        self.c= {'c1':0, 'c2':0, 'c3':0, 'c4':0}
    def theta(self, x, y):
        ret = math.acos(x/math.sqrt(max(x**2+y**2, 1e-9)))
        return ret

    def rad(self, x, y):
        return(max(math.sqrt(x**2+y**2), 1e-9))

    def ConformalMapping(self, u, v):
        denominator = u ** 2 + (1 - v) ** 2
        x = (2 * u) / denominator
        y = (1 - u ** 2 - v ** 2) / denominator
        return x, y

    def UpperTempH(self, x, y):
        pi = math.pi
        c1 = self.c['c1']
        c2 = self.c['c2']
        c3 = self.c['c3']
        c4 = self.c['c4']

        ret = (c1 + ((c4-c1)/pi)*self.theta(x-1, y)
               +((c3-c4)/pi)*self.theta(x, y)
               + ((c2-c3)/pi)*self.theta(x+1, y))

        return ret
    def UpperH(self, x, y):
        return np.vectorize(self.UpperTempH)(x, y)

    def DiskH(self, u, v):
        x, y = self.ConformalMapping(u, v)
        ret = self.UpperH(x, y)
        ret[u**2 + v**2 > 1] = np.nan
        return ret

    def UpperTempHConj(self, x, y):
        pi = math.pi
        c1 = self.c['c1']
        c2 = self.c['c2']
        c3 = self.c['c3']
        c4 = self.c['c4']
        ret = (((c4 - c1) / pi) * math.log(self.rad(x - 1, y))
               + ((c3 - c4) / pi) * math.log(self.rad(x, y))
               + ((c2 - c3) / pi) * math.log(self.rad(x + 1, y)))
        return ret

    def UpperHConj(self, x, y):
        return np.vectorize(self.UpperTempHConj)(x, y)

    def DiskHConj(self, u, v):
        x, y = self.ConformalMapping(u, v)
        ret = self.UpperHConj(x, y)
        ret[u**2 + v**2 > 1] = np.nan
        return ret