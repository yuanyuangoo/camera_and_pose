import numpy as np
class camera:
    def __init__(self):
        self.data = []
    def prt(self):
        print(self)
        print(self.__class__)
    def process_options(self,Raff,taff,R,t,S):
        self.Raff = Raff
        self.taff = taff.conj().T
        self.R = R
        self.t = t
        self.S = S
        self.M = np.dot(S, Raff)