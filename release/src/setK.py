import numpy as np
def setK(x,y,f):
    K=np.eye(3)
    K[0,0]=f*x
    K[1,1]=f*x
    K[0,2]=x/2
    K[1,2]=y/2
    return K