import matcompat
class pose:
    def __init__(self):
        self.data = []
    def prt(self):
        print(self)
        print(self.__class__)
    def process_options(self,skel,BOMP,mu,lambda2,lambda1,K,numIter,numIters2,tol1,tol2,ks,optType,viz,annoids,numPoints):
        self.skel=skel
        self.BOMP=BOMP
        self.mu=mu
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.K=K
        self.numIter=numIter
        self.numIters2=numIters2
        self.tol1=tol1
        self.tol2=tol2
        self.ks=ks
        self.optType=optType 
        self.viz=viz
        self.annoids=annoids
        self.numPoints=numPoints

class skel:
    def __init__(self):
        pose.skel.length=''
        pose.skel.mass=''
        pose.skel.angle=''
        pose.skel.type=''
        pose.skel.documentation=''
        pose.skel.name=''
        pose.skel.tree=''
    def prt(self):
        print(self)
        print(self.__class__)
    
