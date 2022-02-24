import numpy as np
import math
from pyspark.sql import Row


"""
Implementation of Lorentz vector
"""
class LorentzVector(object):
    def __init__(self, *args):
        if len(args)>0:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
            self.t = args[3]
    
    def SetPtEtaPhiM(self, pt, eta, phi, mass):
        pt = abs(pt)
        self.SetXYZM(pt*math.cos(phi), pt*math.sin(phi), pt*math.sinh(eta), mass)
        
    def SetXYZM(self, x, y, z, m):
        self.x = x;
        self.y = y
        self.z = z
        if (m>=0):
            self.t = math.sqrt(x*x + y*y + z*z + m*m)
        else:
            self.t = math.sqrt(max(x*x + y*y + z*z - m*m, 0))
            
    def E(self):
        return self.t
    
    def Px(self): 
        return self.x
    
    def Py(self):
        return self.y
    
    def Pz(self):
        return self.z
    
    def Pt(self):
        return math.sqrt(self.x*self.x + self.y*self.y)
    
    def Eta(self):
        cosTheta = self.CosTheta()
        if cosTheta*cosTheta<1:
            return -0.5*math.log((1.0 - cosTheta)/(1.0 + cosTheta))
        if self.z == 0: return 0
    
    def mag(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def CosTheta(self):
        return 1.0 if self.mag()==0.0 else self.z/self.mag()
    
    def Phi(self):
        return math.atan2(self.y, self.x)
    
    def DeltaR(self, other):
        deta = self.Eta() - other.Eta()
        
        dphi = self.Phi() - other.Phi()        
        pi = math.pi
        while dphi >  pi: dphi -= 2*pi
        while dphi < -pi: dphi += 2*pi

        return math.sqrt(deta*deta + dphi*dphi)
    
    
    
"""
Functions used to return the Pt map of selected tracks, neutrals and photons
"""
def ChPtMapp(DR, event):
    pTmap = []
    for h in event.EFlowTrack:
        if h.PT<= 0.5: continue
        pTmap.append([h.Eta, h.Phi, h.PT])
    return np.asarray(pTmap)

def NeuPtMapp(DR, event):
    pTmap = []
    for h in event.EFlowNeutralHadron:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
    return np.asarray(pTmap)

def PhotonPtMapp(DR, event):
    pTmap = []
    for h in event.EFlowPhoton:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
    return np.asarray(pTmap)
"""
Functions used to return the Pt map of selected tracks, neutrals and photons
Versions used for the optimized filtering with Spark SQL and HOF
"""
# get the selected tracks
def ChPtMapp2(Tracks):
    #pTmap = []
    pTmap = np.zeros((len(Tracks), 3))
    for i, h in enumerate(Tracks):
        pTmap[i] = [h["Eta"], h["Phi"], h["PT"]]
    return pTmap

# get the selected neutrals
def NeuPtMapp2(NeutralHadrons):
    pTmap = np.zeros((len(NeutralHadrons), 3))
    for i, h in enumerate(NeutralHadrons):
        pTmap[i] = [h["Eta"], h["Phi"], h["ET"]]
    return pTmap

# get the selected photons
def PhotonPtMapp2(Photons):
    pTmap = np.zeros((len(Photons), 3))
    for i, h in enumerate(Photons):
        pTmap[i] = [h["Eta"], h["Phi"], h["ET"]]
    return pTmap
"""
Get the particle ISO
"""
def PFIso(p, DR, PtMap, subtractPt):
    if p.Pt() <= 0.: return 0.
    DeltaEta = PtMap[:,0] - p.Eta()
    DeltaPhi = PtMap[:,1] - p.Phi()
    twopi = 2.* math.pi
    DeltaPhi = DeltaPhi - twopi*(DeltaPhi >  twopi) + twopi*(DeltaPhi < -1.*twopi)
    isInCone = DeltaPhi*DeltaPhi + DeltaEta*DeltaEta < DR*DR
    Iso = PtMap[isInCone, 2].sum()/p.Pt()
    if subtractPt: Iso = Iso -1
    return float(Iso)
