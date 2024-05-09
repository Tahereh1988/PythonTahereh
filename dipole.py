
from scipy.constants import mu_0,pi
import numpy as np
import random
from math import *

class dipole():
    def __init__(self):
        self.xd=0
        self.yd=0
        self.zd=0
        self.mx=0
        self.my=0
        self.mz=0
    def set(self,xd,yd,zd,mx,my,mz):
        self.xd=xd
        self.yd=yd
        self.zd=zd
        self.mx=mx
        self.my=my
        self.mz=mz
    def b(self,x,y,z):
        xprime=x-self.xd
        yprime=y-self.yd
        zprime=z-self.zd
        rprime=np.sqrt(xprime**2+yprime**2+zprime**2)
        mdotrprime=self.mx*xprime+self.my*yprime+self.mz*zprime
        bx=mu_0/(4*pi)*(3*xprime*mdotrprime/rprime**5-self.mx/rprime**3)
        by=mu_0/(4*pi)*(3*yprime*mdotrprime/rprime**5-self.my/rprime**3)
        bz=mu_0/(4*pi)*(3*zprime*mdotrprime/rprime**5-self.mz/rprime**3)
        return bx,by,bz
    def bx(self,x,y,z):
        xprime=x-self.xd
        yprime=y-self.yd
        zprime=z-self.zd
        rprime=np.sqrt(xprime**2+yprime**2+zprime**2)
        mdotrprime=self.mx*xprime+self.my*yprime+self.mz*zprime
        bx=mu_0/(4*pi)*(3*xprime*mdotrprime/rprime**5-self.mx/rprime**3)
        return bx
    def by(self,x,y,z):
        xprime=x-self.xd
        yprime=y-self.yd
        zprime=z-self.zd
        rprime=np.sqrt(xprime**2+yprime**2+zprime**2)
        mdotrprime=self.mx*xprime+self.my*yprime+self.mz*zprime
        by=mu_0/(4*pi)*(3*yprime*mdotrprime/rprime**5-self.my/rprime**3)
        return by
    def bz(self,x,y,z):
        xprime=x-self.xd
        yprime=y-self.yd
        zprime=z-self.zd
        rprime=np.sqrt(xprime**2+yprime**2+zprime**2)
        mdotrprime=self.mx*xprime+self.my*yprime+self.mz*zprime
        bz=mu_0/(4*pi)*(3*zprime*mdotrprime/rprime**5-self.mz/rprime**3)
        return bz
    def random(self,a): # random position on wall half-length a
        side=random.randint(0,5)
        #print(side)
        u=(random.random()-0.5)*2*a
        v=(random.random()-0.5)*2*a
        if(side==0): # floor
            self.zd=-a
            self.xd=u
            self.yd=v
        elif(side==1): # ceiling
            self.zd=a
            self.xd=u
            self.yd=v
        elif(side==2): # left side
            self.zd=u
            self.xd=v
            self.yd=-a
        elif(side==3): # right side
            self.zd=u
            self.xd=v
            self.yd=a
        elif(side==4): # front side
            self.zd=u
            self.xd=a
            self.yd=v
        else: # back side
            self.zd=u
            self.xd=-a
            self.yd=v
        #print(self.xd,self.yd,self.zd)
        costheta=(random.random()-0.5)*2
        theta=acos(costheta)
        phi=random.random()*2*pi
        self.mx=sin(theta)*cos(phi)
        self.my=sin(theta)*sin(phi)
        self.mz=cos(theta)
        #print(self.mx,self.my,self.mz,sqrt(self.mx**2+self.my**2+self.mz**2))
