#!/usr/bin/python

from Pislib import *

d=scalarpotential()

print("Sigma in spherical coordinates is %s"%d.Sigma_spherical)
print("Sigma in cartesian coordinates is %s"%d.Sigma)

print("Pix is %s"%d.Pix)
print("Piy is %s"%d.Piy)
print("Piz is %s"%d.Piz)

xp=1.0
yp=0.0
zp=1.0
        
print("Pix(%2.1f,%2.1f,%2.1f) = %+2.1f"%(xp,yp,zp,d.fPix(xp,yp,zp)))
print("Piy(%2.1f,%2.1f,%2.1f) = %+2.1f"%(xp,yp,zp,d.fPiy(xp,yp,zp)))
print("Piz(%2.1f,%2.1f,%2.1f) = %+2.1f"%(xp,yp,zp,d.fPiz(xp,yp,zp)))

print

# Example picking ell and m

f=scalarpotential(1,-2)

print("ell is %d, and m is %d"%(f.ell,f.m))

print("Sigma in spherical coordinates is %s"%f.Sigma_spherical)
print("Sigma in cartesian coordinates is %s"%f.Sigma)

print("Pix is %s"%f.Pix)
print("Piy is %s"%f.Piy)
print("Piz is %s"%f.Piz)
