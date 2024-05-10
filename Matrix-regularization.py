#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:11:09 2023

@author: tara
"""
#this a simple example to show regularization method
import numpy as np
from numpy.linalg import inv

Ix = 0
Iy = 0
Iz = 0
l = .5
n = 3
I = np.array([[Ix], [Iy], [Iz]])
M = np.array([[1, 1, 1]])
B = 1
MT = M.transpose()
R1 = M.dot(B)     
T = l**2 * np.eye(n)
D = MT.dot(M) + T
Dinv = inv(D)

print(D)
print(Dinv)

X = Dinv.dot(MT.dot(B))

Ix_result = X[0][0]
Iy_result = X[1][0]
Iz_result = X[2][0]

print("Ix =", Ix_result)
print("Iy =", Iy_result)
print("Iz =", Iz_result)
