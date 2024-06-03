#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:42:51 2024

@author: mfurquan
"""
from circle_in_square_mesh import circ_in_square
from rectangle_mesh import rect
from math import tan, pi
import numpy as np
from case import N, B, Lxu, Lxd, Ly, n, Nxu, Nxd, Ny

# N = 5
# B = 5. # inner box: 2B x 2B
# Lxu = 4. # *B
# Lxd = 9. # *B
# Ly = 4. # *B
# n = 2. # poly mesh stretching
Nr = 3*(N-1)
Ntheta = 8*N
# Nxu = 5
# Nxd = 11
# Ny = N

def uniform(x):
    return x

def poly(x):
    return ((x[0]-x[-1])/(x[0]**n - x[-1]**n))*x**n + (x[0]*x[-1]**n - x[0]**n*x[-1])/(x[-1]**n - x[0]**n)
    #return (x[-1]-x[0])*(x/x[-1])**n + x[0]

def polyr(x):
    def root(u):
        return u*abs(u)**(1/n-1)
    return ((x[0]-x[-1])/(root(x[0]) - root(x[-1])))*root(x) + (x[0]*root(x[-1]) - root(x[0])*x[-1])/(root(x[-1]) - root(x[0]))
    #return (x[0]-x[-1])*(x/x[0])**n + x[-1]

def tan_scale(x):
    return B*tan(pi*x/(4.*B))

m = circ_in_square(1., B, Nr, Ntheta)
m1 = rect(B,-B,Lxd*B, 2.*B, Nxd, Ntheta//4 + 1, poly, np.vectorize(tan_scale))
m.join_mesh(m1,0,2)
m1 = rect(-(Lxu+1.)*B,-B,Lxu*B,2.*B,Nxu,Ntheta//4+1,poly, np.vectorize(tan_scale))
m.join_mesh(m1,1,0)
m.combine(0,6)
m.combine(0,4)
m.combine(1,6)
m.combine(1,4)

m1 = rect(-(Lxu+1.)*B,B,Lxu*B,Ly*B,Nxu,Ny,poly,poly)
m2 = rect(-B,B,2.*B,Ly*B,Ntheta//4+1,Ny,np.vectorize(tan_scale),poly)
m1.join_mesh(m2,0,2)
m2 = rect(B,B,Lxd*B, Ly*B, Nxd, Ny, poly, poly)
m1.join_mesh(m2,3,2)
m1.combine(2,4)
m1.combine(2,6)
m.join_mesh(m1,0,2)

m1 = rect(-(Lxu+1.)*B,-(Ly+1.)*B,Lxu*B,Ly*B,Nxu,Ny,poly,poly)
m2 = rect(-B,-(Ly+1.)*B,2.*B,Ly*B,Ntheta//4+1,Ny,np.vectorize(tan_scale),poly)
m1.join_mesh(m2,0,2)
m2 = rect(B,-(Ly+1.)*B,Lxd*B, Ly*B, Nxd, Ny, poly, poly)
m1.join_mesh(m2,3,2)
m1.combine(0,3)
m1.combine(0,5)
m.join_mesh(m1,0,0)

m.combine(1,6)
m.combine(1,10)
m.combine(2,4)
m.combine(2,6)
m.combine(3,4)
m.combine(3,4)
m.combine(4,5)
m.combine(4,5)

m.plot(0)
m#1.plot(0)
#m2.plot(2)
