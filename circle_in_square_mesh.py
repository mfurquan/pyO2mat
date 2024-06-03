#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:23:34 2024

@author: mfurquan
"""
import numpy as np
from math import pi, sin, cos
from meshes import mesh

def circ_in_square(D,S,Nr,Ntheta):
    Nsd = 2
    Nen = 4
    Nben = 2
    Nface = 4

    Nn = Nr*Ntheta
    Ne = (Nr-1)*Ntheta
    Nb1 = Ntheta//4
    Nb2 = Ntheta//4
    Nb3 = Ntheta//4
    Nb4 = Ntheta//4
    Nb5 = Ntheta


    def r_lim(theta):
        if theta <= pi/4 or abs(theta-pi) <= pi/4 or theta > 7.*pi/4:
            return S/abs(cos(theta))
        else:
            return S/abs(sin(theta))
    
    def r_inflation(r):
        return (r[-1]-r[0])*((r-r[0])/(r[-1]-r[0]))**2 + r[0]

    theta_mesh = np.linspace(0., 2.*np.pi,Ntheta,False)
    r_mesh = np.empty((Ntheta,Nr))
    for i in range(Ntheta):
        r_mesh[i,:] = r_inflation(np.linspace(1.0, r_lim(theta_mesh[i]),Nr))
    
    xy_mesh = np.empty((Nn,Nsd))
    ij2k_map = np.empty((Nr,Ntheta),dtype=int)
    ibn1 = np.empty([Nb1,Nben],dtype=int)
    ibn2 = np.empty([Nb2,Nben],dtype=int)
    ibn3 = np.empty([Nb3,Nben],dtype=int)
    ibn4 = np.empty([Nb4,Nben],dtype=int)
    ibn5 = np.empty([Nb5,Nben],dtype=int)
    k = 0
    for i in range(Nr):
        for j in range(Ntheta):
            xy_mesh[k,:] = np.array([r_mesh[j,i]*np.cos(theta_mesh[j]),r_mesh[j,i]*np.sin(theta_mesh[j])])
            ij2k_map[i,j] = k
            k = k + 1
            
    ien = np.empty((Ne,Nen),dtype=int)
    k = 0
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0
    for i in range(Nr-1):
        for j in range(Ntheta):
            jp = j + 1
            if jp == Ntheta:
                jp = 0
            ien[k,:] = np.array([ij2k_map[i,j],ij2k_map[i+1,j],ij2k_map[i+1,jp],ij2k_map[i,jp]])
            if i==Nr-2:
                if j<Nb1//2:
                    ibn1[i1,0] = k
                    i1 += 1
                elif j<Nb1//2+Nb2:
                    ibn2[i2,0] = k
                    i2 += 1
                elif j<Nb1//2+Nb2+Nb3:
                    ibn3[i3,0] = k
                    i3 += 1
                elif j<Nb1//2+Nb2+Nb3+Nb4:
                    ibn4[i4,0] = k
                    i4 += 1
                else:
                    ibn1[i1,0] = k
                    i1 += 1
            elif i==0:
                    ibn5[i5,0] = k
                    i5 += 1
            k = k + 1
                    
    ibn1[:,1] = 1
    ibn2[:,1] = 1
    ibn3[:,1] = 1
    ibn4[:,1] = 1
    ibn5[:,1] = 3
    return mesh(Nsd,Nen,Nben,Nface,Nn,Ne,[Nb1,Nb2,Nb3,Nb4,Nb5],[ibn1,ibn2,ibn3,ibn4,ibn5],xy_mesh,ien)