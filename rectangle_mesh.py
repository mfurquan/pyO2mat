#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:56:37 2024

@author: mfurquan
"""
import numpy as np
from meshes import mesh

def rect(x0,y0,Lx,Ly,Nx,Ny,fx,fy):
    Nsd = 2
    Nen = 4
    Nben = 2
    Nface = 4

    Nn = Nx*Ny
    Ne = (Nx-1)*(Ny-1)
    Nb1 = Ny-1
    Nb2 = Nx-1
    Nb3 = Ny-1
    Nb4 = Nx-1

    x_mesh = fx(np.linspace(x0,x0+Lx,Nx))
    y_mesh = fy(np.linspace(y0,y0+Ly,Ny))

    xy_mesh = np.empty((Nn,Nsd))
    ij2k_map = np.empty((Nx,Ny),dtype=int)
    ibn1 = np.empty([Nb1,Nben],dtype=int)
    ibn2 = np.empty([Nb2,Nben],dtype=int)
    ibn3 = np.empty([Nb3,Nben],dtype=int)
    ibn4 = np.empty([Nb4,Nben],dtype=int)
    k = 0
    for i in range(Nx):
        for j in range(Ny):
            xy_mesh[k,:] = np.array([x_mesh[i],y_mesh[j]])
            ij2k_map[i,j] = k
            k = k + 1
        
    ien = np.empty((Ne,Nen),dtype=int)
    k = 0
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    for i in range(Nx-1):
        for j in range(Ny-1):
            ien[k,:] = np.array([ij2k_map[i,j],ij2k_map[i+1,j],ij2k_map[i+1,j+1],ij2k_map[i,j+1]])
            if i==Nx-2:
                ibn1[i1,0] = k
                i1 += 1
            if j==Ny-2:
                ibn2[i2,0] = k
                i2 += 1
            if i==0:
                ibn3[i3,0] = k
                i3 += 1
            if j==0:
                ibn4[i4,0] = k
                i4 += 1
            k = k + 1
                
            ibn1[:,1] = 1
            ibn2[:,1] = 2
            ibn3[:,1] = 3
            ibn4[:,1] = 0
    return mesh(Nsd,Nen,Nben,Nface,Nn,Ne,[Nb1,Nb2,Nb3,Nb4],[ibn1,ibn2,ibn3,ibn4],xy_mesh,ien)