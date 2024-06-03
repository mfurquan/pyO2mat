#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:27:11 2024

@author: mfurquan
"""
import numpy as np
from cylinder_mesh import m
import utility as u
from operators import slip_wall, navier_stokes as flow_eqn
from math import atan2
from meshes import edge2nod
import plot as pl
from force import calc_force
from case import Re
import meshio
        
K = np.zeros((m.Nn*u.Ndf,m.Nn*u.Ndf))
F = np.zeros(m.Nn*u.Ndf)

Nitr = 10

def cyl_pot(x):
    u = 1. + 1./(x[0]**2+x[1]**2) - 2.*x[0]**2/(x[0]**2+x[1]**2)**2
    v = -2.*x[0]*x[1]/(x[0]**2+x[1]**2)**2
    p = 0.5*(1.- u**2 - v**2)
    return np.array([u,v,p])

#d = np.array([[1.,0.,0.] for i in range(m.Nn)])
d = np.apply_along_axis(cyl_pot,1,m.x)

u.set_bc(0., m.ibn[0], m.Nb[0],m.ien,0,d)
u.set_bc(0., m.ibn[0], m.Nb[0],m.ien,1,d)
u.set_bc(1., m.ibn[2], m.Nb[2],m.ien,0,d)
u.set_bc(0., m.ibn[2], m.Nb[2],m.ien,1,d)
u.set_bc(0., m.ibn[3], m.Nb[3],m.ien,1,d)
u.set_bc(0., m.ibn[4], m.Nb[4],m.ien,1,d)

for itr in range(Nitr):
    K[:,:] = 0.
    F[:] = 0.
    u.assemble(K,F,flow_eqn,d,m)

    u.assemble_bndry(K,F,slip_wall,m.ibn[3],m.Nb[3],1,d,m)
    u.assemble_bndry(K,F,slip_wall,m.ibn[4],m.Nb[4],1,d,m)
    u.set_dirichlet(K,F, m.ibn[0], m.Nb[0],m.ien,0)
    u.set_dirichlet(K,F, m.ibn[0], m.Nb[0],m.ien,1)
    u.set_dirichlet(K,F, m.ibn[2], m.Nb[2],m.ien,0)
    u.set_dirichlet(K,F, m.ibn[2], m.Nb[2],m.ien,1)
    u.set_dirichlet(K,F, m.ibn[3], m.Nb[3],m.ien,1)
    u.set_dirichlet(K,F, m.ibn[4], m.Nb[4],m.ien,1)

    res = np.linalg.norm(F)/(m.Nn*u.Ndf)
    print('res = ',res)
    if res<1.e-6 or res>0.1:
        break
    dd = np.linalg.solve(K,F)
    d += dd.reshape([m.Nn,u.Ndf])

print('C_F = ',calc_force(m.ibn[0],m.Nb[0],d))

def theta(x):
    return atan2(x[1],x[0])   

n_th = edge2nod(m.ibn[0],m.ien,m.x,theta)
th = np.vectorize(lambda n: theta(m.x[n,:]))(n_th)/np.pi
p = d[n_th,m.Nsd]

pl.plot(m,d)
pl.plt.plot(th,p)

mesh = meshio.Mesh(m.x,[("quad",m.ien)],point_data={"p":d[:,m.Nsd]})
mesh.write("cyl_Re"+str(Re)+".vtu")