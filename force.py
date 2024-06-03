#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:21:38 2024

@author: mfurquan
"""
import numpy as np
from cylinder_mesh import m
import quadrature as q
import utility as u
from case import Re

def normal(x_xi,face):
    match face:
        case 0:
            return np.array([x_xi[0,1],-x_xi[0,0]])
        case 1:
            return np.array([x_xi[1,1],-x_xi[1,0]])
        case 2:
            return np.array([-x_xi[0,1],x_xi[0,0]])
        case 3:
            return np.array([-x_xi[1,1],x_xi[1,0]])

def calc_force(ibn,Nb,d):
    F = np.zeros(m.Nsd)
    for ib in range(Nb):
        xe = m.x[m.ien[ibn[ib,0],:],:]
        pe = d[m.ien[ibn[ib,0],:],m.Nsd]
        ve = d[m.ien[ibn[ib,0],:],:m.Nsd]
        face = ibn[ib,1]
        for iq in range(q.Nint):
            p = np.dot(u.B[ibn[ib,1],iq,:],pe)
            x_xi = np.matmul(u.B_xi[face,iq,:,:],xe)
            xi_x = np.linalg.inv(x_xi)
            grad_B = np.matmul(xi_x,u.B_xi[face,iq,:,:])
            grad_v = np.matmul(grad_B,ve)
            n = normal(x_xi,face)
            F -= q.wg[iq]*(-p*n + np.matmul(grad_v+np.transpose(grad_v),n)/Re)
    return 2*F # Force coefficients