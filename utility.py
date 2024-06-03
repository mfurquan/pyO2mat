#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:30:16 2024

@author: mfurquan
"""
import numpy as np
from meshes import facemap, Nface
import quadrature as q
import shape_fns as s

Ndf = 3

N    = np.apply_along_axis(s.shape_fn,1,q.xg2d)
N_xi = np.apply_along_axis(s.deriv_shape_fn,1,q.xg2d)

B = np.zeros((Nface,q.Nint,s.Nen))
B_xi = np.zeros((Nface,q.Nint,s.Nsd,s.Nen))

for i in range(Nface):
    B[i,:,:] = np.apply_along_axis(s.shape_fn,1,q.xgb[i,:,:])
    B_xi[i,:,:,:] = np.apply_along_axis(s.deriv_shape_fn,1,q.xgb[i,:,:])

def lm(inod,idf):
    return inod*Ndf+idf

def integ_domain(operator,xe,de):
    kq = np.zeros((s.Nen,s.Nen,Ndf,Ndf))
    fq = np.zeros((s.Nen,Ndf))
    for iq in range(q.Nint2d):
        x_xi = np.matmul(N_xi[iq,:,:],xe)
        xi_x = np.linalg.inv(x_xi)
        jac  = np.linalg.det(x_xi)
        grad_N = np.matmul(xi_x,N_xi[iq,:,:])
        kf = operator(N[iq],grad_N,xi_x,jac,de)
        kq += q.wg2d[iq]*kf[0]
        fq += q.wg2d[iq]*kf[1]
    return (kq,fq)       
     
def assemble(K,F,opr,d,m):
    for ie in range(m.Ne):
        xe = m.x[m.ien[ie,:],:]
        de = d[m.ien[ie,:],:]
        kfe = integ_domain(opr,xe,de)
        for i in range(m.Nen):
            for j in range(m.Nen):
                for idf in range(Ndf):
                    for jdf in range(Ndf):
                        K[lm(m.ien[ie,i],idf),lm(m.ien[ie,j],jdf)] += kfe[0][i,j,idf,jdf]
            for idf in range(Ndf):
                F[lm(m.ien[ie,i],idf)] += kfe[1][i,idf]
                        
def set_dirichlet(K,F,ibn,Nbn,ien,idf,g=0.):
    for i in range(Nbn):
        k = lm(ien[ibn[i,0],facemap[ibn[i,1],0]],idf)
        K[k,:] = 0.
        K[k,k] = 1.
        F[k] = g
        k = lm(ien[ibn[i,0],facemap[ibn[i,1],1]],idf)
        K[k,:] = 0.
        K[k,k] = 1.
        F[k] = g

def set_bc(g,ibn,Nbn,ien,idf,d):
    for i in range(Nbn):
        d[ien[ibn[i,0],facemap[ibn[i,1],0]],idf] = g
        d[ien[ibn[i,0],facemap[ibn[i,1],1]],idf] = g
   
def integ_bndry(operator,xe,face,normal,de):
    kq = np.zeros((s.Nen,s.Nen,2))
    fq = np.zeros(s.Nen)
    for iq in range(q.Nint):
        x_xi = np.matmul(B_xi[face,iq,:,:],xe)
        xi_x = np.linalg.inv(x_xi)
        grad_B = np.matmul(xi_x,B_xi[face,iq,:,:])
        kf = operator(B[face,iq,:],grad_B,x_xi[1-normal,1-normal],normal,de)
        kq += q.wg[iq]*kf[0]
        fq += q.wg[iq]*kf[1]
    return (kq,fq)
        
def assemble_bndry(K,F,opr,ibn,Nb,normal,d,m):
    for ie in range(Nb):
        xe = m.x[m.ien[ibn[ie,0],:],:]
        de = d[m.ien[ie,:],:]
        kfe = integ_bndry(opr,xe,ibn[ie,1],normal,de)
        for i in range(m.Nen):
            for j in range(m.Nen):
                 K[lm(m.ien[ibn[ie,0],i],normal),lm(m.ien[ibn[ie,0],j],m.Nsd)] += kfe[0][i,j,0]
                 K[lm(m.ien[ibn[ie,0],i],normal),lm(m.ien[ibn[ie,0],j],normal)] += kfe[0][i,j,1]
            F[lm(m.ien[ibn[ie,0],i],normal)] += kfe[1][i]
    

