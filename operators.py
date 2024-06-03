#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:50:23 2024

@author: mfurquan
"""
import numpy as np
from case import Re, C

# Re = 30.2
# C = 60.0

def stokes(N,grad_N,xi_x,jac,de=None):
    Nen = np.size(N)
    Nsd = np.size(grad_N,0)
    Ndf = Nsd + 1
    
    G = np.matmul(xi_x,np.transpose(xi_x))
    tau_SUPS = Re/np.sqrt(C*np.tensordot(G,G,axes=2))
    #he = 4.*np.sqrt(jac/np.pi)
    #tau_SUPS = (he*he/2.)*Re
    nu_LSIC  = 1./(tau_SUPS*np.trace(G))
    kq = np.zeros((Nen,Nen,Ndf,Ndf))
    for i in range(Nen):
        for j in range(Nen):
            gNtgN = np.tensordot(grad_N[:,i],grad_N[:,j],axes=1)
            gNgNt = np.tensordot(grad_N[:,j],grad_N[:,i],axes=0)
            gNgN  = np.tensordot(grad_N[:,i],grad_N[:,j],axes=0)
            kq[i,j,:Nsd,:Nsd] = jac*((gNtgN*np.eye(Nsd,Nsd) + gNgNt)/Re + nu_LSIC*gNgN)
            kq[i,j,:Nsd,Nsd]  =-jac*grad_N[:,i]*N[j]
            kq[i,j,Nsd,:Nsd]  = jac*N[i]*grad_N[:,j]
            kq[i,j,Nsd,Nsd]   = jac*tau_SUPS*gNtgN
    return kq

def oseen(N,grad_N,xi_x,jac,de):
    Nen = np.size(N)
    Nsd = np.size(grad_N,0)
    Ndf = Nsd + 1
    
    U = np.array([1.,0.])
    p = np.matmul(N,de[:,Nsd])
    G = np.matmul(xi_x,np.transpose(xi_x))
    grad_u = np.matmul(grad_N,de[:,:Nsd])
    div_u  = np.trace(grad_u)
    grad_p = np.matmul(grad_N,de[:,Nsd])
    Ugu    = np.matmul(U,grad_u)
    tau_SUPS = 1./np.sqrt(np.dot(U,np.matmul(G,U)) + C*np.tensordot(G,G,axes=2)/Re**2)
    #he = 4.*np.sqrt(jac/np.pi)
    #tau_SUPS = (he*he/2.)*Re
    nu_LSIC  = 1./(tau_SUPS*np.trace(G))
    kq = np.zeros((Nen,Nen,Ndf,Ndf))
    fq = np.zeros((Nen,Ndf))
    for i in range(Nen):
        for j in range(Nen):
            gNtgN = np.tensordot(grad_N[:,i],grad_N[:,j],axes=1)
            gNgNt = np.tensordot(grad_N[:,j],np.transpose(grad_N[:,i]),axes=0)
            gNgN  = np.tensordot(grad_N[:,i],np.transpose(grad_N[:,j]),axes=0)
            kq[i,j,:Nsd,:Nsd] = jac*(N[i]*np.dot(U,grad_N[:,j])*np.eye(Nsd,Nsd)
                                     + (gNtgN*np.eye(Nsd,Nsd) + gNgNt)/Re
                                     + nu_LSIC*gNgN
                                     + tau_SUPS*np.dot(U,grad_N[:,i])*np.dot(U,grad_N[:,j])*np.eye(Nsd,Nsd))
            kq[i,j,:Nsd,Nsd]  = jac*(-grad_N[:,i]*N[j] + tau_SUPS*np.dot(U,grad_N[:,i])*grad_N[:,j])
            kq[i,j,Nsd,:Nsd]  = jac*(N[i]*grad_N[:,j] + tau_SUPS*grad_N[:,i]*np.dot(U,grad_N[:,j]))
            kq[i,j,Nsd,Nsd]   = jac*tau_SUPS*gNtgN
        fq[i,:Nsd] = jac*(-N[i]*Ugu + p*grad_N[:,i] - np.matmul(grad_u+np.transpose(grad_u),grad_N[:,i])/Re
                          - nu_LSIC*div_u*grad_N[:,i] - tau_SUPS*(np.dot(U,grad_N[:,i])*Ugu
                                                                  + np.dot(U,grad_N[:,i])*grad_p))
        fq[i,Nsd]  = jac*(-N[i]*div_u - tau_SUPS*(np.dot(grad_N[:,i],Ugu) + np.dot(grad_N[:,i],grad_p)))
    return (kq,fq)

def navier_stokes(N,grad_N,xi_x,jac,de):
    Nen = np.size(N)
    Nsd = np.size(grad_N,0)
    Ndf = Nsd + 1
    
    u = np.matmul(N,de[:,:Nsd])
    p = np.matmul(N,de[:,Nsd])
    G = np.matmul(xi_x,np.transpose(xi_x))
    grad_u = np.matmul(grad_N,de[:,:Nsd])
    div_u  = np.trace(grad_u)
    grad_p = np.matmul(grad_N,de[:,Nsd])
    ugu    = np.matmul(u,grad_u)
    #tau_SUPS = 1./np.sqrt(np.dot(u,np.matmul(G,u)) + C*np.tensordot(G,G,axes=2)/Re**2)
    he = 4.*np.sqrt(jac/np.pi)
    tau_SUPS = (he*he/2.)*Re
    nu_LSIC = 1./(tau_SUPS*np.trace(G))
    kq = np.zeros((Nen,Nen,Ndf,Ndf))
    fq = np.zeros((Nen,Ndf))
    for i in range(Nen):
        for j in range(Nen):
            kq[i,j,:Nsd,:Nsd] = jac*((N[i]*np.dot(u,grad_N[:,j])
                                      + np.dot(grad_N[:,i],grad_N[:,j])/Re)*np.eye(Nsd,Nsd)
                                     + N[i]*N[j]*np.transpose(grad_u)
                                     + np.tensordot(grad_N[:,j],grad_N[:,i],axes=0)/Re
                                     + nu_LSIC*np.tensordot(grad_N[:,i],grad_N[:,j],axes=0)
                                     + tau_SUPS*(np.tensordot(ugu,grad_N[:,i]*N[j],axes=0)
                                                 + np.dot(u,grad_N[:,i])*N[j]*np.transpose(grad_u)
                                                 + np.dot(u,grad_N[:,i])*np.dot(u,grad_N[:,j])*np.eye(Nsd,Nsd)
                                                 + np.tensordot(grad_p,grad_N[:,i],axes=0)*N[j]))
            kq[i,j,:Nsd,Nsd]  = jac*(-grad_N[:,i]*N[j]
                                     + tau_SUPS*np.dot(u,grad_N[:,i])*grad_N[:,j])
            kq[i,j,Nsd,:Nsd]  = jac*(N[i]*grad_N[:,j]
                                     + tau_SUPS*(np.matmul(grad_N[:,i],np.transpose(grad_u))*N[j]
                                                + grad_N[:,i]*np.dot(u,grad_N[:,j])))
            kq[i,j,Nsd,Nsd]   = jac*tau_SUPS*np.dot(grad_N[:,i],grad_N[:,j])            
        fq[i,:Nsd] = jac*(-N[i]*ugu + p*grad_N[:,i] - np.matmul(grad_u+np.transpose(grad_u),grad_N[:,i])/Re
                          - nu_LSIC*div_u*grad_N[:,i] - tau_SUPS*(np.dot(u,grad_N[:,i])*ugu
                                                                  + np.dot(u,grad_N[:,i])*grad_p))
        fq[i,Nsd]  = jac*(-N[i]*div_u - tau_SUPS*(np.dot(grad_N[:,i],ugu) + np.dot(grad_N[:,i],grad_p)))
    return (kq,fq)
            
def slip_wall(N,grad_N,x_xi,normal,de):
    Nen = np.size(N)
    Nsd = np.size(grad_N,0)
    
    du_dn = np.dot(grad_N[normal,:],de[:,normal])
    p = np.matmul(N,de[:,Nsd])
    kq = np.zeros((Nen,Nen,2))
    fq = np.zeros(Nen)
    for i in range(Nen):
        for j in range(Nen):
            kq[i,j,0] = x_xi*N[i]*N[j]
            kq[i,j,1] =-x_xi*N[i]*grad_N[normal,j]/Re
        fq[i] = N[i]*(-p + du_dn/Re)
    return (kq,fq)