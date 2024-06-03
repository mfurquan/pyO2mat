#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:57:31 2024

@author: mfurquan
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

facemap = np.array([[0,1],[1,2],[2,3],[3,0]])
Nface = 4

class mesh:
    def __init__(self,Nsd,Nen,Nben,Nface,Nn,Ne,Nb,ibn,x,ien):
        self.Nsd = Nsd
        self.Nen = Nen
        self.Nben = Nben
        self.Nface = Nface
        self.Nn = Nn
        self.Ne = Ne
        self.Nb = Nb
        self.ibn = ibn
        self.x = x
        self.ien = ien
    def plot(self,ib):
        for k in range(self.Ne):
            plt.plot(self.x[np.append(self.ien[k,:],self.ien[k,0]),0],self.x[np.append(self.ien[k,:],self.ien[k,0]),1],'k-')
        for k in range(self.Nb[ib]):
            plt.plot(self.x[self.ien[self.ibn[ib][k,0],facemap[self.ibn[ib][k,1],:]],0],self.x[self.ien[self.ibn[ib][k,0],facemap[self.ibn[ib][k,1],:]],1],'rX')
    def join_mesh(self,m2,ib1,ib2):
        bn1 = edge2nod(self.ibn[ib1],self.ien,self.x)
        bn2 = edge2nod(m2.ibn[ib2],m2.ien,m2.x)
        mask = np.full([m2.Nn,m2.Nsd],False)
        mask[bn2,:] = True
        self.x = np.append(self.x, np.reshape(np.ma.masked_array(m2.x,mask).compressed(),[m2.Nn-np.size(bn2),m2.Nsd]),axis=0)
        ienc = m2.ien + self.Nn
        for k in range(np.size(bn2)):
            for i in range(m2.Ne):
                for j in range(m2.Nen):        
                    if m2.ien[i,j] == bn2[k]:
                        ienc[i,j] = bn1[k]
                    elif m2.ien[i,j] > bn2[k]:
                        ienc[i,j] -= 1
        self.ien = np.append(self.ien, ienc,axis=0)
        self.Nb = copy.deepcopy(self.Nb[:ib1] + self.Nb[ib1+1:] + m2.Nb[:ib2] + m2.Nb[ib2+1:])
        ibn = copy.deepcopy(self.ibn[:ib1] + self.ibn[ib1+1:] + m2.ibn[:ib2] + m2.ibn[ib2+1:])
        for i in range(len(self.ibn)-1,len(ibn)):
            ibn[i][:,0] += self.Ne
        self.ibn = ibn
        self.Nn = self.Nn + m2.Nn - np.size(bn1)
        self.Ne = self.Ne + m2.Ne
    def combine(self,ib1,ib2):
        self.ibn[ib1] = np.append(self.ibn[ib1],self.ibn[ib2],axis=0)
        self.ibn = self.ibn[:ib2]+self.ibn[ib2+1:]
        self.Nb[ib1] = self.Nb[ib1]+self.Nb[ib2]
        self.Nb = self.Nb[:ib2] + self.Nb[ib2+1:]

def edge2nod(ibn,ien,x,fn=np.sum):
    a = np.unique(np.array([ien[ibn[k,0],facemap[ibn[k,1],0]] for k in range(np.size(ibn,0))]
                         + [ien[ibn[k,0],facemap[ibn[k,1],1]] for k in range(np.size(ibn,0))]))
    d = np.vectorize(lambda n: fn(x[n,:]))(a)
    return a[np.argsort(d)]

def join_meshes(m1,m2,ib1,ib2):
    bn1 = edge2nod(m1.ibn[ib1],m1.ien,m1.x)
    bn2 = edge2nod(m2.ibn[ib2],m2.ien,m2.x)
    mask = np.full([m2.Nn,m2.Nsd],False)
    mask[bn2,:] = True
    x = np.append(m1.x, np.reshape(np.ma.masked_array(m2.x,mask).compressed(),[m2.Nn-np.size(bn2),m2.Nsd]),axis=0)
    ienc = m2.ien + m1.Nn
    for k in range(np.size(bn2)):
        for i in range(m2.Ne):
            for j in range(m2.Nen):        
                if m2.ien[i,j] == bn2[k]:
                    ienc[i,j] = bn1[k]
                elif m2.ien[i,j] > bn2[k]:
                    ienc[i,j] -= 1
    ien = np.append(m1.ien, ienc,axis=0)
    Nb = copy.deepcopy(m1.Nb[:ib1] + m1.Nb[ib1+1:] + m2.Nb[:ib2] + m2.Nb[ib2+1:])
    ibn = copy.deepcopy(m1.ibn[:ib1] + m1.ibn[ib1+1:] + m2.ibn[:ib2] + m2.ibn[ib2+1:])
    for i in range(len(m1.ibn)-1,len(ibn)):
        ibn[i][:,0] += m1.Ne
    Nn = m1.Nn + m2.Nn - np.size(bn1)
    Ne = m1.Ne + m2.Ne
    return mesh(m1.Nsd,m1.Nen,m1.Nben,Nn,Ne,Nb,ibn,x,ien)