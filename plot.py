#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:26:01 2024

@author: mfurquan
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(m,d):
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')

    tri = np.empty((2*m.Ne,3),dtype='int')
    tri[:m.Ne,:] = m.ien[:,0:3]
    tri[m.Ne:,:2] = m.ien[:,2:4]
    tri[m.Ne:,2]  = m.ien[:,0]

    tcf=ax1.tricontourf(m.x[:,0],m.x[:,1],d[:,0],triangles=tri,levels=20)#, z_test_refi, levels=levels, cmap='terrain')
    fig1.colorbar(tcf)
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.set_aspect('equal')
    tcf=ax2.tricontourf(m.x[:,0],m.x[:,1],d[:,1],triangles=tri,levels=20)#, z_test_refi, levels=levels, cmap='terrain')
    fig2.colorbar(tcf)
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.set_aspect('equal')
    tcf=ax3.tricontourf(m.x[:,0],m.x[:,1],d[:,2],triangles=tri,levels=20)#, z_test_refi, levels=levels, cmap='terrain')
    fig3.colorbar(tcf)
    #q=ax3.quiver(m.x[:,0],m.x[:,1],d[:,0],d[:,1],scale=5.)
    plt.show()