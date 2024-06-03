#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:21:45 2024

@author: mfurquan
"""
import numpy as np

Nint = 2
Nint2d = Nint**2

match Nint:
    case 1:
        xg = np.array([0.])
        wg = np.array([2.])
    case 2:
        xg = np.array([-1.,1.])/np.sqrt(3.)
        wg = np.array([1.,1.])
    case 3:
        xg = np.array([-1.,0.,1.])*np.sqrt(3./5.)
        wg = np.array([5.,8.,5.])/9.

Nint2d = Nint**2
xg2d = np.array([[x,y] for x in xg for y in xg])
wg2d = np.array([x*y for x in wg for y in wg])

xgb = np.empty((4,Nint,2))
xgb[0,:,:] = np.array([[x,-1.] for x in xg])
xgb[1,:,:] = np.array([[1.,y] for y in xg])
xgb[2,:,:] = np.array([[x,1.] for x in xg])
xgb[3,:,:] = np.array([[-1.,y] for y in xg])
