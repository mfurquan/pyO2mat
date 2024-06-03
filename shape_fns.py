#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:19:51 2024

@author: mfurquan
"""
import numpy as np

Nsd = 2
Nen = 4

def shape_fn(xi):
    return np.array([(1.-xi[0])*(1.-xi[1]),
                     (1.+xi[0])*(1.-xi[1]),
                     (1.+xi[0])*(1.+xi[1]),
                     (1.-xi[0])*(1.+xi[1])])/4.

def deriv_shape_fn(xi):
    return np.array([[-(1.-xi[1]),(1.-xi[1]),(1.+xi[1]),-(1.+xi[1])],
                     [-(1.-xi[0]),-(1.+xi[0]),(1.+xi[0]),(1.-xi[0])]])/4.
