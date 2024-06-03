#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:12:05 2024

@author: mfurquan
"""

#case 1:
# N = 5
# B = 5. # inner box: 2B x 2B
# Lxu = 4. # *B
# Lxd = 9. # *B
# Ly = 4. # *B
# n = 2. # poly mesh stretching
# Nr = 3*(N-1)
# Ntheta = 8*N
# Nxu = 5
# Nxd = 11
# Ny = N

# Re = 30.2
# C = 60.0


# case 2
N = 5*2
B = 5. # inner box: 2B x 2B
Lxu = 4. # *B
Lxd = 9. # *B
Ly = 4.*2 # *B
n = 3. # poly mesh stretching
Nr = 3*(N-1)
Ntheta = 8*N
Nxu = 5
Nxd = 15
Ny = N

Re = 30.2
C = 60.0