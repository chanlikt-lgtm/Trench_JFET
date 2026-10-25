#!/usr/bin/env python3
"""Polynomial screening surrogate; NOT full self-consistent Vbd.

Trained from two-carrier ionization feedback on zero-II Poisson-DD 2D fields.
Training domain should be restricted to convex hull of reliable OFF samples.
"""
import numpy as np
COEFFICIENTS=[42.52690354913682, -0.21797459510121886, 2.2153703829295646, 19.072902575540038, 12.500286961378931, 7.998982854902565, -10.665675459118951, -14.942257655864436, -32.244826791332414, -13.590443097111145]
TERMS=[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]

def predict_vbd_screen(a_um, NA_cm3):
    a=np.asarray(a_um,dtype=float)
    NA=np.asarray(NA_cm3,dtype=float)
    if np.any((a<0.3)|(a>0.8)|(NA<3.16227766017e+16)|(NA>6.81292069058e+17)):
        raise ValueError('Outside training bounding range')
    u=(a-.5)/.3
    v=np.log10(NA)-17.5
    return sum(c*u**i*v**j for c,(i,j) in zip(COEFFICIENTS,TERMS))

if __name__=='__main__':
    print(predict_vbd_screen(0.5,1e17))
