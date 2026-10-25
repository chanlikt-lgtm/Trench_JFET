#!/usr/bin/env python3
"""Surrogate regression for exploratory JFET Ld=10um e/h avalanche-onset model.
Validity restricted to computed reliable region. NOT measured VBD.
"""
import numpy as np
COEFF=[40.575192616088415, -16.650720149055342, -9.199092889879378, 35.89987202841777, 15.007656978624176, 13.32913769915503]
TERMS=[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
A_MIN=0.3
A_MAX=0.6
NA_MIN=3.16227766017e+16
NA_MAX=3.16227766017e+18

def predict_vbd(a_um, NA_cm3):
    a_um=np.asarray(a_um, dtype=float)
    NA_cm3=np.asarray(NA_cm3, dtype=float)
    if np.any((a_um<A_MIN)|(a_um>A_MAX)|(NA_cm3<NA_MIN)|(NA_cm3>NA_MAX)):
        raise ValueError('Outside reliable fitted bounding range')
    u=(a_um-0.5)/0.3
    v=np.log10(NA_cm3)-17.5
    return sum(cc*u**ii*v**jj for cc,(ii,jj) in zip(COEFF,TERMS))

if __name__=='__main__':
    print('Example:',predict_vbd(0.5,1e17))
