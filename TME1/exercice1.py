# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:32:07 2020

@author: arnaud
"""
import numpy as np
from collections import Counter

#1.1
def entropie(vect):
    vect = np.array(vect)
    count = Counter(vect)
    entropie = 0
    for v in vect:
        entropie = entropie + count[v]*np.log(count[v])
    return -entropie

#1.2
def entropie_conditionnelle(list_vect)
    return

#1.4
    