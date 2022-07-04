# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:20:48 2022

@author: m1380
"""
import numpy as np
mylist=[[4,3],[5,2],[7,4],[2,1]]
list=np.array(mylist)
print(np.array(mylist).shape)

i=sorted(mylist, key=lambda x:x[1])
print(i)
print(list[:,1])