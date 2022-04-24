# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:17:59 2022

@author: cyy
"""

# Quadratic Assignment Problem (minimization problem)

# Problem statement (From Wikipedia): There are a set of n facilities and a set of n locations. 
# For each pair of locations, a distance is specified, and for each pair of facilities a weight or flow is specified 
# (e.g., the amount of supplies transported between the two facilities). 
# The problem is to assign all facilities to different locations with the goal of minimizing the sum of the distances multiplied by the corresponding flows.

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random


init_Dist = pd.read_excel('CombinatorialProblemData.xlsx','Dist',index_col=0)
Flow = pd.read_excel('CombinatorialProblemData.xlsx','Flow',index_col=0) # flow between location pair won't change when facilities (X0) change


def F(x,Dist,Flow):
    Dist = Dist.reindex(columns=x, index=x)
    Dist_Array = np.array(Dist)
    
    Obj_matrix = pd.DataFrame(Dist_Array*Flow)
    Obj_Array = np.array(Obj_matrix)
    Obj_sum = sum(sum(Obj_Array))
    
    return Obj_sum



def SA(X0,Dist,Flow,T0,M,N,T_rate):
    T = []
    obj = []
    for i in range(M):
        for j in range(N):
            ran = random.sample(range(0,len(X0)),2)
            ran1 = ran[0]
            ran2 = ran[1]                   
            # two random values ran1 and ran2 are used as index to swap the facilities location in X0  
            # for example if ran1 = 0, ran2 = 3, X0=['B','D','A','E','C','F','G','H']   
            # after swap, x0 becomes ['E','D','A','B','C','F','G','H']      
            xt = X0.copy()
            xt[ran1],xt[ran2] = xt[ran2],xt[ran1]
           
            Obj_current = F(X0,Dist,Flow)
            Obj_neighbor = F(xt,Dist,Flow)
            
            ran3 = np.random.rand()
            probability = 1/(np.exp(Obj_neighbor - Obj_current)/T0)
            
            if Obj_neighbor < Obj_current:
                X0 = xt
            elif ran3 <= probability:
                X0 = xt
            else:
                X0 = X0
            
        T.append(T0)
        obj.append(Obj_current)
        
        T0 = T0*T_rate
        
        
    return X0, T, obj  
    


T0 = 1500 # initialize the start temperature with a high value
T0_max = T0 # For plotting use
M = 250 # the number of iterations to decrease the temperature
N = 20 # the number of iterations to search the neighborhood within each M iteration
T_rate = 0.9 # the update (decrease) rate of temperature


# initial the start solution
X0 = ['B','D','A','E','C','F','G','H']


solution = SA(X0, init_Dist, Flow, T0, M, N, T_rate)
Temperature = solution[1]
objective = solution[2]
print('final solution = ',solution[0])
print('final obj = ',objective[-1])


plt.figure(figsize=(10,6))
plt.plot(Temperature,objective)
plt.title('trend relationship between T and Obj', fontsize=20, fontweight='bold')
plt.xlabel('T', fontsize=18, fontweight='bold')
plt.ylabel('Obj', fontsize=18, fontweight='bold')
plt.xlim(T0_max, 0)
plt.xticks(np.arange(min(Temperature),max(Temperature),100), fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
