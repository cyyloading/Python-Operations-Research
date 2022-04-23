# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:24:25 2022

@author: TA005
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the objective function: use himmelblau's function as example
def F(x,y):
    f = ((x**2)+y-11)**2 + (x+(y**2)-7)**2
    return f



def SimulatedAnnealing(x,y,T0,M,N,T_rate,k):            
    T = []
    Obj = []
    for i in range(M): # temperature decrease iteration loop
        for j in range(N): # at each temperature, look for N neighborhoods
            
            # random the x value of neighborhood
            x1 = np.random.rand()
            random_value1 = np.random.rand()
            
            if random_value1 > 0.5: # greater than 0.5, then direction of neighbor will be positive (increase)
                x_step = k*x1
            else:
                x_step = -k*x1
                
            # random the y value of neighborhood
            y1 = np.random.rand()
            random_value2 = np.random.rand()
            
            if random_value2 > 0.5: # greater than 0.5, then direction of neighbor will be positive (increase)
                y_step = k*y1
            else:
                y_step = -k*y1
            
            x_neighbor = x + x_step
            y_neighbor = y + y_step
            
            obj_neighbor = F(x_neighbor, y_neighbor)
            obj_current = F(x, y)
            
            rand_value3 = np.random.rand()
            probability = 1/(np.exp((obj_neighbor - obj_current)/T0))
            
            if obj_neighbor < obj_current:
                x = x_neighbor
                y = y_neighbor
            
            elif rand_value3 <= probability:
                x = x_neighbor
                y = y_neighbor
            
            else:
                x = x
                y = y
            
        
        T.append(T0)
        Obj.append(obj_current)
        
        T0 = T0*T_rate
            
    
    return x, y, T, Obj  
    


T0 = 1000 # initialize the start temperature with a high value
T0_max = T0 # For plotting use
M = 250 # the number of iterations to decrease the temperature
N = 20 # the number of iterations to search the neighborhood within each M iteration
T_rate = 0.9 # the update (decrease) rate of temperature
k = 0.1 # used to reduce the neighborhood-search step-size
        # for example, current x value is 1, the random neighborhood will be (1+k*np.random.rand())
        # Since (1+np.random.rand()) without multipling k will result in the neighborhood point with large value (means far away from current point)

# initial the start point
x = 2
y = 2

solution = SimulatedAnnealing(x, y, T0, M, N, T_rate, k)
Temperature = solution[2]
objective = solution[3]
print('final x = %0.3f'% solution[0])
print('final y = %0.3f'% solution[1])
print('final obj = %0.3f'% objective[-1])


plt.plot(Temperature,objective)
plt.title('trend relationship between T and Obj', fontsize=20, fontweight='bold')
plt.xlabel('T', fontsize=18, fontweight='bold')
plt.ylabel('Obj', fontsize=18, fontweight='bold')
plt.xlim(T0_max, 0)
plt.xticks(np.arange(min(Temperature),max(Temperature),100), fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
