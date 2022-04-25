# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:23:22 2022

@author: TA005
"""

import numpy as np
import random



# Define the continous objective function and return the objective function value
def fitness(x,lb,ub): # x can be multi-dimensional array, lb is lower bounds of each dimensions, ub is upper bounds of each dimensions
    t = 1  # because we start at the last element of the x [index -1] to decode
    decoded_x = []
    for i in range(dim):
        z = 0  # because we start at 2^0 when decode each dimension
        bit_sum = 0
        for j in range(n_bits):
            bit = x[-t]*(2**z)
            bit_sum = bit_sum + bit
            t = t+1
            z = z+1
        
        precision = (ub[i]-lb[i])/((2**n_bits)-1)
        decoded_x.append(bit_sum*precision+lb[i])
    
    # the objective function is himmelblau function in our example
    # min ((x**2)+y-11)**2+(x+(y**2)-7)**2
    obj = ((decoded_x[0]**2)+decoded_x[1]-11)**2+(decoded_x[0]+(decoded_x[1]**2)-7)**2
    return decoded_x, obj
    
            
        
# Use tournament selection method to select the parents, select the best one as a parent among each 3 individuals
def selection(all_population,lb,ub,k=3):
    all_parents = []
    for i in range(len(all_population)):
        temp_best = 10**8
        temp_best_parent = 0
        ran1 = random.sample(range(0,len(all_population)),k)  # generate k random values in range(0,N), ran1 contains 3 selected parents index
        for j in ran1:
            temp_parent = all_population[j]
            temp = fitness(temp_parent,lb,ub)
            if temp[1]<temp_best:
                temp_best = temp[1]
                temp_best_parent = temp_parent

        all_parents.append(temp_best_parent)
    return all_parents


# There are many ways to crossover: one-point crossover, two-points crossover, multi-points crossover
# In our example, we use two-points crossover
def crossover(all_parents):
    all_children = []
    for i in range(len(all_parents)//2):
        ran2 = np.random.rand() 
        if ran2<r_cross:
            ran3 = random.sample(range(0,n_bits*dim),2) # generate 2 random values to represent two crossover locations
            ran3.sort() # make the index in ascending order
            child_1 = np.concatenate((all_parents[i][:ran3[0]], all_parents[i+1][ran3[0]:(ran3[1]+1)], all_parents[i][(ran3[1]+1):]))   # merge the first and third segments of parent1 with the second segment of parent2 as child_1
            child_2 = np.concatenate((all_parents[i+1][:ran3[0]], all_parents[i][ran3[0]:(ran3[1]+1)], all_parents[i+1][(ran3[1]+1):]))   # merge the first and third segments of parent2 with the second segment of parent1 as child_2
        else:
            child_1 = all_parents[i]
            child_2 = all_parents[i+1]
        all_children.append(child_1)
        all_children.append(child_2)
    return all_children
        
        
def mutation(all_children):
   mutated_children = []
   for i in range(len(all_children)):
       for j in range(n_bits*dim):
           ran4 = np.random.rand()
           if ran4<r_mut:
               all_children[i][j] = 1 - all_children[i][j]
       mutated_children.append(all_children[i])
   return mutated_children
    
    
def GA(M,N,n_bits,dim,r_cross,r_mut,lower_bound,upper_bound):
    # initial population of random bitstring
    all_population = np.random.randint(0,2,size=(N,n_bits*dim))  # create population with N inidivuals, the chromosome total length of each individual is n_bits*dim (dim is dimension of variable, n_bits is the length of each dimension)
    
    score = [fitness(c,lower_bound,upper_bound) for c in all_population]
    best,best_score = 0,10**8
    for k in score:
        if best_score>k[1]:
            best = k[0]
            best_score = k[1]
    
    for i in range(M):
        parents = selection(all_population,lower_bound,upper_bound)
        children = crossover(parents)
        mut_children = mutation(children)
        all_population = mut_children
        
        new_score = [fitness(c,lower_bound,upper_bound) for c in all_population]
        for p in new_score:
            if best_score>p[1]:
                best = p[0]
                best_score = p[1]
    return best, best_score


        
M = 100  # define the total iterations
N = 100      # define the population size
n_bits = 20  # bits (the number of genes in chromosome)
dim = 2 # dimensions of the variable (solution)
r_cross = 0.9  # crossover rate
r_mut = 1.0 / float(n_bits)  # mutation rate
lower_bound = [-6,-6] # our example is 2-dimensional optimization problem, the lower bounds of each dimension are -6
upper_bound = [6,6]  # the lower bounds of each dimension are 6

solution = GA(M,N,n_bits,dim,r_cross,r_mut,lower_bound,upper_bound)
print('The best solution = [%5.2f, %5.2f]'%(solution[0][0],solution[0][1]))
print('The best objective function value = %0.3f'%solution[1])