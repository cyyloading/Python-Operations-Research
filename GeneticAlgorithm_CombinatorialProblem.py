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


def fitness(x,Dist,Flow):
    Dist = Dist.reindex(columns=x, index=x)
    Dist_Array = np.array(Dist)
    
    Obj_matrix = pd.DataFrame(Dist_Array*Flow)
    Obj_Array = np.array(Obj_matrix)
    Obj_sum = sum(sum(Obj_Array))
    
    return Obj_sum



# Use tournament selection method to select the parents, select the best one as a parent among each 3 individuals
def selection(all_population,Dist,Flow,k=3):
    all_parents = []
    for i in range(len(all_population)):
        temp_best = 10**8
        temp_best_parent = 0
        ran1 = random.sample(range(0,len(all_population)),k)  # generate k random values in range(0,N), ran1 contains 3 selected parents index
        for j in ran1:
            temp_parent = all_population[j]
            temp = fitness(temp_parent,Dist,Flow,)
            if temp<temp_best:
                temp_best = temp
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
            ran3 = random.sample(range(0,len(all_parents[0])),2) # generate 2 random values to represent two crossover locations
            ran3.sort() # make the index in ascending order
            
            parent1 = all_parents[i]
            p1_font = parent1 [:ran3[0]]          # the first segment of parent 1 
            p1_mid = parent1[ran3[0]:(ran3[1])]     # the second segment of parent 1 
 
            parent2 = all_parents[i+1]
            p2_font = parent2[:ran3[0]]          # the first segment of parent 2 
            p2_mid = parent2[ran3[0]:(ran3[1])]     # the second segment of parent 2 


            temp2_font = []
            for i in p2_font:
                    if i not in p1_mid:
                        temp2_font = np.append(temp2_font,i)
            child_1 = np.append(temp2_font, p1_mid)
            

            temp2_last = []
            for k in parent2:
                if k not in child_1:
                    temp2_last.append(k)
            child_1 = np.append(child_1, temp2_last)
             
            temp1_font = []
            for p in p1_font:
                    if p not in p2_mid:
                        temp1_font = np.append(temp1_font,p)
            child_2 = np.append(temp1_font, p2_mid)
            
            temp1_last = []
            for j in parent1:
                if j not in child_2:
                    temp1_last.append(j)
            child_2 = np.append(child_2, temp1_last)
            
        else:
            child_1 = all_parents[i]
            child_2 = all_parents[i+1]
        all_children.append(child_1)
        all_children.append(child_2)
    return all_children
        
        
def mutation(all_children):
   mutated_children = []
   for i in range(len(all_children)):
       ran4 = np.random.rand()
       if ran4<r_mut:
           ran5 = random.sample(range(0,len(all_children[0])),2)
           ran5.sort() # make the index in ascending order
           temp_seg = list(reversed(all_children[i][ran5[0]:ran5[1]+1]))
           temp_1 = all_children[i][:ran5[0]]
           temp_2 = all_children[i][ran5[1]+1:]
           all_children[i] = np.concatenate((temp_1, temp_seg, temp_2))
           
       mutated_children.append(all_children[i])
   return mutated_children
    
    
def GA(Dist,Flow,M,N,r_cross,r_mut):
    # initial population of random bitstring
    X0 = ['B','D','A','E','C','F','G','H']
    all_population = [np.array(random.sample(X0,len(X0))) for i in range(N)]
    
    score = [fitness(c,Dist,Flow) for c in all_population]
    best,best_score = 0,10**8
    for k in range(len(score)):
        if best_score>score[k]:
            best = all_population[k]
            best_score = score[k]
    
    for i in range(M):
        parents = selection(all_population,Dist,Flow)
        children = crossover(parents)
        mut_children = mutation(children)
        all_population = mut_children
        

        new_score = [fitness(c,Dist,Flow) for c in all_population]
        for p in range(len(new_score)):
            if best_score>new_score[p]:
                best = all_population[p]
                best_score = new_score[p]
    return best,best_score


        
M = 40  # define the total iterations
N = 600      # define the population size
r_cross = 0.9  # crossover rate
r_mut = 0.5  # mutation rate

solution = GA(init_Dist,Flow,M,N,r_cross,r_mut)
print('The best solution =',solution[0])
print('The best objective function value =',solution[1])
