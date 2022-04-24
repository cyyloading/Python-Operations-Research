'''
CopyRight 2022
@author Yuyang Chen
All rights reserved. 
'''

# A: the treatment time for each person in queue, the index 0 in A represents the 1st person, index 1 in A represents the 2nd person, ...
# output 'quene' is the optimal queue with minimun total waiting time. For example the value '1' in 'queue' means the 1st person in original queue.


import numpy as np
import random

A = random.sample(range(1,101), 50)

queue = []
waiting_time = 0
B = A.copy()
for i in B:
    waiting_time += min(A)
    queue.append(B.index(min(A))+1)
    A.pop(A.index(min(A)))
    
    
waiting_time = waiting_time - max(B)
print(waiting_time,'\n',queue)
