# -*- coding: utf-8 -*-
"""
Improved Bayesian Optimization for nonlinear dynamic SEIR optimization system

@author: cyy
"""


import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import torch.nn.functional as F
import pyro.contrib.gp as gp
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import mplcyberpunk
import csv


def dS(S,I):
    ds = tao - beta*S*I - tao*S
    return ds

def dE(S,I,E):
    de = beta*S*I - (tao + alpha)*E
    return de

def dI(E,I,u):
    di = alpha*E - (tao + gama)*I - u*I
    return di
    
def RungeKutta(S,E,I,R,u):
    obj1 = [0]*st
    obj2 = 0
    for i in range(1,st):
        s1 = dS(S[i-1],I[i-1])
        e1 = dE(S[i-1],I[i-1],E[i-1])
        i1 = dI(E[i-1],I[i-1],u[i-1])
        
        s2 = dS(S[i-1]+(h/2)*s1,I[i-1]+(h/2)*i1)
        e2 = dE(S[i-1]+(h/2)*s1,I[i-1]+(h/2)*i1,E[i-1]+(h/2)*e1)
        i2 = dI(E[i-1]+(h/2)*e1,I[i-1]+(h/2)*i1,1/2*(u[i-1]+u[i]))
        
        s3 = dS(S[i-1]+(h/2)*s2,I[i-1]+(h/2)*i2)
        e3 = dE(S[i-1]+(h/2)*s2,I[i-1]+(h/2)*i2,E[i-1]+(h/2)*e2)
        i3 = dI(E[i-1]+(h/2)*e2,I[i-1]+(h/2)*i2,1/2*(u[i-1]+u[i]))
        
        s4 = dS(S[i-1]+h*s3,I[i-1]+h*i3)
        e4 = dE(S[i-1]+h*s3,I[i-1]+h*i3,E[i-1]+h*e3)
        i4 = dI(E[i-1]+h*e3,I[i-1]+h*i3,u[i])
        
        S[i] = S[i-1]+(h/6)*(s1+2*s2+2*s3+s4)
        E[i] = E[i-1]+(h/6)*(e1+2*e2+2*e3+e4)
        I[i] = I[i-1]+(h/6)*(i1+2*i2+2*i3+i4)
        
        S[i] = min(max(0,S[i]),1)
        E[i] = min(max(0,E[i]),1)
        I[i] = min(max(0,I[i]),1)                
        R[i] = 1 - S[i] - E[i] - I[i] 
    
        obj1[i] = C1*I[i] + C2*abs(0.3*torch.sin(u[i]*10) + torch.sin(13*u[i]) + 0.9*torch.sin(42*u[i]) + 0.2*torch.sin(12*u[i]) + u[i]*u[i]) 
        obj2 = obj2 + obj1[i]*h
    return obj2, S, E, I, R


def f(u):    
    S = [0]*st
    E = [0]*st
    I = [0]*st
    R = [0]*st
    S[0] = 0.5
    E[0] = 0.45
    I[0] = 0.05
    R[0] = 1 - S[0] - E[0] - I[0] 
    result1, result2, result3, result4, result5 = RungeKutta(S,E,I,R,u)          
    return result1, result2, result3, result4, result5


def lower_confidence_bound(x, kappa=2):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    return mu - kappa * sigma


def find_a_candidate(x_init, lower_bound=0, upper_bound=1):
    x_init = torch.tensor([x_init.detach().numpy()]) 
    constraint = constraints.interval(lower_bound, upper_bound)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)

    learning_rate = 0.055
    #minimizer = optim.Adamax([unconstrained_x],lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
    minimizer = optim.Adam([unconstrained_x],lr=learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.1, amsgrad=True)
    #minimizer = optim.AdamW([unconstrained_x],lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    #minimizer = optim.SGD([unconstrained_x],lr=learning_rate, momentum=0.5, dampening=0.2, weight_decay=0.1, nesterov=False)
    #minimizer = optim.Rprop([unconstrained_x],lr=learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    #minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')
    scheduler = StepLR(minimizer, step_size=3, gamma=0.1) #gamma is decay rate
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(minimizer, st, eta_min=0.0025)
    scheduler2 = ReduceLROnPlateau(minimizer, mode='min', factor=0.2, patience=2, verbose=True)
    scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.25, last_epoch=-1)

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y1= lower_confidence_bound(x, kappa=2)
        #scheduler.step()
        #autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        autograd.backward(unconstrained_x, autograd.grad(y1, unconstrained_x))
        return y1

    #minimizer.step(closure)
    y_min=10e8
    x_opt = torch.tensor([0.0]*st)
    for i in range(5):
        minimizer.step(closure)
        
        x = transform_to(constraint)(unconstrained_x)
        y_temp=closure()
        if (y_temp < y_min):
            y_min = y_temp
            x_opt=x
        #scheduler2.step(y_temp)
        scheduler3.step()
    return x_opt.detach()


def next_x(lower_bound=0, upper_bound=1, num_candidates=5):
    candidates = []
    values = []

    x_init = gpmodel.X[-1:][0]
    for i in range(num_candidates):
        x1 = find_a_candidate(x_init, lower_bound, upper_bound)  
        y1 = lower_confidence_bound(x1)
        candidates.append(x1)
        values.append(y1)
        x_init = x1.new_empty(st).uniform_(lower_bound, upper_bound)

    
    #print("Best values = ", torch.cat(values))
    #print("Best values = ", values)
    #print('Values type = ',type(values))
    #print(" ---->", np.argmin(values, axis=0))
    #argmin = torch.min(torch.cat(values), dim=0)[1].item()
    argmin = np.argmin(values, axis=0)
    return candidates[argmin]



def update_posterior(x_new):
    y = f(x_new[0])[0] # evaluate f at new point.
#    x_new = torch.tensor([x_new.numpy()])
    y = torch.tensor([y])
    X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation, cat means concatnate two torch array
    y = torch.cat([gpmodel.y, y]) # similar to append
    gpmodel.set_data(X, y) # optimize the GP hyperparameters using Adam with lr=0.001
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)



h = 0.5
st = 100  # days int(50/h)
tao = 0.00003
beta = 0.24
alpha = 0.33
gama = 0.25
C1=50000
C2=100.0

u1 = torch.rand(st,1)*0.1
y1 = f(u1)[0]
u2 = torch.rand(st,1)*0.4
y2 = f(u2)[0]
u3 = torch.rand(st,1)*0.8
y3 = f(u3)[0]
#X = torch.tensor([[0.01]*st,[0.5]*st,[0.95]*st])
X = torch.cat([torch.transpose(u1,0,1),torch.transpose(u2,0,1),torch.transpose(u3,0,1)],dim=0)
y = torch.tensor([y1,y2,y3])
gpmodel = gp.models.GPRegression(X,y, gp.kernels.Matern52(input_dim=st),noise=torch.tensor(0.1), jitter=1.0e-4)
#gpmodel = gp.models.GPRegression(X,y, gp.kernels.RBF(input_dim=st),mean_function=None, jitter=1.0e-4)

learning_rate = 0.1
weight_decay = 0.005
momentum = 0.9
optimizer = torch.optim.Adam(gpmodel.parameters(), lr=learning_rate)
scheduler3 = StepLR(optimizer, step_size=3, gamma=0.2) #gamma is decay rate
#scheduler = lr_scheduler.CosineAnnealingLR(optimizer, st, eta_min=learning_rate)
    
gp.util.train(gpmodel, optimizer);




x_sto = []
x_sto.append(torch.transpose(u1,0,1))
x_sto.append(torch.transpose(u2,0,1))
x_sto.append(torch.transpose(u3,0,1))

y_LB_min = []

localmin = []
localmin.append(y1.numpy())
localmin.append(y2.numpy())
localmin.append(y3.numpy())



for i in range(15):
    xmin = next_x()
    x_sto.append(xmin)
    localmin.append(f(xmin[0])[0].numpy())
    dist = []
    for j in range(0,len(x_sto)-1):
        dist.append(F.pairwise_distance(x_sto[j], x_sto[-1], p=2))
    #print('dist for iteration {} is {}'.format(i,dist))
    y_LB = lower_confidence_bound(xmin)
    y_LB_min.append(y_LB)
    update_posterior(xmin)



def FinalOptimal(xmin):
    constraint = constraints.interval(0, 1)
    unconstrained_x_init1 = transform_to(constraint).inv(xmin)
    unconstrained_x1 = unconstrained_x_init1.clone().detach().requires_grad_(True)
    learning_rate = 0.8
    minimizer = optim.Adam([unconstrained_x1],lr=learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.1, amsgrad=True)
    scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
    
    def closure1():
        minimizer.zero_grad()
        x_1= transform_to(constraint)(unconstrained_x1)
        y_1 = f(x_1[0])[0]
        autograd.backward(unconstrained_x1, autograd.grad(y_1, unconstrained_x1))
        #print(unconstrained_x1.grad)
        return y_1,x_1
    
    #minimizer.step(closure)
    y_min_1=10e8
    x_opt_1 = torch.tensor([0.0]*st)
    ite = 0
    q = 50
    epsilon = 0.0001
    while ite < q:
        minimizer.step(closure1)
        #x_final = transform_to(constraint)(unconstrained_x1)
        x_final = closure1()[1]
        y_temp_1=closure1()[0]
        if (y_temp_1 < y_min_1):
            y_min_1 = y_temp_1
            x_opt_1 = x_final
        x_sto.append(x_opt_1.detach())
        localmin.append(y_min_1.detach().numpy())
        #scheduler2.step(y_temp)
        scheduler3.step()
        if (abs(localmin[-1] - localmin[-2])/localmin[-2]) <= epsilon:
            break
        else:
            ite = ite + 1
    
    SusceptibleFinal = f(x_opt_1[0])[1]
    ExposeFinal = f(x_opt_1[0])[2]
    InfectFinal = f(x_opt_1[0])[3]
    RecoverFinal = f(x_opt_1[0])[4]
    
    return x_opt_1.detach(),y_min_1,SusceptibleFinal,ExposeFinal,InfectFinal,RecoverFinal



best = min(y_LB_min) 
FinalControl,FinalObj,FinalSusceptible,FinalExpose,FinalInfect,FinalRecover = FinalOptimal(x_sto[y_LB_min.index(best) + 3])

FinalControl = np.array(FinalControl[0])
FinalSusceptible = np.array(FinalSusceptible)
FinalExpose = np.array(FinalExpose)
FinalInfect = np.array(FinalInfect)
FinalRecover = np.array(FinalRecover)

NullControlExpose = []
NullControlInfect = []

with open('NULLnonconvex.csv','r') as bofile:
    null = list(csv.reader(bofile))
null=np.asarray(null)

for row in null:
    NullControlExpose.append(float(row[0]))
    NullControlInfect.append(float(row[1]))


#plt.style.use("cyberpunk")
time0 = list(range(0,len(FinalSusceptible)))
plt.figure(figsize=(8,4))
#plt.plot(time0, FinalSusceptible, color='blue')
line1, = plt.plot(time0, FinalExpose, color='green',linewidth=3)
line2, = plt.plot(time0, FinalInfect,':', color='green',linewidth=3)
#plt.plot(time0, FinalRecover, color='green')
line3, = plt.plot(time0, NullControlExpose, color='red',linewidth=3)
line4, = plt.plot(time0, NullControlInfect, ':',color='red',linewidth=3)
plt.legend([line3, line4, line1, line2],['Expose Population w\O Control','Infect Population w\O Control','Expose Population w\ IBO Optimal Control','Infect Population w\ IBO Optimal Control'],loc='upper right',fontsize = 12)
mplcyberpunk.make_lines_glow()
plt.xlabel('Time',fontsize = 15)
plt.ylabel('Population Rate',fontsize = 15)
#plt.title('Population rate over time',fontsize = 15)
plt.show()



# NullControlExpose = []
# NullControlInfect = []

# with open('NULLnonconvex.csv','r') as bofile:
#     null = list(csv.reader(bofile))
# null=np.asarray(null)

# for row in null:
#     NullControlExpose.append(float(row[0]))
#     NullControlInfect.append(float(row[1]))


# #plt.style.use("cyberpunk")
# time0 = list(range(0,len(FinalSusceptible)))
# plt.figure(figsize=(8,4))
# #plt.plot(time0, FinalSusceptible, color='blue')
# #line1, = plt.plot(time0, FinalExpose, color='green',linewidth=3)
# #line2, = plt.plot(time0, FinalInfect,':', color='green',linewidth=3)
# #plt.plot(time0, FinalRecover, color='green')
# line3, = plt.plot(time0, NullControlExpose, color='red',linewidth=3)
# line4, = plt.plot(time0, NullControlInfect, ':',color='red',linewidth=3)
# plt.legend([line3, line4],['Expose Population w\O Control','Infect Population w\O Control'],loc='upper right',fontsize = 12)
# mplcyberpunk.make_lines_glow()
# plt.xlabel('Time',fontsize = 15)
# plt.ylabel('Population Rate',fontsize = 15)
# #plt.title('Population rate over time',fontsize = 15)
# plt.show()



# Contorl over time
time1 = list(range(0,len(FinalControl)))
plt.figure(figsize=(8,4))
plt.plot(time1, FinalControl, color='green', marker='*', mec='g',ms=7)
plt.xlabel('Time')
plt.ylabel('FinalControl')
plt.title('Control strategy over time')
plt.show()


distance = []
for j in range(0,len(x_sto)-1):
    distance.append(F.pairwise_distance(x_sto[j], x_sto[-1], p=2))
distance = np.array(distance)
#print('distance of consecutive x ',distance)

    
iteration1 = list(range(1,len(x_sto)))
plt.figure(figsize=(8,4))
plt.plot(iteration1, distance, color='green', marker='*', mec='g',ms=7)
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.title('Distance between consecutive x‘s')
plt.show()


iteration2 = list(range(len(localmin)-3))
plt.figure(figsize=(8,4))
plt.plot(iteration2, localmin[3:], color='deeppink', ls='--',marker='o', mec='pink',ms=7)
plt.xlabel('Iteration')
plt.ylabel('Best Y')
plt.title('Value of best selected sample')
plt.show()
