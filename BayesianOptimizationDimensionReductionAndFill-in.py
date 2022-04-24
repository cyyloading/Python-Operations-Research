# -*- coding: utf-8 -*-
"""
Improved Bayesian Optimization algorithm for dynamic linear SEIR optimization system
    
@author: cyy
    
# Colab package installations:
!pip3 install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.0-{platform}-linux_x86_64.whl
!pip3 install torchvision
!pip3 install pyro-ppl
"""


### This code is implementing for different fill-in strategies


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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


def dropout(strategy):
    BestControl_copy = torch.linspace(-2,-1,st)
    count = 0
    last = -1

    
    if strategy == 'Identical Value':
        for h1 in range(int(step[w]-1),st,int(step[w])):
            if count<len(BestControl):
                for t in range(last+1,st):
                    BestControl_copy[t] = BestControl[count]
            count = count + 1
            last = h1 
            
        if step[w] == 1:
            BestControl_copy = BestControl
            
    elif strategy == 'Uniform Distribution':
        for h1 in range(int(step[w]-1),st,int(step[w])):
            if count<len(BestControl):
                for t in range(last+1,st):   
                    if count == 0:
                        BestControl_copy[t] = BestControl[count]
                    else:
                        BestControl_copy[t] = BestControl[count].new_empty(1).uniform_(min(BestControl[count-1:count+1]),max(BestControl[count-1:count+1]))
            count = count + 1
            last = h1 
            
        if step[w] == 1:
            BestControl_copy = BestControl
            
            
    elif strategy == 'Gaussian Regression':
        # Build a model
        #K= ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))
        K = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
        gp0 = GaussianProcessRegressor(kernel=K)
         
        # Some data
        xobs = [[h1] for h1 in range(int(step[w]-1),st,int(step[w]))]
        xobs = np.array(xobs[:len(BestControl)])
        yobs = np.array(BestControl.numpy())
             
        # Fit the model to the data (optimize hyper parameters)
        gp0.fit(xobs, yobs)
        
        for h1 in range(int(step[w]-1),st,int(step[w])):
            if count<len(BestControl):
                for t in range(last+1,st):
                    if count == 0:
                        BestControl_copy[t] = BestControl[count]
                    else:
                        x_test = np.array([[t]])
                        means,sigmas = gp0.predict(x_test,return_std=True)
                        #x_test_tem = gp0.sample_y(x_test,1 )
                        x_test_tem = np.random.normal(means,0,1)
                        BestControl_copy[t] = max(0,min(torch.tensor(x_test_tem.item()),1))
            count = count + 1
            last = h1
        
                
        if w == 0:
            BestControl_copy = torch.tensor([1.0]*st)
        if w == 1:
            BestControl_copy = torch.tensor([0.999]*st)
        if step[w] == 1:
            BestControl_copy = BestControl
            
            
    elif strategy == 'Normal Distribution':
        for h1 in range(int(step[w]-1),st,int(step[w])):
            if count<len(BestControl):
                for t in range(last+1,st):
                    if count == 0:
                        BestControl_copy[t] = BestControl[count]
                    else:
                        means = BestControl[count-1:count+1].mean()
                        sigmas = BestControl[count-1:count+1].std()
                        BestControl_copy[t] = max(0,min(torch.normal(means,sigmas),1))
            count = count + 1
            last = h1
            
        if step[w] == 1:
            BestControl_copy = BestControl
            
            
    elif strategy == 'Linear Approximate':
        xobs = [h1 for h1 in range(int(step[w]-1),st,int(step[w]))]
        xobs = np.array(xobs[:len(BestControl)])
        yobs = np.array(BestControl.numpy())
    
        for h1 in range(int(step[w]-1),st,int(step[w])):
            if count<len(BestControl):
                for t in range(last+1,st):
                    if count == 0:
                        BestControl_copy[t] = BestControl[count]
                    else:
                        modeltrain = np.polyfit(xobs[count-1:count+1],yobs[count-1:count+1],1)
                        y_test = t*modeltrain[0] + modeltrain[1] 
                        BestControl_copy[t] = max(0,min(torch.tensor(y_test.item()),1))
            count = count + 1
            last = h1 
            
        if step[w] == 1:
            BestControl_copy = BestControl

    return BestControl_copy


def dS(S,I):
    ds = tao - beta*S*I - tao*S
    return ds

def dE(S,I,E):
    de = beta*S*I - (tao + alpha)*E
    return de

def dI(E,I,u):
    di = alpha*E - (tao + gama)*I - u*I
    return di
    
def RungeKutta(S,E,I,R,u,st_estimate):
    obj1 = [0]*st_estimate
    obj2 = 0
    for i in range(1,st_estimate):
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
    
        obj1[i] = C1*I[i] + C2*(0.3*abs(torch.sin(10.0*u[i])) + abs(torch.sin(u[i])) + 0.9*abs(torch.sin(u[i])) + 0.2*abs(torch.sin(u[i])) + u[i]*u[i]) 
        obj2 = obj2 + obj1[i]*h
    return obj2, S, E, I, R


def f(u,st_estimate):    
    S = [0]*st_estimate
    E = [0]*st_estimate
    I = [0]*st_estimate
    R = [0]*st_estimate
    S[0] = 0.57
    E[0] = 0.43
    I[0] = 0.0
    R[0] = 0.0
    result1, result2, result3, result4, result5 = RungeKutta(S,E,I,R,u,st_estimate)          
    return result1, result2, result3, result4, result5


def lower_confidence_bound(x, kappa=2):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    #posterior = gpmodel.sample_y
    return mu - kappa * sigma


def find_a_candidate(x_init, lower_bound, upper_bound, points):
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
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(minimizer, points, eta_min=0.0025)
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
    x_opt = torch.tensor([[0.0]*numpoints[w]])
    for i in range(1):
        minimizer.step(closure)                
        x = transform_to(constraint)(unconstrained_x)
        y_temp=closure()
        if (y_temp < y_min):
            y_min = y_temp
            x_opt=x
        #scheduler2.step(y_temp)
        scheduler3.step()
    return x_opt.detach()


def next_x(X, y, n_bandit, reward, points, lower_bound_1, upper_bound_1, lower_bound=0, upper_bound=1):
    u_cut = np.linspace(lower_bound, upper_bound, n_bandit)
    u_section = []
    for i in range(n_bandit - 1):
        u_section.append([u_cut[i],u_cut[i+1]])
        
    candidates = []
    values = []
    index = []
    
    candidates_1 = []
    values_1 = []
    x_init = gpmodel.X[-1:][0]
    for j in range(len(u_section)):
        num_candidates = reward[j]
    #X_grid = torch.linspace(u_section[j][0], u_section[j][1]-0.01,num_candidates)
    
    X_grid = torch.linspace(u_section[j][0], u_section[j][1]-0.01,num_candidates*numpoints[w])
    X_sample = torch.reshape(X_grid,(num_candidates,numpoints[w]))
    
    for i in range(num_candidates):
        #x_init = torch.ones(1,numpoints[w])*X_grid[i].item()
        #x1 = find_a_candidate(x_init[0], u_section[j][0], u_section[j][1],points) 
        #print(X_sample[i],'\nsort = ',torch.sort(X_sample[i],descending=True)[0])
        x_init = torch.sort(X_sample[i],descending=True)[0]
        x1 = find_a_candidate(x_init, u_section[j][0], u_section[j][1],points)                 
        y1 = lower_confidence_bound(x1)

        candidates.append(x1)
        values.append(y1)
        index.append(j)

    sorted,indices = torch.sort(torch.tensor(values))
    reward[index[indices[0]]] += 1
    reward[index[indices[1]]] += 1
    reward[index[indices[-2]]] -= 1
    reward[index[indices[-1]]] -= 1

    x_init_1 = gpmodel.X[-1:][0]
    for i in range(20):
        x_init_1 = x_init_1.new_empty(numpoints[w]).uniform_(lower_bound_1, upper_bound_1)
        x1_1 = find_a_candidate(x_init_1, lower_bound, upper_bound,points)
        y1_1 = lower_confidence_bound(x1_1)
        candidates_1.append(x1_1)
        values_1.append(y1_1)

    min_y1 = np.min(values)    # Thompson sampling
    min_y1_1 = np.min(values_1)   # uniform sampling

    if min_y1 < min_y1_1:
        argmin = np.argmin(values, axis=0)
        candidate_best = candidates[argmin]
        lower_bound_1 += 0.03
        #print('thompson')
    else:
        argmin = np.argmin(values_1, axis=0)
        candidate_best = candidates_1[argmin]
        #print('uniform')

    return candidate_best,reward,X,y,lower_bound_1



def update_posterior(x_new,y_LB):
    y = y_LB
    y = torch.tensor([y])
    X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation, cat means concatnate two torch array
    y = torch.cat([gpmodel.y, y]) # similar to append
    gpmodel.set_data(X, y) # optimize the GP hyperparameters using Adam with lr=0.001
    #optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.005)
    #scheduler3 = StepLR(optimizer, step_size=3, gamma=0.2)
    #gp.util.train(gpmodel, optimizer)
    
def FinalOptimal(xmin):
    #print('x_sto[Index][0].numpy() = ',min(xmin[0]).item(),'\n',max(xmin[0]).item())
    constraint = constraints.interval(0, 1)
    unconstrained_x_init1 = transform_to(constraint).inv(xmin)
    unconstrained_x1 = unconstrained_x_init1.clone().detach().requires_grad_(True)
    learning_rate = 3.5
    minimizer = optim.Adam([unconstrained_x1],lr=learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.1, amsgrad=True)
    #scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
    
    def closure1():
        minimizer.zero_grad()
        x_1= transform_to(constraint)(unconstrained_x1)
        y_1 = f(x_1[0],numpoints[w])[0]
        autograd.backward(unconstrained_x1, autograd.grad(y_1, unconstrained_x1))
        return y_1,x_1
    
    #minimizer.step(closure)
    
    x_opt_1 = torch.tensor([[0.0]*st])
    ite = 0
    q = 20
    while ite < q:
        y_min_1=10e8
        minimizer.step(closure1)
        #x_final = transform_to(constraint)(unconstrained_x1)
        x_final = closure1()[1]
        y_temp_1=closure1()[0]
        if (y_temp_1 < y_min_1):
            y_min_1 = y_temp_1
            x_opt_1 = x_final
            localmin.append(y_min_1.detach().numpy())
        else:
            localmin.append(y_min_1)
        x_sto.append(x_opt_1.detach())
        ite = ite + 1
    
    SusceptibleFinal = f(x_opt_1[0],numpoints[w])[1]
    ExposeFinal = f(x_opt_1[0],numpoints[w])[2]
    InfectFinal = f(x_opt_1[0],numpoints[w])[3]
    RecoverFinal = f(x_opt_1[0],numpoints[w])[4]
    
    return x_opt_1.detach(),y_min_1,SusceptibleFinal,ExposeFinal,InfectFinal,RecoverFinal


torch.manual_seed(101)
strategy = ['Identical Value','Uniform Distribution','Gaussian Regression','Normal Distribution','Linear Approximate']

tao = 0.00003
beta = 0.4  
alpha = 0.33 
gama = 0.15
C1=25000  
C2=C1/500.0  

numpoints = [5,10,20,30,40,55,60,70,80,90,100]
step = [20,10,5,100/30,2.5,100/55,100/60,100/70,1.25,100/90,1]
st = 100
OBJResults = []

for gg in range(0,5):
    print('\033[1;31mThe Result for Dropout Strategy =\033[0m',strategy[gg],'\033[1;31m:\033[0m')
   
    allcontrol =[]
    allOBJ = []

    for w in range(0,len(step)):
        h = 0.5
        x_sto = []
        y_LB_min = []
        localmin = []
        
        x_bounds = (0,1)  
        interval_resolution = 20*numpoints[w]
        X_grid = torch.linspace(x_bounds[0], x_bounds[1]-0.01, interval_resolution)
        X_sample = torch.reshape(X_grid,(int(interval_resolution/numpoints[w]),numpoints[w]))
        X_grid_0 = np.linspace(x_bounds[0], x_bounds[1]-0.01, interval_resolution)
        X_sample_0 = X_grid_0.reshape(int(interval_resolution/numpoints[w]),numpoints[w])

        number = 20
        X = X_sample[0:20]
        y = []

        for i in range(number):
            X_data = [1]*numpoints[w]
            X_tem = X_sample[i]
            Y_tem = f(X_tem,numpoints[w])
            X_data = X_sample_0[i]
            XX = torch.tensor([X_data.tolist()])
        
            y_tem = Y_tem[0].item()
            y.append(y_tem)
        
            x_sto.append(XX)
            localmin.append(Y_tem[0].numpy())
            y_LB_min.append(Y_tem[0])
        
        y = torch.tensor(y)
        #gpmodel = gp.models.GPRegression(X,y, gp.kernels.Matern52(input_dim=numpoints[w], lengthscale=lscale, active_dims=active_dms),noise=torch.tensor(0.1), jitter=1.0e-1)
        gpmodel = gp.models.GPRegression(X,y, gp.kernels.Matern52(input_dim=numpoints[w]),noise=torch.tensor(0.1), jitter=1.0e-4)
        #gpmodel = gp.models.GPRegression(X,y, gp.kernels.RBF(input_dim=st),mean_function=None, jitter=1.0e-4)
        
        learning_rate = 0.1
        weight_decay = 0.005
        momentum = 0.9
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=learning_rate)
        scheduler3 = StepLR(optimizer, step_size=3, gamma=0.2) #gamma is decay rate
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, st, eta_min=learning_rate)
        gp.util.train(gpmodel, optimizer);
        
        
        n_bandit = 4
        reward = [3]*(n_bandit-1)   
        
        lower_bound_1 = 0
        upper_bound_1 = 1 
        
        for i in range(15):
            xmin,reward,X,y,lower_bound_1 = next_x(X,y,n_bandit,reward, numpoints[w], lower_bound_1, upper_bound_1)
            x_sto.append(xmin)
            localmin.append(f(xmin[0],numpoints[w])[0].numpy())
            dist = []
            for j in range(0,len(x_sto)-1):
                dist.append(F.pairwise_distance(x_sto[j], x_sto[-1], p=2))
            y_LB = f(xmin[0],numpoints[w])[0]
            y_LB_min.append(y_LB)
            update_posterior(xmin,y_LB)
        
        best = min(y_LB_min) 
        Index = y_LB_min.index(best)        
        x_best = torch.tensor([x_sto[Index].numpy()])
        x_best = x_best[0]
                
        FinalControl,FinalObj,FinalSusceptible,FinalExpose,FinalInfect,FinalRecover = FinalOptimal(x_best)
        MinimumObj = min(localmin)
        MinIndex = localmin.index(MinimumObj)        
        BestControl = x_sto[MinIndex]     
        BestControl =  BestControl[0]          
        BestControl_copy = dropout(strategy[gg])
        BestObj = f(BestControl_copy,st)[0].item()
        #print('\nWhen # of points = ',numpoints[w],', Optimal objective value =',BestObj)

        allcontrol.append(BestControl_copy.numpy())
        allOBJ.append(BestObj)
        
    OBJResults.append(allOBJ)
    
    plt.figure(figsize=(8,4))
    line1,=plt.plot(allcontrol[0],'g-^')
    line2,=plt.plot(allcontrol[1],'r--P')
    line3,=plt.plot(allcontrol[2],'b-.p')
    line4,=plt.plot(allcontrol[3],'m:o')
    line5,=plt.plot(allcontrol[4],'k--D')
    line6,=plt.plot(allcontrol[5],'y-*')
    line7,=plt.plot(allcontrol[6],'g:d')
    line8,=plt.plot(allcontrol[7],'r--H')
    line9,=plt.plot(allcontrol[8],'m-+')
    line10,=plt.plot(allcontrol[9],'b:<')
    line11,=plt.plot(allcontrol[10],'c-->')
    plt.legend([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11],["d = 5", "d = 10","d = 20","d = 30","d = 40", "d = 50","d = 60","d = 70", "d = 80", "d = 90","d = 100"],loc='lower left')
    plt.xlabel('Time')
    plt.ylabel('BestControl')
    #plt.title('Control strategy over time for different selected dimensions')
    plt.show()
        
    print('\n\n')

    
plt.figure(figsize=(14,6))
x0 = [5,10,20,30,40,50,60,70,80,90,100]
line21,=plt.plot(x0,OBJResults[0],'g-^')
line22,=plt.plot(x0,OBJResults[1],'r--P')
line23,=plt.plot(x0,OBJResults[2],'b-.p')
line24,=plt.plot(x0,OBJResults[3],'m:o')
line25,=plt.plot(x0,OBJResults[4],'k--D')


plt.legend([line21,line22,line23,line24,line25],['Identical Value','Uniform Distribution','Gaussian Regression','Normal Distribution','Linear Approximate'],loc='upper right')
plt.xlabel('d',fontsize=20)
plt.ylabel('Best Objective Function Value',fontsize=20)
#plt.title('Infectious population rate over time for different selected dimensions')
plt.show()  
