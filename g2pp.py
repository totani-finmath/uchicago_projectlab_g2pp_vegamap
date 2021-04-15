#%% packages
import numpy as np
import pandas as pd
import scipy.integrate as itg

#%% constants
T = np.linspace(2.0, 7.0, num=11)
tau = [t2 - t1 for t1, t2 in zip(T[:-1], T[1:])]

#%% parameters
alpha1 = 0.01
alpha2 = 0.02
sigma1 = 0.1
sigma2 = 0.2
rho = 0.1
X = 0.50

#%% functions: convert parameters
def M_x_g2pp(s,t,T,alpha1,alpha2,sigma1,sigma2,rho):
    trm1 = (sigma1**2/alpha1**2 + rho*(sigma1*sigma2)/alpha1/alpha2) * (1.0 - np.exp(-alpha1*(t-s)))
    trm2 = -1.0 * sigma1**2*0.5/alpha1**2 * (np.exp(-alpha1*(T-t)) - np.exp(-alpha1*(T+t-2*s)))
    trm3 = -1.0 * rho*sigma1*sigma2/alpha2/(alpha1+alpha2) * (np.exp(-alpha2*(T-t)) - np.exp(-alpha2*T-alpha1*t + (alpha1+alpha2)*s))
    return trm1 + trm2 + trm3
def M_y_g2pp(s,t,T,alpha1,alpha2,sigma1,sigma2,rho):
    trm1 = (sigma2**2/alpha2**2 + rho*(sigma1*sigma2)/alpha1/alpha2) * (1.0 - np.exp(-alpha2*(t-s)))
    trm2 = -1.0 * sigma2**2*0.5/alpha2**2 * (np.exp(-alpha2*(T-t)) - np.exp(-alpha2*(T+t-2*s)))
    trm3 = -1.0 * rho*sigma1*sigma2/alpha1/(alpha1+alpha2) * (np.exp(-alpha1*(T-t)) - np.exp(-alpha1*T-alpha2*t + (alpha1+alpha2)*s))
    return trm1 + trm2 + trm3
def sigma_x_g2pp(T,alpha1,sigma1):
    return sigma1 * np.sqrt((1.0-np.exp(-2*alpha1*T))*0.5/alpha1)
def sigma_y_g2pp(T,alpha2,sigma2):
    return sigma2 * np.sqrt((1.0-np.exp(-2*alpha2*T))*0.5/alpha2)
def rho_xy_g2pp(T,alpha1,alpha2,sigma1,sigma2,rho):
    return rho*sigma1*sigma2/(alpha1 + alpha2)/sigma1/sigma2 * (1.0 - np.exp(-(alpha1+alpha2)*T))
def ci(i):
    if i is not n:
        return X * tau(i)
    else:
        return 1.0 + X * tau(i)

#%% functions: bond pricings
def V(t,T,alpha1,alpha2,sigma1,sigma2,rho):
    f1 = lambda x: (1.0-np.exp(-alpha1*(T-x)))**2
    trm1 = sigma1**2/alpha1**2 * itg.quad(f1,t,T)
    f2 = lambda x: (1.0-np.exp(-alpha2*(T-x)))**2
    trm2 = sigma2**2/alpha2**2 * itg.quad(f2,t,T)
    f3 = lambda x: (1.0-np.exp(-alpha1*(T-x))) * (1.0-np.exp(-alpha2*(T-x)))
    trm3 = 2*sigma1*sigma2/alpha1/alpha2 * itg.quad(f3,t,T)
    return trm1 + trm2 + trm3
def A(t,T,alpha1,alpha2,sigma1,sigma2,rho):
    return PM(0,T)/PM(0,t)*np.exp(0.5*(V(t,T,alpha1,alpha2,sigma1,sigma2,rho)\
                                    - V(0,T,alpha1,alpha2,sigma1,sigma2,rho)\
                                    + V(0,t,alpha1,alpha2,sigma1,sigma2,rho)))
def B(z,t,T):
    return (1.0-np.exp(-z*(T-t)))/z

#%% functions: input functionals
def h1(x,y_bar,mu_x,mu_y,rho_xy):
    return (y_bar - mu_y)/sigma_y/np.sqrt(1.0-rho_xy**2) - rho_xy*(x-mu_x)/sigma_x/np.sqrt(1.0-rho_xy**2)
def h2(x,i,sigma_y,rho_xy):
    return h1(x,y_bar,mu_x,mu_y,rho_xy) + B(alpha2,T,t[i])*sigma_y*np.sqrt(1.0-rho_xy**2)
def lamb(x,i,A_i,B_ai):
    return ci(i)*A_i*np.exp(-B_ai*x)
def kappa(x,A_i,B_bi):
    trm = mu_y - 0.5*(1.0-rho_xy**2)*sigma_y**2*B_bi + rho_xy*sigma_y*(x-mu_x)/sigma_x
    return (-1.0)*B_bi*trm

#%% optimization













