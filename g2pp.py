import numpy as np
import pandas as pd
import scipy.integrate as ing

#%% Constants
T = np.linspace(0.0, 5.0, num=11)
tau = [t2 - t1 for t1, t2 in zip(T[:-1], T[1:])]

#%% Parameters
alpha1 = 0.01
alpha2 = 0.02
sigma1 = 0.1
sigma2 = 0.2
rho = 0.1
X = 0.50

#%% functions
def sigma_x_g2pp(T):
    return sigma1 * np.sqrt((1.0-np.exp(-2*alpha1*T))*0.5/alpha1)
def sigma_y_g2pp(T):
    return sigma2 * np.sqrt((1.0-np.exp(-2*alpha2*T))*0.5/alpha2)
def rho_xy_g2pp(T):
    return rho*sigma1*sigma2/(alpha1 + alpha2)/sigma1/sigma2 * (1.0 - np.exp(-(alpha1+alpha2)*T))
def M_x_g2pp(s, t, T):
    trm1 = (sigma1**2/alpha1**2 + rho*(sigma1*sigma2)/alpha1/alpha2) * (1.0 - np.exp(-alpha1*(t-s)))
    trm2 = -1.0 * sigma1**2*0.5/alpha1**2 * (np.exp(-alpha1*(T-t)) - np.exp(-alpha1*(T+t-2*s)))
    trm3 = -1.0 * rho*sigma1*sigma2/alpha2/(alpha1+alpha2) * (np.exp(-alpha2*(T-t)) - np.exp(-alpha2*T-alpha1*t + (alpha1+alpha2)*s))
    return trm1 + trm2 + trm3
def M_y_g2pp(s, t, T):
    trm1 = (sigma2**2/alpha2**2 + rho*(sigma1*sigma2)/alpha1/alpha2) * (1.0 - np.exp(-alpha2*(t-s)))
    trm2 = -1.0 * sigma2**2*0.5/alpha2**2 * (np.exp(-alpha2*(T-t)) - np.exp(-alpha2*(T+t-2*s)))
    trm3 = -1.0 * rho*sigma1*sigma2/alpha1/(alpha1+alpha2) * (np.exp(-alpha1*(T-t)) - np.exp(-alpha1*T-alpha2*t + (alpha1+alpha2)*s))
    return trm1 + trm2 + trm3
def c_i(i):
    if i is not n:
        return X * tau(i)
    else:
        return 1.0 + X * tau(i)














