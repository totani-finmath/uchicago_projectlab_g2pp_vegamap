#%% --------------------------------------------------
# import packages
# ----------------------------------------------------
import numpy as np
import pandas as pd
import scipy.integrate as itg
from scipy.stats import norm
from matplotlib import pyplot as plt

#%% --------------------------------------------------
# constants
# ----------------------------------------------------
upr_itg = 100.0
lwr_itg = -100.0

#%% --------------------------------------------------
# contracts
# ----------------------------------------------------
t_mat = 2.0
t_tnr = 5.0
X = 0.005
omega = 1
num_grid = int(t_tnr * 2) + 1
ts = np.linspace(t_mat, t_mat+t_tnr, num=num_grid)
tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
# aggregate
contract = {'grids'  : ts,
            'periods': tau,
            'strike' : X,
            'side'   : omega}

#%% --------------------------------------------------
# sample market data
# ----------------------------------------------------
# bond
r_flat = 0.01
# assumed market pbservable
def PM(t,T,r=r_flat):
    return np.exp(-r*(T-t))
# assumed G2++ calibrated
def P(t,T,r=r_flat):
    return np.exp(-r*(T-t))


#%% --------------------------------------------------
# parameters initial
# ----------------------------------------------------
alpha1 = 0.01
alpha2 = 0.02
sigma1 = 0.1
sigma2 = 0.2
rho    = 0.1
y_bar  = 0.001 # dammy parameter for constraint
# aggregate
params = [alpha1, alpha2, sigma1, sigma2, rho, y_bar]

#%% --------------------------------------------------
# functions: convert parameters
# ----------------------------------------------------
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
def sigma_xy_g2pp(t0,alpha,sigma):
    return sigma * np.sqrt((1.0-np.exp(-2*alpha*t0))*0.5/alpha)
def rho_xy_g2pp(t0,alpha1,alpha2,sigma1,sigma2,rho):
    return rho*sigma1*sigma2/(alpha1 + alpha2)/sigma1/sigma2 * (1.0 - np.exp(-(alpha1+alpha2)*t0))
def ci_g2pp(i,X,tau): # 1<=i<=n
    if i is not len(tau):
        return X*tau[i-1]
    else:
        return 1.0 + X*tau[i-1]

#%% --------------------------------------------------
# functions: bond pricings
# ----------------------------------------------------
def V_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho):
    f1 = lambda x: (1.0-np.exp(-alpha1*(T-x)))**2
    trm1 = sigma1**2/alpha1**2 * itg.quad(f1,t,T)[0] # discard error
    f2 = lambda x: (1.0-np.exp(-alpha2*(T-x)))**2
    trm2 = sigma2**2/alpha2**2 * itg.quad(f2,t,T)[0] # discard error
    f3 = lambda x: (1.0-np.exp(-alpha1*(T-x))) * (1.0-np.exp(-alpha2*(T-x)))
    trm3 = 2*sigma1*sigma2/alpha1/alpha2 * itg.quad(f3,t,T)[0] # discard error
    return trm1 + trm2 + trm3
def A_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho):
    return PM(0,T)/PM(0,t)*np.exp(0.5*(V_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho)\
                                    - V_g2pp(0,T,alpha1,alpha2,sigma1,sigma2,rho)\
                                    + V_g2pp(0,t,alpha1,alpha2,sigma1,sigma2,rho)))
def B_g2pp(z,t,T):
    return (1.0-np.exp(-z*(T-t)))/z

#%% --------------------------------------------------
# functions: input functionals
# ----------------------------------------------------
def h1_g2pp(x,y_bar,mu_x,mu_y,sigma_x,sigma_y,rho_xy):
    return (y_bar - mu_y)/sigma_y/np.sqrt(1.0-rho_xy**2) - rho_xy*(x-mu_x)/sigma_x/np.sqrt(1.0-rho_xy**2)
def h2_g2pp(sigma_y,rho_xy,h1,B_bi):
    return h1 + B_bi*sigma_y*np.sqrt(1.0-rho_xy**2)
def lamb_g2pp(x,i,A_i,B_ai,X,tau):
    return ci_g2pp(i,X,tau)*A_i*np.exp(-B_ai*x)
def kappa_g2pp(x,A_i,B_bi,mu_x,mu_y,sigma_x,sigma_y,rho_xy):
    trm = mu_y - 0.5*(1.0-rho_xy**2)*sigma_y**2*B_bi + rho_xy*sigma_y*(x-mu_x)/sigma_x
    return (-1.0)*B_bi*trm

#%% --------------------------------------------------
# optimization
# ----------------------------------------------------
def swpn_integrand_g2pp(x,params,contract):
    # unpack parameters
    alpha1, alpha2 = params[0], params[1]
    sigma1, sigma2 = params[2], params[3]
    rho, y_bar     = params[4], params[5]
    ts  = contract['grids']
    tau = contract['periods']
    K   = contract['strike']
    omega = contract['side']
    num_grid = len(ts)
    # preparation #1
    mu_x    = -M_x_g2pp(0,ts[0],ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    mu_y    = -M_y_g2pp(0,ts[0],ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    sigma_x = sigma_xy_g2pp(ts[0],alpha1,sigma1)
    sigma_y = sigma_xy_g2pp(ts[0],alpha2,sigma2)
    rho_xy  = rho_xy_g2pp(ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    # preparation #2
    As = [A_g2pp(ts[0],ts[i+1],alpha1,alpha2,sigma1,sigma2,rho) for i in range(num_grid-1)]
    B1s = [B_g2pp(alpha1,ts[0],ts[i+1]) for i in range(num_grid-1)]
    B2s = [B_g2pp(alpha2,ts[0],ts[i+1]) for i in range(num_grid-1)]
    # preparation #3
    h1  = h1_g2pp(x,y_bar,mu_x,mu_y,sigma_x,sigma_y,rho_xy)
    h2s = [h2_g2pp(sigma_y,rho_xy,h1,B2s[i]) for i in range(num_grid-1)]
    lambs  = [lamb_g2pp(x,i+1,As[i],B1s[i],K,tau) for i in range(num_grid-1)]
    kappas = [kappa_g2pp(x,As[i],B2s[i],mu_x,mu_y,sigma_x,sigma_y,rho_xy) for i in range(num_grid-1)]
    # integrand
    trm1 = np.exp(-0.5*(((x-mu_y)/sigma_x)**2)) / sigma_x/np.sqrt(2.0*np.pi)
    trm2 = norm.cdf(-omega*h1)
    trm3 = np.sum([lambs[i]*np.exp(kappas[i])*norm.cdf(-omega*h2s[i]) for i in range(num_grid-1)])
    return trm1 * (trm2 - trm3)

def swpn_price(params,contract):
    ts = contract['grids']
    omega = contract['side']
    integral = itg.quad(swpn_integrand_g2pp,lwr_itg,upr_itg,args=(params,contract))[0] # discard error
    return integral * omega * P(0, ts[0])


#%% --------------------------------------------------
# workspace
# ----------------------------------------------------
test = swpn_price(params,contract)
print(test)



#%% --------------------------------------------------
# test
# ----------------------------------------------------
xs = np.linspace(-10.0, 10.0, num=1000)
ys = np.array([swpn_integrand_g2pp(x,params,contract) for x in xs])
plt.plot(xs,ys)

#%% --------------------------------------------------
# test
# ----------------------------------------------------

