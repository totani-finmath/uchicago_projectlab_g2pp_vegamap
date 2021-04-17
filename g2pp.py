#%% --------------------------------------------------
# import packages
# ----------------------------------------------------
import numpy as np
import pandas as pd
from scipy import integrate as itg
from scipy import optimize as opt
from scipy.stats import norm
from matplotlib import pyplot as plt
import time as tm
import pickle as pk

#%% --------------------------------------------------
# constants
# ----------------------------------------------------
upr_itg = np.inf
lwr_itg = -np.inf
upr_itg = 100.0
lwr_itg = -100.0
y_bar_init = 0.001
lst_bounds = [(1.0e-5,5.0), (1.0e-5,5.0), (1.0e-5,5.0), (1.0e-5,5.0), (-1.0,1.0)]

#%% --------------------------------------------------
# contracts sample
# ----------------------------------------------------
t_mat = 2.0
t_tnr = 5.0
X = 0.01
omega = 1
def build_contract(t_mat,t_tnr,strike,omega):
    num_grid = int(t_tnr * 4) + 1
    ts = np.linspace(t_mat, t_mat+t_tnr, num=num_grid)
    contract = {'grids':ts,'strike':X,'side': omega}
    return contract
contract = build_contract(t_mat,t_tnr,X,omega)

#%% --------------------------------------------------
# sample market data
# ----------------------------------------------------
# bond
r_flat = 0.01
# assumed market observable
def PM(t,T,r=r_flat):
    return np.exp(-r*(T-t))
# assumed G2++ calibrated
def P(t,T,r=r_flat):
    return np.exp(-r*(T-t))
# swap rate
def swap_rate(ts,t=0.0):
    t_stt = ts[0]
    t_end = ts[-1]
    taus = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    denom = np.sum([taus[i]*P(t,ts[i+1]) for i in range(len(taus))])
    numer = P(t,t_stt) - P(t,t_end)
    return numer/denom

#%% --------------------------------------------------
# parameters initial
# ----------------------------------------------------
alpha1_init = 1.0
alpha2_init = 0.1
sigma1_init = 0.2
sigma2_init = 0.01
rho_init    = -0.8
# aggregate
params_init = [alpha1_init, alpha2_init, sigma1_init, sigma2_init, rho_init]

#%% ##################################################
# G2++ implementation
# ####################################################
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
def rho_xy_g2pp(t0,alpha1,alpha2,sigma1,sigma2,rho,sigma_x,sigma_y):
    return rho*sigma1*sigma2/(alpha1 + alpha2)/sigma_x/sigma_y * (1.0 - np.exp(-(alpha1+alpha2)*t0))
def ci_g2pp(i,X,tau): # 1<=i<=n
    if i is not len(tau)-1:
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
def lamb_g2pp(x,i,A_i,B_ai,ci):
    return ci*A_i*np.exp(-B_ai*x)
def kappa_g2pp(x,A_i,B_bi,mu_x,mu_y,sigma_x,sigma_y,rho_xy):
    trm = mu_y - 0.5*(1.0-rho_xy**2)*sigma_y**2*B_bi + rho_xy*sigma_y*(x-mu_x)/sigma_x
    return (-1.0)*B_bi*trm

#%% --------------------------------------------------
# optimization
# ----------------------------------------------------
# constraint of swaption formula
def constraint_ybar(y,x,cs,As,B1s,B2s):
    trms = [cs[i]*As[i]*np.exp(-B1s[i]*x-B2s[i]*y) for i in range(len(cs))]
    return np.sum(trms) - 1.0
# swaption formula integrand
def swpn_integrand_g2pp(x,params,contract):
    # unpack parameters
    alpha1, alpha2 = params[0], params[1]
    sigma1, sigma2 = params[2], params[3]
    rho = params[4]
    ts  = contract['grids']
    tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    K   = contract['strike']
    omega = contract['side']
    num_grid = len(ts)
    # preparation #1
    mu_x    = -M_x_g2pp(0,ts[0],ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    mu_y    = -M_y_g2pp(0,ts[0],ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    sigma_x = sigma_xy_g2pp(ts[0],alpha1,sigma1)
    sigma_y = sigma_xy_g2pp(ts[0],alpha2,sigma2)
    rho_xy  = rho_xy_g2pp(ts[0],alpha1,alpha2,sigma1,sigma2,rho,sigma_x,sigma_y)
    # preparation #2
    As = [A_g2pp(ts[0],ts[i+1],alpha1,alpha2,sigma1,sigma2,rho) for i in range(num_grid-1)]
    B1s = [B_g2pp(alpha1,ts[0],ts[i+1]) for i in range(num_grid-1)]
    B2s = [B_g2pp(alpha2,ts[0],ts[i+1]) for i in range(num_grid-1)]
    cs  = [ci_g2pp(i,K,tau) for i in range(num_grid-1)]
    # root finding
    y_bar = opt.root(constraint_ybar,y_bar_init,\
                args=(x,cs,As,B1s,B2s),method='hybr').x[0]
    # preparation #3
    h1  = h1_g2pp(x,y_bar,mu_x,mu_y,sigma_x,sigma_y,rho_xy)
    h2s = [h2_g2pp(sigma_y,rho_xy,h1,B2s[i]) for i in range(num_grid-1)]
    lambs  = [lamb_g2pp(x,i+1,As[i],B1s[i],cs[i]) for i in range(num_grid-1)]
    kappas = [kappa_g2pp(x,As[i],B2s[i],mu_x,mu_y,sigma_x,sigma_y,rho_xy) for i in range(num_grid-1)]
    # integrand
    trm1 = np.exp(-0.5*(((x-mu_y)/sigma_x)**2)) / sigma_x/np.sqrt(2.0*np.pi)
    trm2 = norm.cdf(-omega*h1)
    trm3 = np.sum([lambs[i]*np.exp(kappas[i])*norm.cdf(-omega*h2s[i]) for i in range(num_grid-1)])
    return trm1 * (trm2 - trm3)

def swpn_price_g2pp(params,contract):
    ts = contract['grids']
    omega = contract['side']
    integral = itg.quad(swpn_integrand_g2pp,lwr_itg,upr_itg,args=(params,contract))[0] # discard error
    return integral * omega * P(0, ts[0])


#%% --------------------------------------------------
# test: pricing
# ----------------------------------------------------
# test setting
t_mat_test = 1.0
t_tnr_test = 5.0
omega_test = 1
# test preparation
num_grid_test = int(t_tnr_test * 4) + 1
ts_test = np.linspace(t_mat_test, t_mat_test+t_tnr_test, num=num_grid_test)
X_test = swap_rate(ts_test)
cont_test = build_contract(t_mat_test,t_tnr_test,X_test,omega_test)
# calculation
price_test = swpn_price_g2pp(params_init,cont_test)
print('test price',t_mat_test,'x',t_tnr_test,':',price_test)

#%% --------------------------------------------------
# test: plot
# ----------------------------------------------------
xs = np.linspace(-10.0, 10.0, num=100)
ys = np.array([swpn_integrand_g2pp(x,params_init,cont_test) for x in xs])
plt.plot(xs,ys)

#%% ##################################################
# optimization part
# ####################################################
#%% --------------------------------------------------
# iv/price conversions
# ----------------------------------------------------
# black model
def black(S,K,T,sigma,omega):
    d1 = (np.log(S/K) + 0.5*sigma**2*T)/sigma/np.sqrt(T)
    d2 = (np.log(S/K) - 0.5*sigma**2*T)/sigma/np.sqrt(T)
    return omega * (S*norm.cdf(omega*d1) - K*norm.cdf(omega*d2))
# swaption by log-normal implied vol
def vol_to_price_black(S,K,sigma,omega,ts):
    tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    ann = np.sum([tau[i] * P(0.0, ts[i+1]) for i in range(len(tau))])
    return black(S,K,ts[0],sigma,omega) * ann
# bachelier model
def bachelier(S,K,T,sigma,omega):
    d1 = (S-K)/sigma/np.sqrt(T)
    return omega*(S-K)*norm.cdf(omega*d1) + sigma*np.sqrt(T)*norm.pdf(omega*d1)
# swaption by normal implied vol
def vol_to_price_normal(S,K,sigma,omega,ts):
    tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    ann = np.sum([tau[i] * P(0.0, ts[i+1]) for i in range(len(tau))])
    return bachelier(S,K,ts[0],sigma,omega) * ann

#%% --------------------------------------------------
# import sample data
# ----------------------------------------------------
filename = 'matrix_normal.xlsx'
df_vols = pd.read_excel(filename,sheet_name='vols_normal',index_col=0)
df_wgts = pd.read_excel(filename,sheet_name='weights_normal',index_col=0)

# generating data
lst_data = []
for idx in df_vols.index:
    for col in df_vols.columns:
        if df_wgts[col][idx] > 0:
            lst_data.append([idx,col,df_vols[col][idx]/10000])
# generate contract list
lst_contracts = []
for i in range(len(lst_data)):
    mem = lst_data[i]
    # this contract
    num_grid = int(mem[1] * 4) + 1
    ts = np.linspace(mem[0], mem[0]+mem[1], num=num_grid)
    swp_atm = swap_rate(ts)
    omega = 1
    # swaption price
    price = vol_to_price_normal(swp_atm,swp_atm,mem[2],omega,ts)
    contract = {'tenor'  : (mem[0],mem[1]),
                'grids'  : ts,
                'strike' : swp_atm,
                'side'   : omega,
                'price'  : price}
    lst_contracts.append(contract)


#%% --------------------------------------------------
# objective function
# ----------------------------------------------------
# objective function
def opt_objective_g2pp(params, lst_contracts):
    err_square = []
    for contract in lst_contracts:
        err = contract['price'] - swpn_price_g2pp(params,contract)
        err_square.append(err**2)
    return np.sqrt(np.mean(err_square))

#%% --------------------------------------------------
# test
# ----------------------------------------------------
#rerr = opt_objective_g2pp(params,lst_contracts)
for cont in lst_contracts:
    price = swpn_price_g2pp(params,cont)
    print(cont['tenor'])
    print(price)

#%% --------------------------------------------------
# calibration
# ----------------------------------------------------
# optimizer
is_calib = False
if is_calib:
    params_init = params.copy()
    time_stt = tm.time()
    #res = opt.minimize(opt_objective_g2pp,params_init,method='BFGS',args=(lst_contracts),bounds=lst_bounds)
    res = opt.minimize(opt_objective_g2pp,params_init,method='Powell',args=(lst_contracts),bounds=lst_bounds)
    time_end = tm.time()
    time_exe = time_end - time_stt
    print(time_exe/60.0,'[min]')
    pk.dump(res,open('res_calib_org.pk','wb'))
else:
    res = pk.load(open('res_calib_org.pk','rb'))

#%% --------------------------------------------------
# review results
# ----------------------------------------------------
'''
method='Powell'
upr_itg = np.inf
lwr_itg = -np.inf
params_init = [0.5, 2.5, 1.00, 0.1, 0.1]
508.2066081007322 [min]

direc: array([[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
       [ 0.01191397, -0.48691017, -0.04548408,  0.00980303, -0.06100452]])
     fun: 0.020514307912352633
 message: 'Optimization terminated successfully.'
    nfev: 268
     nit: 4
  status: 0
 success: True
       x: array([ 4.84108185,  4.137148  ,  1.17343152,  0.07240115, -0.00771107])
'''

params_new = [4.84108185,  4.137148  ,  1.17343152,  0.07240115, -0.00771107]
for cont in lst_contracts:
    price = swpn_price_g2pp(params_new,cont)
    print('Tenor >> ', cont['tenor'])
    print('G2++ price   :', price)
    print('Market price :', cont['price'])





# %%


