import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
#some functions partially adapted from LensTools

def delta_x(map_stat,fid_map_stat): 
    #find difference between two stat vectors
    diff=np.zeros((len(map_stat),1))
    diff[:,0]=map_stat-fid_map_stat
    return diff

def loglike(cov, delta_x, n_stat, N_sims):
    #compute log likelihood
    step1=np.linalg.solve(cov,delta_x)
    step2=np.dot(delta_x.T,step1*((N_sims-n_stat-2.0)/(N_sims-1.0)))
    return step2
 
def get_centroid(L, params):
    int_1=np.trapz(np.multiply(params,L),params)
    int_2=np.trapz(L,params)
    centroid=int_1/int_2
    return centroid

def get_like_level(Ls, fracs=np.array([0.68,0.95])):
    
    #flatten
    Ls_flat_orig=Ls.flatten()
    #integrate
    tot_int_Ls=np.sum(Ls_flat_orig)
   
    #put in descending order
    ind_sort=np.argsort(Ls_flat_orig)
    Ls_flat=Ls_flat_orig[ind_sort][::-1]
    
    #compute 1 and 2 sigma contour levels
    lims=np.zeros(len(fracs)) #array to store likelihood levels
    k=np.zeros(len(fracs), dtype=int) #array to store indices

    for i, frac in enumerate(fracs):
        
        diff=frac-np.cumsum(Ls_flat/tot_int_Ls)
        ind_negative=np.where(diff<0)[0] #find first ind where the sign switches
        ind_closest=np.where(diff==diff[ind_negative[0]])[0] #up to this index the values fall within the likelihood level
        
        k[i]=ind_closest
        lims[i]=Ls_flat[ind_closest]
        
    return lims, k, ind_sort
    
def get_marginal_param_limits(Ls, lims, params):
    
    param_1=np.zeros(len(lims))
    param_2=np.zeros(len(lims))

    for i in range(len(lims)):
        
        #get the first closest index
        Ls_diff=np.abs(Ls-lims[i])
        min_ind_1=np.argmin(Ls_diff)
        
        #get the second closest index
        Ls_diff_2=np.delete(Ls_diff,min_ind_1)
        min_ind_2=np.argmin(Ls_diff_2)
        
        #get the two parameters associated with that
        param_1_value=params[min_ind_1]
        param_2_value=np.delete(params, min_ind_1)[min_ind_2]
        
        #place in correct order, low then high
        if param_1_value<param_2_value:
            param_1[i]=param_1_value
            param_2[i]=param_2_value
        elif param_1_value>param_2_value:
            param_1[i]=param_2_value
            param_2[i]=param_1_value

    #double check the params are returned in the right order
    if (param_1<param_2).all():
        return param_1, param_2
    else:
        sys.exit("check marginalized parameters")

def plot_marginal(params, Ls, lims,
                  param_low, param_high, param_centroid, save_plot=''):
    
    dtheta_min=param_centroid-param_low[0]
    dtheta_max=param_high[0]-param_centroid
    
    plt.figure()
    #level
    plt.scatter(params, Ls, color='grey')
    plt.hlines(lims[0],xmin=np.min(params), xmax=np.max(params), color='green')
    plt.hlines(lims[1],xmin=np.min(params), xmax=np.max(params), color='green')
    #1 sigma
    plt.vlines(param_low[0],ymin=0, ymax=np.max(Ls), ls=':', label='1 sigma', color='black')
    plt.vlines(param_high[0],ymin=0, ymax=np.max(Ls), ls='-.', color='black')

    #2 sigma
    plt.vlines(param_low[1],ymin=0, ymax=np.max(Ls), ls=':', label='2 sigma', color='blue')
    plt.vlines(param_high[1],ymin=0, ymax=np.max(Ls), ls='-.', color='blue')
    
    #mean
    plt.vlines(param_centroid,ymin=0, ymax=np.max(Ls), ls='--', color='red')
    
    plt.xlim([param_low[1]-dtheta_min*2.0,param_high[1]+dtheta_max*2.0])
    plt.legend()
    if len(save_plot)>0:
        plt.savefig(save_plot+".pdf")

# def alpha_param(alpha, x_fid, grid_x, grid_y, Ls):
    
#     Sigma=(grid_y*(grid_x/x_fid)**alpha).flatten()
#     L_Sigma=Ls.flatten()/np.sum(Ls)
    
#     E=np.trapz(np.multiply(Sigma,L_Sigma), Sigma)
#     V=np.trapz(np.multiply((Sigma-E)**2.0, L_Sigma),Sigma)

#     return np.sqrt(V)/E

# def find_alpha(alpha_guess=0.2):
#     alpha_param=minimize(alpha_param, alpha_guess, args=())
#     return alpha_param.x 
