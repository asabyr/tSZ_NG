import numpy as np
import sys
sys.path.append('../forecast/')
from nsims_convergence_funcs import *
from constraints import Constraints
from scipy.interpolate import *

#some helper functions to check and make the plots
#for convergence wrt to mean & covariance

#mean convergence: use covariance at Nfid but mean for some specified Nsim
FID_COSMO='Oc_0.264_s8_0.811'
CONSTRAINTS_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8"
OUT_DIR="/scratch/07833/tg871330/tSZ_NG/data/"

def lims_nsims_mean(prefix,
                    stat,
                    nsims=np.arange(500,5500, 500), some_ind=np.array([])):
    
    #number of sims
    nsims=np.concatenate((nsims, np.array([5112])))

    #save full likelihood surface for plots
    Ls_nsims_stat=np.zeros((1000,1000, len(nsims)))
    
    #save limits and areas
    lims_nsims_stat=np.zeros(len(nsims))
    areas_nsims_stat=np.zeros(len(nsims))
    
    
    for i,n in enumerate(nsims):
        stat_results_N=Constraints(prefix=f'{prefix}_{n}',stat=stat, 
                            stat_dir=f'{CONSTRAINTS_DIR}/{stat}/{stat}_nsims/', 
                            n_grid_points=1000, 
                        interpolator=CloughTocher2DInterpolator, 
                        fid_Nsims=34560, test_interpolator=False,
                    fid_covariance_file=f'{CONSTRAINTS_DIR}/{stat}/{prefix}_{stat}_{FID_COSMO}.npy', some_ind=some_ind)

        stat_results_N.likelihood_grid()
        lims_stat_N, params_stat_N=stat_results_N.compute_contours()


        lims_nsims_stat[i]=lims_stat_N[0]
        Ls_nsims_stat[:,:,i]=stat_results_N.Ls
        areas_nsims_stat[i]=stat_results_N.contour_area_1sigma
    
    nsims_dict={}
    nsims_dict['nsims']=nsims
    nsims_dict['lims']=lims_nsims_stat
    nsims_dict['Ls']=Ls_nsims_stat
    nsims_dict['areas']=areas_nsims_stat

    np.save(f"{OUT_DIR}/{stat}_nsims", nsims_dict)
    assert len(stat_results_N.fid_value)==len(some_ind)

    return nsims, lims_nsims_stat, Ls_nsims_stat, areas_nsims_stat




def lims_nsims_cov(prefix,
                    stat,
                    nsims=np.arange(3456,34560,3456), some_ind=np.array([])):
    
    #number of sims
    nsims=np.concatenate((nsims, np.array([34560])))

    #save full likelihood surface for plots
    Ls_nsims_stat=np.zeros((1000,1000, len(nsims)))
    
    #save limits and areas
    lims_nsims_stat=np.zeros(len(nsims))
    areas_nsims_stat=np.zeros(len(nsims))
    
    
    for i,n in enumerate(nsims):
        stat_results_N=Constraints(prefix=f'{prefix}',stat=stat, 
                            stat_dir=f'{CONSTRAINTS_DIR}/{stat}/', 
                            n_grid_points=1000, 
                        interpolator=CloughTocher2DInterpolator, 
                        fid_Nsims=n, test_interpolator=False,
                    fid_covariance_file=f'{CONSTRAINTS_DIR}/{stat}/Nsims_{n}_{prefix}_{stat}_{FID_COSMO}.npy', some_ind=some_ind)

        stat_results_N.likelihood_grid()
        lims_stat_N, params_stat_N=stat_results_N.compute_contours()


        lims_nsims_stat[i]=lims_stat_N[0]
        Ls_nsims_stat[:,:,i]=stat_results_N.Ls
        areas_nsims_stat[i]=stat_results_N.contour_area_1sigma
    
    nsims_dict={}
    nsims_dict['nsims']=nsims
    nsims_dict['lims']=lims_nsims_stat
    nsims_dict['Ls']=Ls_nsims_stat
    nsims_dict['areas']=areas_nsims_stat

    np.save(f"{OUT_DIR}/{stat}_nsims_cov", nsims_dict)
    assert len(stat_results_N.fid_value)==len(some_ind)

    return nsims, lims_nsims_stat, Ls_nsims_stat, areas_nsims_stat
