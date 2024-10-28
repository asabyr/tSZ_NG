import sys
sys.path.append('../forecast/')
sys.path.append('../convergence/')
from nsims_convergence_funcs import *
import matplotlib.pyplot as plt
from constraints import Constraints

###############################################################
#                                                             #
#      example script to compute convergence of constraints   #   
#      wrt to simulations used for the mean/covariance        #                                            
###############################################################

CONSTRAINTS_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8"

prefix=sys.argv[1]
stat=sys.argv[2]

if len(sys.argv)==4:
    print("picking indices")
    if "all" in prefix: # this is hard-coded for the specific set-up needed for the paper
        other_stats=np.arange(0,153,1) 
        moments_ind=np.asarray(sys.argv[3].split(','),dtype=int)
        some_ind=np.concatenate((other_stats,moments_ind))
    else:
        some_ind=np.asarray(sys.argv[3].split(','),dtype=int)
else:
    some_ind=np.array([])

#this will save files, see nsims_convergence_funcs.py
nsims, lims_nsims, Ls_nsims, areas_nsims=lims_nsims_mean(prefix=prefix,stat=stat, some_ind=some_ind)

################### PLOT ##############

results=Constraints(prefix=prefix,
                       stat=stat, 
                       stat_dir=f'{CONSTRAINTS_DIR}/{stat}/', 
                       n_grid_points=1000, 
                 interpolator=CloughTocher2DInterpolator, 
                       fid_Nsims=34560, test_interpolator=False, some_ind=some_ind)
results.likelihood_grid()
lims, params=results.compute_contours()

assert len(results.fid_value)==int(len(some_ind))

#plot
colormap = plt.cm.viridis
all_legends=[]
all_labels=[]

for i,n in enumerate(nsims):
    stat_plot=plt.contour(results.grid_X, results.grid_Y,  Ls_nsims[:,:,i],[lims_nsims[i]], 
                   color=colormap(i/len(nsims)), alpha=(i+2)/(len(nsims)+2))
    legend,_ = stat_plot.legend_elements()
    
    all_legends.append(legend[0])
    all_labels.append(f'{n}')

plt.legend(all_legends,
           all_labels,
           fontsize=15, 
            loc='upper right')
plt.ylim([0.8,0.82])
#plt.xlim([0.25,0.28])
plt.ylim([0.78,0.85])
plt.savefig(f"/scratch/07833/tg871330/tSZ_NG/figs/{stat}_conv_mean.pdf")
################### PLOT as a check ##############

#this will save files, see nsims_convergence_funcs.py
nsims_cov, lims_nsims_cov, Ls_nsims_cov, areas_nsims_cov=lims_nsims_cov(prefix=prefix,stat=stat, some_ind=some_ind)

################### PLOT as a check ##############
colormap = plt.cm.viridis
all_legends=[]
all_labels=[]

for i,n in enumerate(nsims_cov):
    stat_plot=plt.contour(results.grid_X, results.grid_Y,  Ls_nsims_cov[:,:,i],[lims_nsims_cov[i]],
                   color=colormap(i/len(nsims_cov)), alpha=(i+2)/(len(nsims_cov)+2))
    legend,_ = stat_plot.legend_elements()

    all_legends.append(legend[0])
    all_labels.append(f'{n}')

plt.legend(all_legends,
           all_labels,
           fontsize=15,
            loc='upper right')
plt.ylim([0.8,0.82])
#plt.xlim([0.25,0.28])
plt.ylim([0.78,0.85])
plt.savefig(f"/scratch/07833/tg871330/tSZ_NG/figs/{stat}_conv_cov.pdf")
################### PLOT ##############
