import os
import glob
import numpy as np
from read_combined_stats import *
import sys


############################################################
# The script computes the averaged statistic across the suite.
# It takes user-specified in-line arguments and outputs *npy files
# with averaged quantities
# 
# INPUT:
# prefix: array or singular prefix for the statistics to read (e.g. 77ell_cl,9bins_peaks etc.)
# tot_dim: the total dimensions of the vector (e.g. 60 for MFs etc.)
# which_stats: string that contains the statistics to be computed. 
# It has to either equal or be a substring of the following: 
# cl_peaks_MF_V0_V1_V2_minima_moments_sigma0_sigma1_S0_S1_S2_K0_K1_K2_K3
# out_prefix: prefix to the output averaged files.
# prefix_dir: set to 1 if the statistics are found in directories $PREFIX_stat$ rather than $stat$ 
# (this is just based on how the code changed over time, should always be set to 1 for the current version)
# fid_sims: how many sims to use for fiducial. 
# Choices are 5,17,30,50 corresponding to Nsims=5184,17280,34560,51840
# (optional)
# n_sims_arr: for testing convergence with respect to sims, 
# you can supply n_min, n_max, dn to input into np.arange(n_min, n_max, dn). 
# Then the averaging will be done for each. Note this number is only used for cosmologies 
# other than the fiducial.
#
# OUTPUT:
# *npy files with averaged statistics/covariances in $WHICH_STATS$ directory
############################################################


#user input
prefix=np.asarray(sys.argv[1].split(','), dtype="<U200")
tot_dim=int(sys.argv[2])
which_stats=sys.argv[3]
out_prefix=sys.argv[4]
prefix_dir=int(sys.argv[5])
fid_sims=int(sys.argv[6])

exclude_ind=np.array([1,-1])
if len(sys.argv)==8:
    nsims_arr=np.asarray(sys.argv[7].split(','),dtype=int)
    assert len(nsims_arr)==3, "Need to specify number of sims as following: nsim_min,nsim_max,dnsim"
    nsims_conv=np.arange(nsims_arr[0],nsims_arr[1],nsims_arr[2])
else:
    nsims_conv=np.array([])


#main directories
SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/"
AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/"
COSMO_INI_PATH='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/ini_files/'

#cosmologies
all_cosmo_inis=np.array(glob.glob(COSMO_INI_PATH+"*ini"))
all_cosmo_names=[]
for cosmo_path in all_cosmo_inis:
    clean_cosmo=cosmo_path.replace(COSMO_INI_PATH, "").replace(".ini","")
    all_cosmo_names.append(clean_cosmo)
all_cosmo=np.array(all_cosmo_names)

#for these I need summary statistics for each map, so averaging only for fiducial
if "M15" in out_prefix:
    print("computing averages for fisher forecast only")
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/0pt5_M15/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/0pt5_M15/"
    all_cosmo=np.array(['Oc_0.264_s8_0.811']) 
if "M13" in out_prefix:
    print("computing averages for fisher forecast only")
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M13/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M13/"
    all_cosmo=np.array(['Oc_0.264_s8_0.811'])  
if "M12" in out_prefix:
    print("computing averages for fisher forecast only")
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M12/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M12/"
    all_cosmo=np.array(['Oc_0.264_s8_0.811'])

#make directory to hold avg stats
if not os.path.exists(os.path.join(AVG_DIR,which_stats)):
    os.mkdir(os.path.join(AVG_DIR,which_stats)) 

#which stats are we actually computing
def check_stat_files(selected_stats, tot_sims, root_dir,stat_prefix):
    possible_stats=np.array(['cl','peaks','MF','minima','moments'])
    match=np.isin(selected_stats.split("_"), possible_stats)
    all_stats=np.array(selected_stats.split("_"))[match]
    
    enough_files=True
    for i in range(len(all_stats)):
        if prefix_dir==1:
            all_stat_files=glob.glob(os.path.join(root_dir,stat_prefix[i], stat_prefix[i]+"*"))
            # print(os.path.join(root_dir,stat_prefix[i], stat_prefix[i]+"*"))
        else:
            all_stat_files=glob.glob(os.path.join(root_dir,all_stats[i],stat_prefix[i]+"*"))
        # print(len(all_stat_files))
        if len(all_stat_files)<tot_sims:
            enough_files=False
            return enough_files 
    return enough_files

#loop through cosmologies
for j in range(len(all_cosmo)):
    print(all_cosmo[j])    
    #specify number of sims and check that the number of files actually match
    if all_cosmo[j]=='Oc_0.264_s8_0.811':
        if fid_sims==30:
            Nsims=34560
        elif fid_sims==17:
            Nsims=17280
        elif fid_sims==50:
            Nsims=51840
        elif fid_sims==5:
            Nsims=5112 
    else:
        Nsims=5112
     
    enough_stat_files=check_stat_files(selected_stats=which_stats, tot_sims=Nsims,
                        root_dir=f'{SIM_DIR}{all_cosmo[j]}/',stat_prefix=prefix)
     
    if enough_stat_files==False:
        print("not enough files")
        print(all_cosmo[j])
        continue
    
    
    #check if you've already computed this average
    avg_stat_file=os.path.join(AVG_DIR, which_stats, out_prefix+"_"+which_stats+"_"+all_cosmo[j])
    # print(avg_stat_file)
    # print(os.path.exists(avg_stat_file+".npy")) 
    if os.path.exists(avg_stat_file+".npy")==True and len(nsims_conv)<1:
        continue
     
    try: 
        read_cosmo=ReadCombinedStats(prefix,f'{SIM_DIR}{all_cosmo[j]}', nfiles=Nsims,
                        tot_dim=tot_dim, which_stats=which_stats, out_prefix=out_prefix, 
                        out_dir_prefix=prefix_dir, exclude_ind=exclude_ind)
        avg_cosmo=read_cosmo.process_dir()
    
    
        print(f"will save avg stat to {avg_stat_file}")
        np.save(avg_stat_file, avg_cosmo)
    except:
        pass
    
    for k,nsims in enumerate(nsims_conv):
        avg_stat_file_nsims=os.path.join(AVG_DIR, which_stats, out_prefix+"_"+str(int(nsims))+"_"+which_stats+"_"+all_cosmo[j])
        
        if os.path.exists(avg_stat_file_nsims)==True:
            continue
        read_cosmo=ReadCombinedStats(prefix,f'{SIM_DIR}{all_cosmo[j]}/', nfiles=nsims,
                        tot_dim=tot_dim, which_stats=which_stats, out_prefix=out_prefix+"_"+str(int(nsims)),
                        out_dir_prefix=prefix_dir, exclude_ind=exclude_ind)
        avg_cosmo=read_cosmo.process_dir()
        print(f"will save avg stat to {avg_stat_file_nsims}")
        np.save(avg_stat_file_nsims, avg_cosmo)
        


