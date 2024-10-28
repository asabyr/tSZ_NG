import numpy as np
import glob
import os
import sys
sys.path.append('../suite_code/')
from help_funcs import *
import copy
###############################################################
#                                                             #
#       example script to prepare jobs/ini files to           #
#                  run various MFs set-ups                    #
###############################################################


########## user input for ini files ##########

MF_prefix=sys.argv[1]
MF_linspace=sys.argv[2]
batch=int(sys.argv[3]) #which ini files to use to determine all cosmologies in the suite

#noisy case
if len(sys.argv)>4:
    base_filter=sys.argv[4]
    which_noise=sys.argv[5]

    if which_noise!='baseline' and which_noise!='goal' and which_noise!='flat' and which_noise!='S4':
        which_noise=''
    
    stdev=float(sys.argv[6])
    wiener_filter=sys.argv[7]

    if wiener_filter.lower()=='none':
        wiener_filter=''
    
    noise_level=sys.argv[8] #white noise
    noise_seed=sys.argv[9] #should be set to 0 for random noise realizations

else:
    #noiseless case
    base_filter=''
    which_noise=''
    wiener_filter=''
    noise_level=0
    noise_seed=1 #not relevant for noiseless
    stdev=1.5 

#extra smoothing
if len(sys.argv)>10:
    FWHM_arcmin=float(sys.argv[10])
else:
    FWHM_arcmin=0

stat="MF"
MF_test=MF_prefix+"_"+"MF"

###############################################################
# define for which cosmologies to write the ini files
SIM_DIR='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/'
STAT_INI_PATH='/scratch/07833/tg871330/tSZ_NG_ini_jobs/suite_stat_ini_files/'

if batch==0:
    COSMO_INI_PATH='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/ini_files/'
    previous_cosmo=0
elif batch==1:
    COSMO_INI_PATH='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/ini_files/add_sims/'
    previous_cosmo=5
elif batch==2:
    COSMO_INI_PATH='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/ini_files/add_sims_ellipse/'
    previous_cosmo=6


all_cosmo=glob.glob(COSMO_INI_PATH+"*ini")
print(f"total cosmo:{len(all_cosmo)}")

#collect cosmologies
all_cosmo_names=[]
for cosmo_path in all_cosmo:
    clean_cosmo=cosmo_path.replace(COSMO_INI_PATH, "").replace(".ini","")
    all_cosmo_names.append(clean_cosmo)
all_cosmo_names=np.array(all_cosmo_names)

#make directories
for cosmo in all_cosmo_names:
    if not os.path.exists(os.path.join(SIM_DIR,cosmo,MF_test)):
        print(f"will make this directory:{os.path.join(SIM_DIR,cosmo,MF_test)}")
        os.mkdir(os.path.join(SIM_DIR,cosmo,MF_test))
#sys.exit(0)
##########################################################################
MF_dict={}
MF_dict['MF_linspace']=copy.deepcopy(MF_linspace)
MF_dict['sigma']=str(stdev)


MF_stat_ini_dir=os.path.join(STAT_INI_PATH, MF_test)
if not os.path.exists(MF_stat_ini_dir):
    os.mkdir(MF_stat_ini_dir)

for i in range(len(all_cosmo)):

    cosmo_ini=os.path.basename(os.path.normpath(all_cosmo[i]))
    if cosmo_ini=='Oc_0.264_s8_0.811.ini':
        n_cores=20
    else:
        n_cores=2
    stat_cosmo_ini=os.path.join(STAT_INI_PATH, MF_test, MF_test+"_"+cosmo_ini)
    write_map_ini(fname=stat_cosmo_ini, prefix=MF_prefix, stat="MF",
                  cosmo=cosmo_ini.replace(".ini", ""), extra_args=MF_dict, n_cores=n_cores, 
                    base_filter=base_filter, which_noise=which_noise, wiener_filter=wiener_filter,
                     noise_level=noise_level, FWHM_arcmin=FWHM_arcmin, noise_seed=noise_seed)

##########################################################################

JOB_DIR='/scratch/07833/tg871330/tSZ_NG_ini_jobs/job_files_suite/'
CODE_DIR='/scratch/07833/tg871330/tSZ_NG/code/general'

all_cosmo_with_fid=np.array(glob.glob(COSMO_INI_PATH+"/*ini"))
ind_fid=np.where(all_cosmo_with_fid==COSMO_INI_PATH+"Oc_0.264_s8_0.811.ini")
all_cosmo=np.delete(all_cosmo_with_fid,ind_fid)

cosmo_chunks=np.array_split(all_cosmo, np.arange(20,len(all_cosmo),20))

chunks=2
TIME="48:00:00"
CORES=9
NODES=8
TOT_CORES=CORES*len(cosmo_chunks[0])*chunks

#make relevant directories
if not os.path.exists(os.path.join(JOB_DIR,MF_test)):
    os.mkdir(os.path.join(JOB_DIR,MF_test))
if not os.path.exists(os.path.join(JOB_DIR,MF_test, 'launcher_py')):
    os.mkdir(os.path.join(JOB_DIR,MF_test, 'launcher_py'))

for j in range(len(cosmo_chunks)):

    #make list of cosmologies
    cosmo_list=[]
    for k in range(len(cosmo_chunks[j])):

        clean_cosmo=os.path.basename(os.path.normpath(cosmo_chunks[j][k])).replace('.ini','')
        cosmo_list.append(clean_cosmo)
        cosmo=np.array(cosmo_list)
    
    if j==len(cosmo_chunks)-1:
        TOT_CORES=CORES*len(cosmo_chunks[j])*chunks
        NODES=int(np.ceil(TOT_CORES/45)) 
    
    #write tasks, 20 cosmologies, 20 tasks per file 
    job_name=MF_test+"_"+str(j+previous_cosmo)
    task_fname=os.path.join(JOB_DIR, MF_test, job_name+".tasks")
    write_task_file(fname=task_fname,stat=MF_test,cosmo=cosmo,chunks=chunks,code_dir=CODE_DIR)
    write_launcher_file(job_name=job_name, job_dir=JOB_DIR+MF_test+"/",
                                cores_per_task=CORES)
    write_pbs_file(job_name=job_name, job_dir=JOB_DIR+MF_test+"/", nodes=NODES,
                            tot_cores=TOT_CORES, time=TIME, stat=MF_test)


