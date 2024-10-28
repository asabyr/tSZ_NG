#############################################
# some helpful functions for various files
# (there's definitely a better way to do it,
# but this works)
#############################################


#############################################
# ini-related files 
#############################################
import numpy as np
import glob


#for hmpdf cosmology ini files (i.e. for CLASS)
def write_ini(fname, Oc, s8):
    ini_file=open(fname, "w")
    ini_file.write("""#planck 2018: https://arxiv.org/pdf/1807.06209.pdf
#from abstract + TT,TE,EE+lowE+lensing results
# vary Omega_cdm and sigma8 (keep Omega_b the same)
h = 0.674
Omega_b = 0.0493\n""")
    ini_file.write(f"Omega_cdm = {Oc}\n")
    ini_file.write(f"sigma8 = {s8}\n")
    ini_file.write("""N_ur = 3.046
n_s = 0.965
tau_reio=0.054
N_ncdm=0
m_ncdm=0

#verbose settings
input_verbose = 0
background_verbose = 0
thermodynamics_verbose = 0
perturbations_verbose = 0
transfer_verbose = 0
primordial_verbose = 0
spectra_verbose = 0
nonlinear_verbose = 0
lensing_verbose = 0
output_verbose = 0

# Pk
output = mPk
P_k_max_h/Mpc = 10.0\n""")

#for map analysis inis
def write_map_ini(fname, prefix, stat, cosmo, extra_args, n_cores=int(1), 
                    base_filter='', which_noise='', wiener_filter='', 
                    noise_level=0.0, FWHM_arcmin=0, noise_seed=1):
    ini_file=open(fname, "w")
    ini_file.write(f"""[General]
prefix={prefix}
stat={stat}
cosmo={cosmo}
n_cores={n_cores}

[Maps]
trim_major=5,6
n_small_maps=36
trim_minor=50
map_dtype=float
resolution [arcmin]=0.1
save_small=1
one_side_trim=False
smooth=1.4
apodize_pixel=5
apodize_type=cos_pixell
subtract_mean=True
""")
    if len(base_filter)!=0:
        ini_file.write(f"base_filter={base_filter}\n")
    if len(wiener_filter)!=0:
        ini_file.write(f"wiener_filter={wiener_filter}\n")
        
    if FWHM_arcmin>0:
        ini_file.write(f"smooth_tot={FWHM_arcmin}\n")
    if stat=='cl':
        ini_file.write("[Stat]\n")
        ini_file.write(f"""ell_edges={extra_args['ell_edges']}
delta_ell={extra_args['delta_ell']}
""")
    elif stat=='peaks':
        ini_file.write("[Stat]\n")
        ini_file.write(f"""peak_heights={extra_args['peak_heights']}
sigma={extra_args['sigma']}
""")
    elif stat=='minima':
        ini_file.write("[Stat]\n")
        ini_file.write(f"""minima_heights={extra_args['minima_heights']}
sigma={extra_args['sigma']}
""")
    elif stat=='MF':
        ini_file.write("[Stat]\n")
        ini_file.write(f"""MF_linspace={extra_args['MF_linspace']}
sigma={extra_args['sigma']}
""")
    if len(which_noise)!=0:
        ini_file.write("[Noise]\n")
        ini_file.write(f"which_noise={which_noise}\n")
        if which_noise=='flat':
            ini_file.write(f"noise_level_muK_arcmin={noise_level}\n")
        if noise_seed!=1:
            ini_file.write(f"seed=0")
                
#############################################
#           for job-related files           
#############################################


#for task files
def write_task_file(fname,stat,cosmo,chunks,code_dir):

    tasks=open(fname,"w")
    for c in range(len(cosmo)):
        for i in range(chunks):
            tasks.write(f"python {code_dir}/calc_stats.py "+stat+"_"+cosmo[c]+".ini "+str(i)+"\n")
    tasks.close()

#for launcher files
def write_launcher_file(job_name, job_dir, cores_per_task):
    #launcher .py file
    pyfile=open(job_dir+"launcher_py/"+job_name+".py","w")
    pyfile.write(f"""#!/usr/bin/env python
import pylauncher as launcher
job_files_dir="{job_dir}"
launcher.ClassicLauncher(job_files_dir+"{job_name}.tasks",cores={cores_per_task},debug="job")""")

#for job files
def write_pbs_file(job_name, job_dir, nodes, tot_cores, time, stat):
    
    jobfile=open(job_dir+job_name+".pbs","w")
    jobfile.write("""#!/bin/sh
#SBATCH -A TG-AST140041
#SBATCH -J """+job_name+"\n")
    jobfile.write(f"""#SBATCH -N {nodes}
#SBATCH -n """+str(tot_cores)+"\n")
    jobfile.write("""#SBATCH -p skx
#SBATCH -t """+time+"\n")
    jobfile.write(f"#SBATCH -o /scratch/07833/tg871330/tSZ_NG_ini_jobs/job_files_suite/{stat}/output_files/launcher_"""+job_name+".o%j\n""")
    jobfile.write("""#SBATCH --mail-user=a.sabyr@columbia.edu
#SBATCH --mail-type=all
export PYTHONPATH=/scratch/07833/tg871330/software_scratch/pylauncher/src/pylauncher:${PYTHONPATH}\n""")
    jobfile.write(f"""source /home1/07833/tg871330/miniconda3/bin/activate tSZ
python launcher_py/{job_name}.py\n""")
    jobfile.close()

#############################################
# directory functions 
#############################################

def get_cosmologies(ini_path):
    #cosmologies
    all_cosmo_inis=np.array(glob.glob(ini_path+"*ini"))

    all_cosmo_names=[]
    for cosmo_path in all_cosmo_inis:
        clean_cosmo=cosmo_path.replace(ini_path, "").replace(".ini","")
        all_cosmo_names.append(clean_cosmo)
    return np.array(all_cosmo_names)
