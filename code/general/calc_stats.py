import sys
import configparser
import numpy as np
import os
from analyze_maps import AnalyzeMaps 

MAP_DIR='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps'

#####################################################################
# This script computes statistics on maps given an ini file.
#####################################################################

############## define working directory ##############
this_dir=os.getcwd()
this_dir=os.path.dirname(os.path.abspath(__file__))
project_dir=this_dir.replace('/code','').replace('/general','').replace('/tSZ_NG','')

############## ini file ##############
ini_file=str(sys.argv[1])
n_arr=int(sys.argv[2]) #parallelization chunck

config=configparser.ConfigParser()

#specify where ini files are
if "fisher" in ini_file:
    stat_ini=ini_file[:np.char.find(ini_file,"Oc")][:-1]
    ini_file_dir=os.path.join(project_dir,'tSZ_NG_ini_jobs','fisher_stat_ini_files',stat_ini, ini_file)
elif "Oc" in ini_file:
    stat_ini=ini_file[:np.char.find(ini_file,"Oc")][:-1]
    ini_file_dir=os.path.join(project_dir,'tSZ_NG_ini_jobs','suite_stat_ini_files',stat_ini, ini_file)
else:
    ini_file_dir=os.path.join(project_dir,'tSZ_NG_ini_jobs', 'ini_files_num_conv', ini_file)


config.read(ini_file_dir)

#general settings
prefix=config['General']['prefix']
stat=config['General']['stat'] #string with stats that need to be computed "cl_peaks_MF"
cosmo=config['General']['cosmo']
if "Oc" in ini_file:
    if stat not in stat_ini:
        sys.exit("stats don't match")
if prefix.lower()=='none':
    prefix=''

#check
if cosmo not in ini_file.replace(".ini",''):
    sys.exit('ini file name and cosmology do not match')

if "Oc" in cosmo:
    data_dir=f"{MAP_DIR}/Oc_s8/"
else:
    data_dir=f"{MAP_DIR}/num_convergence/"

#add fisher & mass range tests
if "fisher" in prefix:
    data_dir=f"{MAP_DIR}/fisher"
elif "M15" in prefix:
   data_dir=f"{MAP_DIR}/0pt5_M15/" 
elif "M13" in prefix:
    data_dir=f"{MAP_DIR}/M13/"
elif "M12" in prefix:
    data_dir=f"{MAP_DIR}/M12/"

map_dir=os.path.join(data_dir, cosmo)

n_cores=config.getint('General','n_cores')

#map settings

#trim the same number of pixels on both sides or not
if "," in config['Maps']['trim_major']:
    trim_major=np.asarray(config.get('Maps','trim_major').split(','),dtype=int)
else:
    trim_major=config.getint('Maps','trim_major')
n_small_maps=config.getint('Maps','n_small_maps') 
trim_minor=config.getint('Maps','trim_minor')
map_dtype_c=config['Maps']['map_dtype'] 
res=config.getfloat('Maps','resolution [arcmin]')

#smoothing details
if config.has_option('Maps','smooth'):
    smooth=config.getfloat('Maps','smooth')
else:
    smooth=0.

#trim one side of the large map
one_side_trim=config.getboolean('Maps', 'one_side_trim')

#apodization details
if config.has_option('Maps','apodize_pixel'):
    apodize_pixel=config.getint('Maps','apodize_pixel')
    # print("apodize pixel nonzero")
else:
    apodize_pixel=0

#apodization type
if config.has_option('Maps','apodize_type'):
    apodize_type=config.get('Maps','apodize_type')
else:
    apodize_type='cos_pixell'

if config.has_option('Maps','divide_fsky'):
    divide_fsky=config.getboolean('Maps', 'divide_fsky')
else:
    divide_fsky=False

#all hmpdf files should be saved in float to save memory
if map_dtype_c=='double':
    map_dtype=np.float64
elif map_dtype_c=='float':
    map_dtype=np.float32
else:
    sys.exit("maps can only be in double or float precision")

#divide out pixel window function via pixell
if config.has_option('Maps','pixel_window_map'):
    pixel_window_map=config.getboolean('Maps','pixel_window_map')
else:
    pixel_window_map=False

#coupling matrix for namaster
if config.has_option("Stat","w00_file"):
    w00_file=config.get("Stat","w00_file")
else:
    w00_file=''

if config.has_option("Maps", "subtract_mean"):
    subtract_mean=config.getboolean("Maps", "subtract_mean")
else:
    subtract_mean=False

#filtering low and high ell
if config.has_option('Maps','base_filter'):
    #base_filter=np.asarray(config.get('Maps','base_filter').split(','), dtype=np.float64)
    base_filter=config.get('Maps','base_filter')
else:
    base_filter=''

#wiener filter
if config.has_option('Maps','wiener_filter'):
    wiener_filter=config.get('Maps','wiener_filter')
else:    
    wiener_filter=''

#adding noise
if config.has_section('Noise'):
    which_noise=config.get('Noise','which_noise')
else:
    which_noise=''

#smoothing total map
if config.has_option('Maps','smooth_tot'):
    smooth_tot=config.getfloat('Maps','smooth_tot')
else:
    smooth_tot=0.0

#adding white noise
if which_noise=='flat':
    noise_level=config.getfloat('Noise', 'noise_level_muK_arcmin')
else:
    noise_level=0.0

if config.has_option('Noise', 'seed'):
    noise_seed=config.getint('Noise', 'seed')
    if noise_seed==0:
        noise_seed=None
else:
    noise_seed=1

#################################
########### stats ###############
#################################

#statistics
l_edges=[]
peak_heights=[]
minima_heights=[]
MF_thresh=[]

#power spectrum
if 'cl' in stat or 'namaster' in stat:
    
    ell_edges=np.asarray(config.get('Stat','ell_edges').split(','), dtype=np.float64)
    min_ell=ell_edges[0]
    max_ell=ell_edges[1]

    if config.has_option('Stat', 'delta_ell'):
        dell=config.getfloat('Stat','delta_ell')
        l_edges=np.arange(min_ell,max_ell,dell)
    if config.has_option('Stat', 'delta_log_ell'):
        dlogell=config.getfloat('Stat','delta_log_ell')
        l_edges=np.exp(np.arange(np.log(min_ell), np.log(max_ell), dlogell))
    
    
#peak counts -- bin edges
if 'peaks' in stat:

    peak_heights=np.asarray(config.get('Stat','peak_heights').split(','), dtype=np.float64)
#    print(peak_heights)
    if config.has_option('Stat', 'sigma'):
        sigma=config.getfloat('Stat','sigma')
        peak_heights=peak_heights*sigma*1e-6
#    print(peak_heights)

if 'minima' in stat:
    minima_heights=np.asarray(config.get('Stat','minima_heights').split(','), dtype=np.float64)
    
    if config.has_option('Stat', 'sigma'):
        sigma=config.getfloat('Stat','sigma')
        minima_heights=minima_heights*sigma*1e-6

#Minkowski functionals
if 'MF' in stat:

    MF_linspace=np.asarray(config.get('Stat','MF_linspace').split(','), dtype=np.float64)
    if len(MF_linspace)!=3:
        sys.exit("Wrong specification of MF bins. Need minimimum value, maximum value and number of bins separated by commas.")

    MF_thresh=np.linspace(MF_linspace[0],MF_linspace[1],int(MF_linspace[-1]))
    if config.has_option('Stat', 'sigma'):
        sigma=config.getfloat('Stat','sigma')
        MF_thresh=MF_thresh*sigma*1e-6

#compute statistics
print("computing stats")
print(map_dir)
calc_stats=AnalyzeMaps(prefix=prefix, stat=stat, map_dir=map_dir,
                       trim_major=trim_major, n_small_maps=n_small_maps, 
                       trim_minor=trim_minor, map_dtype=map_dtype,one_side_trim=one_side_trim, res=res, smooth=smooth,
                       apodize_pixel=apodize_pixel, apodize_type=apodize_type, pixel_window_map=pixel_window_map,subtract_mean=subtract_mean,
                        n_cores=n_cores, n_arr=n_arr,l_edges=l_edges, peak_heights=peak_heights, 
                        minima_heights=minima_heights, MF_thresh=MF_thresh, w00_file=w00_file, divide_fsky=divide_fsky,
                        base_filter=base_filter, which_noise=which_noise, wiener_filter=wiener_filter, smooth_tot=smooth_tot,
                        noise_level=noise_level, noise_seed=noise_seed)

#computing standard deviation
if 'stdev' in stat:
    calc_stats.run_dir_stdev()
else:
    calc_stats.run_dir_stat()
