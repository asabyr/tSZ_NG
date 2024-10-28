import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( '../convergence/')
sys.path.append( '../suite_code/')
from stat_conv import StatConv
from read_combined_stats import *

###############################################################
#                                                             #
#       example script to compute convergence of moments      #
#                                                             #
###############################################################

cosmo=np.asarray(sys.argv[1].split(','), dtype="<U200")
prefix=np.asarray(sys.argv[2].split(','), dtype="<U200")
nfile_order=int(sys.argv[3])
nspacing=np.array([1000,15552,1000])
nspacing_fid=np.array([1000,34560,2300])


moments_each=np.array(['sigma0', 'sigma1', 'S0', 'S1', 'S2', 'K0','K1', 'K2', 'K3'])

SIM_DIR='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/'
fid_dir=SIM_DIR+'Oc_0.264_s8_0.811/full_moments/'
Oc_dir=SIM_DIR+f'{cosmo[1]}/{prefix[1]}/'
s8_dir=SIM_DIR+f'{cosmo[0]}/{prefix[0]}/'


for i in range(len(moments_each)):
    
    #fid
    
    moments_conv_fid_obj=StatConv([], 'full_moments',
                                fid_dir, 
                                'moments', 
                       nspacing=nspacing_fid, inds=np.array([0]), moments=moments_each[i], error=True,max_nsims=nspacing_fid[1], nfile_order=nfile_order)
    moments_conv_fid, moments_conv_error_fid, moments_conv_nsims_fid=moments_conv_fid_obj.compute_for_nsims()
    
    #high s8
    moments_conv_s8_obj=StatConv([], prefix[0],
                                s8_dir, 
                                'moments', 
                       nspacing=nspacing, inds=np.array([0]), moments=moments_each[i], error=True, max_nsims=nspacing[1], nfile_order=nfile_order)
    moments_conv_s8, moments_conv_error_s8, moments_conv_nsims_s8=moments_conv_s8_obj.compute_for_nsims()

    #high Oc
    moments_conv_Oc_obj=StatConv([], prefix[1],
                                Oc_dir, 
                                'moments', 
                       nspacing=nspacing, inds=np.array([0]), moments=moments_each[i], error=True, max_nsims=nspacing[1],nfile_order=nfile_order)
    moments_conv_Oc, moments_conv_error_Oc, moments_conv_nsims_Oc=moments_conv_Oc_obj.compute_for_nsims()

    conv_dict={}
    conv_dict['fid']={}
    conv_dict['fid']['moments']=moments_conv_fid
    conv_dict['fid']['error']=moments_conv_error_fid
    conv_dict['fid']['nsims']=moments_conv_nsims_fid
    
    conv_dict['high_s8']={}
    conv_dict['high_s8']['moments']=moments_conv_s8
    conv_dict['high_s8']['error']=moments_conv_error_s8
    conv_dict['high_s8']['nsims']=moments_conv_nsims_s8
    
    conv_dict['high_Oc']={}
    conv_dict['high_Oc']['moments']=moments_conv_Oc
    conv_dict['high_Oc']['error']=moments_conv_error_Oc
    conv_dict['high_Oc']['nsims']=moments_conv_nsims_Oc
    
    np.save(f"/scratch/07833/tg871330/tSZ_NG/data/{moments_each[i]}_{cosmo[0]}_{cosmo[1]}_{nfile_order}",conv_dict)
