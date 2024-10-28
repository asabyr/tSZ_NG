import getdist
from getdist.gaussian_mixtures import GaussianND
from getdist import plots
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../forecast')
from fisher_funcs import *
from fisher import Fisher

###############################################################
#                                                             #
#       example script to compute fisher forecasts            #
#                      +tests                                 #
###############################################################

plot_params= {
    'figure.figsize': (10,8),
    'axes.labelsize': 20,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 15,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 7,
    'xtick.minor.size': 3,
    'ytick.major.size': 7,
    'ytick.minor.size': 3,
    'legend.frameon':False
}
plt.rcParams.update(plot_params)

THETA_FID_LABELS={r'\Omega_c':0.264,r'\sigma_8':0.811}
SIM_DIR='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/fisher/'
FIG_DIR='/scratch/07833/tg871330/tSZ_NG/figs/fisher_masses/'

#covariance computed using half or all simulations
def get_covariances(file_dir, prefix):
    
    dict_cov={}
    
    cov_file=np.load(f'{file_dir}/{prefix}_MF_V0_V1_V2_Oc_0.264_s8_0.811.npy', allow_pickle=True).item()
    covariance=cov_file[f'cov_{prefix}_MF_V0_V1_V2']

    cov_file_half=np.load(f'{file_dir}/Nsims_17280_{prefix}_half_sims_MF_V0_V1_V2_Oc_0.264_s8_0.811.npy', allow_pickle=True).item()
    covariance_half=cov_file_half[f'cov_{prefix}_half_sims_MF_V0_V1_V2']

    cov_file_half_other=np.load(f'{file_dir}/Nsims_17280_{prefix}_half_sims_other_MF_V0_V1_V2_Oc_0.264_s8_0.811.npy', allow_pickle=True).item()
    covariance_half_other=cov_file_half_other[f'cov_{prefix}_half_sims_other_MF_V0_V1_V2']
    
    dict_cov['fid_covariance']=covariance
    dict_cov['covariance_half']=covariance_half
    dict_cov['covariance_half_other']=covariance_half_other

    return dict_cov

#plot as a check
def plot_fisher(cov1, cov2, cov3, labels, file_name):
    
    THETA_FID_LABELS={r'\Omega_c':0.264,r'\sigma_8':0.811}
    SIM_DIR='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/'
    FIG_DIR='/scratch/07833/tg871330/tSZ_NG/figs/fisher_masses/'
    
    names = list(THETA_FID_LABELS.keys())
    labels_param = list(THETA_FID_LABELS.keys())
    fid_arr=list(THETA_FID_LABELS.values())

    matrix1 = GaussianND(fid_arr, cov1, 
                        labels = labels_param, names = names, label=labels[0])
    matrix2 = GaussianND(fid_arr, cov2, 
                        labels = labels_param, names = names, label=labels[1])
    matrix3 = GaussianND(fid_arr, cov3, 
                        labels = labels_param, names = names, label=labels[2])

    g = plots.get_single_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.6
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize=15
    g.settings.legend_fontsize=15

    g.triangle_plot([matrix1, matrix2, matrix3],names, filled = False,legend_loc = 'upper right',
                    contour_colors = ['red', 'black', 'blue'])

    plt.savefig(FIG_DIR+file_name+".pdf")

def mass_range_fisher(prefix, 
                      root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8', 
                      covariance_conv=True, Nsims=np.array([1700,2556]), Nsims_max=5112):

    

    high_folder, low_folder, dtheta_values=get_fisher_dirs(root=root)
    # print(high_folder) 
    #check covariance stability 
    covariances=get_covariances(f'{root}/MF_V0_V1_V2', prefix)
    
    fisher_dict={}
    for key,value in covariances.items():

        MF_fisher=Fisher(covariances[key], np.array(['Oc','s8']),
            high_folder,low_folder, d_theta=dtheta_values, deriv_type='central_diff',
                 prefix=np.array([prefix+"_MF"]),
                 which_stats='MF_V0_V1_V2',tot_dim=60)

        MF_cov=MF_fisher.compute_fisher()
        a,b,theta,area=get_fisher_params(MF_fisher.theta_cov[:,:,0])

        fisher_dict[key]=MF_fisher.theta_cov[:,:,0]
        fisher_dict[key+"_area"]=area

    plot_fisher(cov1=fisher_dict['fid_covariance'],cov2=fisher_dict['covariance_half'], cov3=fisher_dict['covariance_half_other'], 
                labels=np.array(['fid_covariance','covariance_half','covariance_half_other']), file_name=prefix+"_covariance")


    #check nsims stability
    Nsims_check=[]
    for n in Nsims:
        Nsims_check.append(int(n))
        Nsims_check.append(int(Nsims_max-n))
    Nsims_check=np.array(Nsims_check)

    file_order = np.empty(int(len(Nsims)*2))
    file_order[::2] = 1
    file_order[1::2] = 0

    for i,n in enumerate(Nsims_check):
        
        MF_fisher=Fisher(covariances['fid_covariance'], np.array(['Oc','s8']),
            high_folder,low_folder, d_theta=dtheta_values, deriv_type='central_diff',
                 prefix=np.array([prefix+"_MF"]),
                 which_stats='MF_V0_V1_V2',tot_dim=60, nfile_order=file_order[i], Nsims=n)

        MF_cov=MF_fisher.compute_fisher()
        a,b,theta,area=get_fisher_params(MF_fisher.theta_cov[:,:,0])

        fisher_dict[f'{n}_{file_order[i]}']=MF_fisher.theta_cov[:,:,0]
        fisher_dict[f'{n}_{file_order[i]}'+'_area']=area
    # print(fisher_dict.keys())
 
    dict_keys=[]
    for key,value in fisher_dict.items():
        dict_keys.append(key)
    dict_keys=np.array(dict_keys)
    
    #just plotting check using different number of sims
    inds=np.array([6,8,10,12])
    for j in range(int(len(inds)/2)):
        
        a=inds[2*j]
        b=inds[2*j+1] 
        plot_fisher(cov1=fisher_dict['fid_covariance'],cov2=fisher_dict[dict_keys[a]], cov3=fisher_dict[dict_keys[b]], 
                labels=np.array(['fid_covariance',dict_keys[a],dict_keys[b]]),file_name=prefix+"_"+dict_keys[a]+"_"+dict_keys[b])

    #check conv with sims
    dnsims=500
    nsims=np.concatenate((np.arange(dnsims,5112,dnsims),[5112]))
    
    areas=np.zeros(len(nsims))
    
    MF_fisher_nsims=Fisher(covariances['fid_covariance'], np.array(['Oc','s8']),
            high_folder,low_folder, d_theta=dtheta_values, deriv_type='central_diff',
                 prefix=np.array([prefix+"_MF"]),
                 which_stats='MF_V0_V1_V2',tot_dim=60, Nsims=nsims)
    MF_cov_nsims=MF_fisher_nsims.compute_fisher()

    for i in range(len(nsims)):
        a_fisher_nsims,b_fisher_nsims,theta_fisher_nsims,area_fisher_nsims=get_fisher_params(MF_fisher_nsims.theta_cov[:,:,i])
        areas[i]=area_fisher_nsims
    
    fisher_dict['covariance_nsims']=MF_fisher_nsims.theta_cov
    fisher_dict['nsims']=nsims
    fisher_dict['nsims_area']=areas
    
    np.save(f"/scratch/07833/tg871330/tSZ_NG/data/fisher_{prefix}", fisher_dict)
    return MF_fisher.theta_cov[:,:,0]


MF_fisher_M16=mass_range_fisher('20bin_1to3sigma', root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8')
MF_fisher_M15=mass_range_fisher('20bin_1to3_M15', root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/0pt5_M15')
MF_fisher_M13=mass_range_fisher('20bin_1to3_M13', root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M13')
MF_fisher_M13_noise=mass_range_fisher('20bin_1to3_M13_0pt01muK', root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M13')
MF_fisher_M12=mass_range_fisher('20bin_1to3_M12', root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M12')
