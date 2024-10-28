import numpy as np
from scipy.special import jve
import constants as const
import copy
import matplotlib.pyplot as plt
from scipy import interpolate

#gaussian beam squared, flat sky
def gauss_beam(ell, FWHM_arcmin):
    FWHM_rad=FWHM_arcmin*const.arcmin_to_rad
    sigma=FWHM_rad*const.FWHM_to_sigma
    return np.exp(-ell**2*sigma**2)

#pixel window, circular pixels
def pixel_window_radial(ell, pix_size_arcmin):
    bessel_order=1
    pix_size_rad=pix_size_arcmin*const.arcmin_to_rad
    return 2*jve(bessel_order,ell*pix_size_rad)/(ell*pix_size_rad)

#units
def muK_arcmin_to_y_rad(mu_K_arcmin, freq_GHz):
    
    #convert relevant quantities
    TCMB_muK=const.TCMB*10.0**6.0
    freq_Hz=freq_GHz*10.0**9
    X=const.h_planck*freq_Hz/(const.k_boltz*const.TCMB)
    
    #spectral function
    g_nu=X/np.tanh(X/2.0)-4.0

    #compute y
    y_arcmin=mu_K_arcmin/TCMB_muK/g_nu
    y_rad=y_arcmin*const.arcmin_to_rad

    return y_rad

#split bin edges except the first & last ones
def split_bins(bin_edges):
    bin_edges_trim=copy.deepcopy(bin_edges[1:-1])
    new_bin_edges=[bin_edges[0]]
    for i in range(len(bin_edges_trim)):
        new_bin_edges.append(bin_edges_trim[i])
        if i<len(bin_edges_trim)-1:
            centers=(bin_edges_trim[:-1] + bin_edges_trim[1:])/2.0
            new_bin_edges.append(np.round(centers[i],3))
    
    new_bin_edges.append(bin_edges[-1])
    
    return np.array(new_bin_edges)

def cov_to_corr(cov):
    #slow way but checks for issues
    corr=np.empty_like(cov)
    for i in range(len(cov[:,0])):
        for j in range(len(cov[0,:])):
            if cov[i,i]==0:
                print("divide by zero")
                print(i)
            elif cov[j,j]==0:
                print("divide by zero")
                print(j)
            else:
                corr[i,j]=cov[i,j]/(np.sqrt(cov[i,i])*np.sqrt(cov[j,j]))       
    return corr

#filter out low and high ells
def tanh_ell(ells, range_to_zero, tanh_steep=0.75):
    center_ell=np.mean(range_to_zero)
    tanh_y=0.5*(np.tanh(tanh_steep*(ells-center_ell))+1)
    return tanh_y

def make_filter(noise_type, low_ell_range=[80,90], 
                high_ell_range=[7940,7950]):
    
    if noise_type=='SO_baseline':
        noise_file="SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
    elif noise_type=='SO_goal':
        noise_file="SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"
    elif noise_type=='S4':
        noise_file="S4_190604d_2LAT_T_default_noisecurves_deproj0_SENS0_mask_16000_ell_TT_yy.txt"
    
    #read file 
    noise_ell_Cl_all=np.loadtxt("/Users/asabyr/Documents/tSZ_NG/data/"+noise_file)
    noise_ells=copy.deepcopy(noise_ell_Cl_all[:,0])
    
    #find relevant indices
    
    min_low, min_high=low_ell_range
    max_low, max_high=high_ell_range
    
    ell_below_min=np.where(noise_ells<=min_low)[0]
    ell_between_min=np.where((noise_ells>min_low) & (noise_ells<min_high))[0]
    ell_above_max=np.where(noise_ells>=max_high)[0]
    ell_between_max=np.where((noise_ells<max_high) & (noise_ells>max_low))[0]
    
    
    tanh_filter=np.ones(len(noise_ells))
    tanh_filter[ell_between_min]=tanh_ell(noise_ells[ell_between_min], low_ell_range, tanh_steep=0.75)
    tanh_filter[ell_between_max]=np.flip(tanh_ell(noise_ells[ell_between_max], high_ell_range, tanh_steep=0.75))
    tanh_filter[ell_above_max]=0.0
    tanh_filter[ell_below_min]=0.0

    tanh_filter_dict={}
    tanh_filter_dict['ell']=noise_ells
    tanh_filter_dict['ell_filter']=tanh_filter

                             
    np.save(f"../data/filter_tanh_low_{int(min_low)}_{int(min_high)}_high_{int(max_low)}_{int(max_high)}", 
            tanh_filter_dict)

    return tanh_filter_dict

def get_ell_map(n_pixels,side_angle_rad):
        
    ellx = np.fft.fftfreq(n_pixels)
    elly = np.fft.fftfreq(n_pixels)

    ellx*=(n_pixels/side_angle_rad)*2*np.pi
    elly*=(n_pixels/side_angle_rad)*2*np.pi

    ell_map=np.sqrt((ellx[:,None]**2 + elly[None,:]**2))
        
    return ell_map

def get_stdev(all_stdev_dir):
    
    import glob
    import copy
    all_stdev=glob.glob(all_stdev_dir+"*")
    devs=np.array([])
    for dev_file in all_stdev:
        
        devs_one_file=np.loadtxt(dev_file)
        if len(devs)==0:
            devs=copy.deepcopy(devs_one_file)
        else:
            devs=np.append(devs, devs_one_file)
    return np.mean(devs)

def construct_wiener(cl_yy_file, N_ell_arr, save_to_file, beam=True):
    
    #make yy interpolator (divide out prefactor)
    cl_yy=np.load(f"/Users/asabyr/Documents/tSZ_NG/suite_constraints/cl/{cl_yy_file}", 
                  allow_pickle=True).item()
    cl_yy_l_edges=np.arange(25.0,8000.0,100.0)
    cl_yy_ell=0.5*(cl_yy_l_edges[:-1] + cl_yy_l_edges[1:])[1:-1]
    cl_yy_prefactor=(10.0**12.0*cl_yy_ell*(cl_yy_ell+1)/(2*np.pi))
    
    Bl2=gauss_beam(N_ell_arr[:,0],1.4)
    
    cl_yy_interp = interpolate.interp1d(np.log(cl_yy_ell),cl_yy['avg_77ell_cl']/cl_yy_prefactor, kind='cubic', fill_value='extrapolate')

    # compute wiener & normalize
    
    if beam==True:
        yy_over_tot=cl_yy_interp(np.log(N_ell_arr[:,0]))/(cl_yy_interp(np.log(N_ell_arr[:,0]))+N_ell_arr[:,1]*Bl2)
    else:
        yy_over_tot=cl_yy_interp(np.log(N_ell_arr[:,0]))/(cl_yy_interp(np.log(N_ell_arr[:,0]))+N_ell_arr[:,1])
    
    max_wiener=np.amax(yy_over_tot)
    
    #check how well interpolation works
    plt.figure()
    plt.plot(cl_yy_ell,cl_yy['avg_77ell_cl']/cl_yy_prefactor, marker='.', label='cl measured')
    plt.plot(N_ell_arr[:,0],cl_yy_interp(np.log(N_ell_arr[:,0])), label='cl interp')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'Cl')
    plt.legend()
    
    #exact ratio
    plt.figure()
    plt.plot(cl_yy_ell,cl_yy['avg_77ell_cl']/cl_yy_prefactor/cl_yy_interp(np.log(cl_yy_ell)), label='cl interp')
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel('ratio')
    plt.legend()
    
    #save filter
    norm_wiener_dict={}
    norm_wiener_dict['ell']=N_ell_arr[:,0]
    norm_wiener_dict['ell_filter']=yy_over_tot/max_wiener
    norm_wiener_dict['yy_power_spectrum']=cl_yy_interp(np.log(N_ell_arr[:,0]))
    
    if beam==True:
        norm_wiener_dict['noise_power_spectrum']=N_ell_arr[:,1]*Bl2
    else:
        norm_wiener_dict['noise_power_spectrum']=N_ell_arr[:,1]
    
    if len(save_to_file)>0:
        np.save("/Users/asabyr/Documents/tSZ_NG/data/"+save_to_file, norm_wiener_dict)
        
    return norm_wiener_dict