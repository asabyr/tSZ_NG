
####################################################################################
# This module includes some useful functions for filtering/smoothing.              #
# Parts are adopted from convergence module in LensTools by Andrea Petri:          #
# https://github.com/apetri/LensTools/tree/master/lenstools/image/convergence.py   #
####################################################################################

import numpy as np
from scipy.ndimage import filters
from scipy import interpolate
import sys
import copy
import os 
import constants as const

this_dir=os.path.dirname(os.path.abspath(__file__))
FILTER_DIR=this_dir.replace("code/general","data")

class MapUtils():
    
    def __init__(self, np_data, side_angle_arcmin):
        
        #assumes the np_data is an (N,N) array -- i.e. equal sides
        self.np_data=np_data
        self.side_angle_arcmin=side_angle_arcmin
    
    def get_ell_map(self):
        
        side_angle_rad=self.side_angle_arcmin*const.arcmin_to_rad

        lx = np.fft.fftfreq(self.np_data.shape[0])
        ly = np.fft.fftfreq(self.np_data.shape[1])

        lx*=(self.np_data.shape[0]/side_angle_rad)*2*np.pi
        ly*=(self.np_data.shape[1]/side_angle_rad)*2*np.pi

        l_map=np.sqrt((lx[:,None]**2 + ly[None,:]**2))
        
        return l_map
    
    def smooth(self,FWHM_arcmin,kind="gaussian",**kwargs):
        
        #just using scipy filter
        if kind=='gaussian':
			
            sigma_arcmin=FWHM_arcmin*const.FWHM_to_sigma
            smoothing_scale_pixel=sigma_arcmin*self.np_data.shape[0]/self.side_angle_arcmin   
            smoothed_data = filters.gaussian_filter(self.np_data, smoothing_scale_pixel,**kwargs)
        
        #smoothing in fourier space
        elif kind=='gaussianFFT':
            
            #figure out sigma in radians
            gauss_smooth_sigma_arcmin=FWHM_arcmin*const.FWHM_to_sigma
            gauss_smooth_sigma_rad=gauss_smooth_sigma_arcmin*const.arcmin_to_rad

            #apply gaussian smoothing in fourier space
            ell_map = self.get_ell_map()
            ell_filter=np.exp(-0.5*ell_map**2.0*gauss_smooth_sigma_rad**2.0)
            smoothed_data = np.fft.ifft2(ell_filter*np.fft.fft2(self.np_data)).real
            
        else:
            
            sys.exit("Only Gaussian smoothing implemented so far")
        
        return smoothed_data
    
    def filter_gauss_smooth_map(self, input_filter, FWHM_arcmin, return_filter=False, **kwargs):
        
        ## filter according to an ell filter in an input .npy file &
        ## smooth with a gaussian beam 
        ## (e.g. good to use if you want to filter our low & high ell and smooth)

        #figure our sigma in radians
        gauss_smooth_sigma_arcmin=FWHM_arcmin*const.FWHM_to_sigma
        self.gauss_smooth_sigma_rad=gauss_smooth_sigma_arcmin*const.arcmin_to_rad
        
        #make filter function from file
        ell_filter_file = np.load(os.path.join(FILTER_DIR,input_filter),allow_pickle=True).item()
        ell_filter_interp = interpolate.interp1d(ell_filter_file['ell'],ell_filter_file['ell_filter'],**kwargs)
        
        #get all physical ells
        ell_map = self.get_ell_map()
        
        #construct total filter & apply to map
        ell_filter=np.exp(-0.5*ell_map**2.0*self.gauss_smooth_sigma_rad**2.0)*ell_filter_interp(ell_map)
        filtered_data=np.fft.ifft2(ell_filter*np.fft.fft2(self.np_data))
        
        if return_filter==True:
            return filtered_data.real, ell_filter, ell_map
        else:
            return filtered_data.real
    
    def filter_map(self, input_filter, return_filter=False, **kwargs):

        ## filter according to an ell filter in an input .npy file
        ## (e.g. good to use if you want to apply a wiener filter)

        #make filter function from file
        #print(f"applying {os.path.join(FILTER_DIR,input_filter)}")
        ell_filter = np.load(os.path.join(FILTER_DIR,input_filter),allow_pickle=True).item()
        ell_filter_interp = interpolate.interp1d(ell_filter['ell'],ell_filter['ell_filter'],**kwargs)
        
        #get all physical ells
        ell_map = self.get_ell_map()

        #construct total filter & apply to map
        ell_filter=ell_filter_interp(ell_map)
        filtered_data=np.fft.ifft2(ell_filter*np.fft.fft2(self.np_data))
        
        if return_filter==True:
            return filtered_data.real, ell_filter, ell_map
        else:
            return filtered_data.real

	    
    
