import os
import sys
import glob
import numpy as np
#import lenstools
#from lenstools import GaussianNoiseGenerator
from lenstools import MapStats
#from lenstools import MapUtils
from map_utils import MapUtils
from signal_noise import SignalNoise
from noise import Noise
from split_maps import SplitMaps
import copy
from pixell import enmap, utils, enplot
from namaster_flat import NamasterFlat
#import astropy.units as u

#####################################################################
# This module computes statistics on maps in a given directory      
# (includes power spectrum, peaks, minima, moments, minkowski functionals).
# Things to keep in mind:
# -- It only computes statistics if the files don't already exist      
# -- Maps are assumed to be .bin files
# -- Saves stats to .npy files      
# -- Assumes statistics are saved to map_dir/prefix_{stat}/
#####################################################################

class AnalyzeMaps:

    def __init__(self, prefix, stat, map_dir,
                 trim_major=np.array([5,6]), n_small_maps=36, trim_minor=50,
                 map_dtype=np.float32, one_side_trim=False, res=0.1, smooth=1.4,
                apodize_pixel=5, apodize_type='cos_pixell', pixel_window_map=False, subtract_mean=False,
                n_cores=1, n_arr=0,
                l_edges=[], peak_heights=[], MF_thresh=[], minima_heights=[], w00_file='', 
                divide_fsky=False, base_filter='', which_noise='', 
                wiener_filter='', smooth_tot=0.0, noise_level=0.0, noise_seed=1):
        
        #general 
        self.prefix=prefix #output files prefix [str]
        self.stat=stat #string with stats "cl_peaks_MF" [str]
        self.map_dir=map_dir #directory to maps[str]

        #map settings
        self.trim_major=trim_major #pixels to trim from large map (either integer or array of 2 integers)
        self.n_small_maps=n_small_maps #number of maps to cut into [int]
        self.trim_minor=trim_minor #pixels to trim from cut maps [int]
        self.map_dtype=map_dtype #map data type (almost always float) [dtype]
        self.one_side_trim=one_side_trim #True/False, whether to trim large map on one side or not [bool]
        
        self.res=res #resolution in arcmins [arcmin]
        self.smooth=smooth #Gaussian smoothing FWHM [arcmin]
        self.apodize_pixel=apodize_pixel # number of pixels used for apodization[int]
        self.apodize_type=apodize_type # only supports cos_pixell right now [str]
        self.pixel_window_map=pixel_window_map # to divide out square pixel window function [bool]
        self.w00_file=w00_file #namster coupling matrix file [str]
        self.subtract_mean=subtract_mean
        self.divide_fsky=divide_fsky
        self.base_filter=base_filter
        self.wiener_filter=wiener_filter
        self.smooth_tot=smooth_tot
        self.noise_level=noise_level
        self.noise_seed=noise_seed
        
        #parallellization 
        self.n_cores=n_cores #number of tasks the job was split into [int]
        self.n_arr=n_arr #which task this is [int]

        #stat settings: exact settings, not in terms of sigma
        self.l_edges=l_edges 
        self.peak_heights=peak_heights 
        self.MF_thresh=MF_thresh
        self.minima_heights=minima_heights
        
        #adding noise
        self.which_noise=which_noise

    def print_some_parameters(self):
        
        print(f"resolution [arcmin]:{self.res}")
        print(f"smooth [FWHM, arcmin]:{self.smooth}")
        print(f"apod pixels :{self.apodize_pixel}")
        
    #compute statistics in a map directory
    def run_dir_stat(self):
        self.print_some_parameters()
        self.list_files()
        self.dir_stats()
    
    #compute standard deviations in a map directory
    def run_dir_stdev(self):
        self.print_some_parameters()

        if self.prefix!="":
            underscore="_"
        else:
            underscore=""

        if self.n_cores>1:

            existing_bin_all=glob.glob(self.map_dir+"/*.bin")
            existing_bin_split=np.array_split(existing_bin_all, self.n_cores)
            self.existing_bin=existing_bin_split[self.n_arr]
        
        else:
            
            self.existing_bin=glob.glob(self.map_dir+"/*.bin")
        
        self.existing_stdev=glob.glob(self.map_dir+"/"+self.prefix+underscore+"stdev/*txt")
        self.dir_stdev()

    
    def list_files(self):
        
        # get all stat files
        if self.prefix!="":
            underscore="_"
        else:
            underscore=""
        
        if "cl" in self.stat:
            self.existing_cl=glob.glob(self.map_dir+"/"+self.prefix+underscore+"cl/"+"*.npy")
            #print(self.existing_cl)
        if "peaks" in self.stat:
            self.existing_peaks=glob.glob(self.map_dir+"/"+self.prefix+underscore+"peaks/"+"*.npy")
        if "minima" in self.stat:
            self.existing_minima=glob.glob(self.map_dir+"/"+self.prefix+underscore+"minima/"+"*.npy")
        if "MF" in self.stat:
            self.existing_MF=glob.glob(self.map_dir+"/"+self.prefix+underscore+"MF/"+"*.npy")
        if "moments" in self.stat:
            self.existing_moments=glob.glob(self.map_dir+"/"+self.prefix+underscore+"moments/"+"*.npy") 
        if "namaster" in self.stat:
            self.existing_namaster=glob.glob(self.map_dir+"/"+self.prefix+underscore+"namaster/"+"*.npy")
        
        #get all map files and split into number of tasks if performed in parallel
        if self.n_cores>1:
            existing_bin_all=glob.glob(self.map_dir+"/*.bin")
            existing_bin_split=np.array_split(existing_bin_all, self.n_cores)
            self.existing_bin=existing_bin_split[self.n_arr]
        else:
            self.existing_bin=glob.glob(self.map_dir+"/*.bin")
    

    def dir_stats(self):
       
        #loop over all map files 
        for fname in self.existing_bin:
            print(fname)            
            self.save_stat(fname)
    
    def dir_stdev(self):
        
        # print(self.existing_bin)
        
        for fname in self.existing_bin:
            print(fname)
            self.calc_stdev(fname)
    
    #####################################
    ########### map functions ###########
    #####################################
       
    def smooth_map(self, one_map):
        
        if self.smooth>0:
            print(f"smoothed map with {self.smooth} arcmin")
            Nside=len(one_map[0,:])
            map_not_smooth=MapUtils(one_map, side_angle_arcmin=self.res*Nside)
            return map_not_smooth.smooth(self.smooth)
        else:
            return one_map

    def apodize_map(self, one_map):
        
        if self.apodize_pixel>0:
            print(f"apodized map with {self.apodize_pixel} pixels")
            return enmap.apod(one_map,self.apodize_pixel)
        else:
            return one_map
    
    def remove_pixel_window(self, one_map):
        
        box=np.array([[-self.Npixels*self.res/2.0, self.Npixels*self.res/2.0],[self.Npixels*self.res/2.0,-self.Npixels*self.res/2.0]])*utils.arcmin
        shape, wcs = enmap.geometry(pos=box, res=self.res*utils.arcmin, proj='car')
        pixell_map=enmap.enmap(one_map, wcs=wcs)
        windowed_map=enmap.apply_window(pixell_map, pow=-1.0) #divide out the pixel window func.
        # print("divided pixel window") 
        return windowed_map
    
    def divide_pixel_window(self, one_map):
        
        if self.pixel_window_map==True:
            print(f"removed pixel window function")
            unpixel_wind_map=self.remove_pixel_window(one_map)
            return unpixel_wind_map
        else:
            return one_map

    def apod_smooth_pl(self, small_map):
        
        #apodization               
        one_map_apod=self.apodize_map(small_map)
        
        #smoothing
        one_map_smooth=self.smooth_map(one_map_apod)
        
        #pixel window function
        one_map_ready=self.divide_pixel_window(one_map_smooth)
        
        if self.subtract_mean==True:
            print("subtracted the mean")
            one_map_final=one_map_ready-np.mean(one_map_ready)
        else:
            print("did not subtract the mean")
            one_map_final=copy.deepcopy(one_map_ready)
        
        return one_map_final

    def compute_apod_fsky(self):
        map_ones=np.ones((self.Npixels, self.Npixels))
        apod_map=enmap.apod(map_ones, self.apodize_pixel)
        return np.mean(apod_map)

     ####################################
     ######## noise and filtering #######
     #################################### 

    def filter_smooth_noise(self, small_map):

        #create noise

        noise_map_obj=Noise(self.Npixels,self.Npixels*self.res)

        if self.which_noise=='flat' and self.noise_level!=0:
            noise_map_raw=noise_map_obj.tSZ_white_noise(noise_muK_arcmin=self.noise_level, seed=self.noise_seed)
        else:
            noise_map_raw=noise_map_obj.tSZ_noise(level=self.which_noise, seed=self.noise_seed,beam_FWHM_arcmin=self.smooth,
                                                    kind='linear',fill_value=0.0, bounds_error=False)
        
        tSZ_noise_obj=SignalNoise(self.Npixels,self.Npixels*self.res)
        apod_tSZ=enmap.apod(small_map, self.apodize_pixel)
        tSZ_noise_map=tSZ_noise_obj.signal_noise_filter_fourier(signal_map=apod_tSZ,
                                                        noise_map_fourier=noise_map_raw,
                                                    base_filter=self.base_filter,
                                                    signal_FWHM_arcmin=self.smooth,
                                                    tot_FWHM_arcmin=self.smooth_tot,
                                                    extra_filter=self.wiener_filter,
                                                    kind='linear', fill_value=0.0, bounds_error=False)

        return tSZ_noise_map-np.mean(tSZ_noise_map)

    def filter_smooth(self, small_map):
        
        #function returns a filtered and smoothed tSZ map 
        print("apodizing, filtering ell and smoothing tSZ map")
        tSZ_map_obj=MapUtils(enmap.apod(small_map, self.apodize_pixel), side_angle_arcmin=self.Npixels*self.res)
        tSZ_map_filt=tSZ_map_obj.filter_gauss_smooth_map(input_filter=self.base_filter, FWHM_arcmin=self.smooth, fill_value=0.0, bounds_error=False) 
        
        return tSZ_map_filt-np.mean(tSZ_map_filt)

    #####################################
    ######## compute stat functions #####
    #####################################

    def calc_stdev(self, map_file):
        
        #map object
        tSZ_map=SplitMaps(fname=map_file, trim_major=self.trim_major, n_small_maps=self.n_small_maps,\
        trim_minor=self.trim_minor, data_type=self.map_dtype, one_side_trim=self.one_side_trim)
        
        self.maps=tSZ_map.process_maps()
        
        self.Npixels=self.maps[0].shape[0]
        #print(self.Npixels)
        #sys.exit(0)

        stdev_file=map_file.replace(self.map_dir+'/','').replace('.bin','.txt')
        stdev_file_path=f'{self.map_dir}/{self.prefix}_stdev/{self.prefix}_stdev_{stdev_file}'
        
        if stdev_file_path not in self.existing_stdev:
            stdev_arr=np.zeros(self.n_small_maps)
            for i in range(self.n_small_maps):
                print(map_file)
                if "Oc_0.264_s8_0.811" in map_file:
                    hmpdf_N=40
                else:
                    hmpdf_N=12
                n_1=float(map_file.split('_')[-2])
                n_2=float(map_file.split('_')[-1].replace(".bin",''))
                
                if self.noise_seed==1:
                    self.noise_seed=int((n_1-1)*hmpdf_N*self.n_small_maps+n_2*self.n_small_maps+i)
                
                if len(self.which_noise)!=0:
                    #adding noise
                    one_map=self.filter_smooth_noise(self.maps[i])
                elif len(self.base_filter)!=0:
                    #filtering low/high ell
                    one_map=self.filter_smooth(self.maps[i])
                else:
                    one_map=self.apod_smooth_pl(self.maps[i])
                stdev_arr[i]=np.std(one_map)

            np.savetxt(stdev_file_path, stdev_arr)        


    def save_stat(self, map_file):
        
        #output file
        if self.prefix=='':
            extra_underscore=''
        else:
            extra_underscore='_'
        
        #define output file
        npy_file=map_file.replace(self.map_dir+'/','').replace('.bin','_X.npy')
        self.out_file=f'{self.map_dir}/{self.prefix}_stat_dir/{self.prefix}{extra_underscore}stat_{npy_file}'

        #map object
        tSZ_map=SplitMaps(fname=map_file, trim_major=self.trim_major, n_small_maps=self.n_small_maps,\
                trim_minor=self.trim_minor, data_type=self.map_dtype, one_side_trim=self.one_side_trim)
        self.maps=tSZ_map.process_maps()
        self.Npixels=self.maps[0].shape[0]
        
        #compute each requested stat
        if "cl" in self.stat:
            self.side_deg=self.maps[0].shape[0]*self.res/60.0 #convert side angle to degrees
            # print(f"side angle in degrees={self.side_deg}")
            self.compute_stat_small_maps("cl")
        if "peaks" in self.stat:
            self.compute_stat_small_maps("peaks")
        if "minima" in self.stat:
            self.compute_stat_small_maps("minima")
        if "MF" in self.stat:
            self.compute_stat_small_maps("MF")
        if "moments" in self.stat:
            self.compute_stat_small_maps("moments")
        
        if "namaster" in self.stat:
            map_ones=np.ones((self.Npixels, self.Npixels))
            self.mask_apod=enmap.apod(map_ones,self.apodize_pixel)
            self.compute_stat_small_maps("namaster")

    def compute_stat_small_maps(self, one_stat):
        
        out_file=self.out_file.replace("stat_dir",one_stat).replace("stat",one_stat)
        
        for i in range(self.maps.shape[0]):
            
            #apply all needed things to the map & initialize map object
            out_file_one=out_file.replace("X",str(int(i)))
            
            n_1=float(out_file_one.split('_')[-3])
            n_2=float(out_file_one.split('_')[-2])
            n_3=float(out_file_one.split('_')[-1].replace(".npy",''))
            
            if "Oc_0.264_s8_0.811" in out_file:
                hmpdf_N=40
            else:
                hmpdf_N=12
            
            if self.noise_seed!=None: 
                self.noise_seed=int((n_1-1)*hmpdf_N*self.n_small_maps+n_2*self.n_small_maps+n_3)
                
            if len(self.which_noise)!=0:
                #adding noise
                one_map_final=self.filter_smooth_noise(self.maps[i])
            elif len(self.base_filter)!=0:
                #filtering low/high ell
                one_map_final=self.filter_smooth(self.maps[i]) 
            else:
                one_map_final=self.apod_smooth_pl(self.maps[i])
            map_object=MapStats(one_map_final)
            
            if one_stat=='cl':
                
                if  out_file_one not in self.existing_cl:
                    
                    ps_dict=map_object.powerSpectrum(l_edges=self.l_edges,side_angle_deg=self.side_deg)
                    
                    if self.divide_fsky==True:
                        fsky=self.compute_apod_fsky()
                        ps_dict['cl']/=fsky
                        ps_dict['cl_scaled']/=fsky
                        print(f"fsky:{fsky}")
                        
                    np.save(out_file_one,ps_dict)
                    print("saved power spectrum to file")

            elif one_stat=='peaks':
                
                if  out_file_one not in self.existing_peaks:
                    peak_dict=map_object.countPeaks(thresholds=self.peak_heights)
                    np.save(out_file_one,peak_dict)
                    print("saved peaks to file")

            elif one_stat=='minima':

                if out_file_one not in self.existing_minima:
                    minima_dict=map_object.countMinima(thresholds=self.minima_heights)
                    np.save(out_file_one,minima_dict)
                    print("saved minima to file")

            elif one_stat=='MF':
                
                if  out_file_one not in self.existing_MF:
                    MF_dict=map_object.minkowskiFunctionals(thresholds=self.MF_thresh)
                    np.save(out_file_one,MF_dict)
                    print("saved MFs to file")

            elif one_stat=='moments':
                if out_file_one not in self.existing_moments:
                    moments_dict=map_object.nine_moments()
                    np.save(out_file_one, moments_dict)
                    print("saved moments to file")

            elif one_stat=='namaster':

                namaster_object=NamasterFlat(map_arr=self.maps[i], mask_arr=self.mask_apod, 
                                         Nside=self.Npixels, res=self.res, ledges=self.l_edges, w00_file=self.w00_file)

                if out_file_one not in self.existing_namaster:
                    ps_dict=namaster_object.compute_namaster()
                    np.save(out_file_one,ps_dict)
