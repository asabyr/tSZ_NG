###########################################################################
# This module is for making noise maps, 							  	  #
# adapted from noise module in LensTools by Andrea Petri:        #	  					  
# https://github.com/apetri/LensTools/tree/master/lenstools/image/noise.py#
###########################################################################

import numpy as np
import constants as const
import os 
from scipy import interpolate
import sys
from extra_funcs import *
import matplotlib.pyplot as plt
import copy
#figure out directory to correctly find the noise files
this_dir=os.path.dirname(os.path.abspath(__file__))
NOISE_DIR=this_dir.replace("code/general","data")
FILTER_DIR=this_dir.replace("code/general","data")

class Noise:
	
	def __init__(self, n_pixels_side, side_angle_arcmin):
		#assumes the shape is a square
		self.n_pixels_side=n_pixels_side
		self.side_angle_arcmin = side_angle_arcmin
		
		#convert to radians
		self.side_angle_rad = self.side_angle_arcmin*const.arcmin_to_rad
		self.ell_map=get_ell_map(self.n_pixels_side, self.side_angle_rad)
		
	def fourierMap(self, power_func, **kwargs):
		
		lpix = 2*np.pi/self.side_angle_rad
		#Check power spectrum shape
		if isinstance(power_func,np.ndarray):
			assert power_func.shape[0] == 2,"If you want an interpolated power spectrum you should pass a (l,Pl) array!"
		
			#Perform the interpolation for all ells
			ell,Pell = power_func
			power_interp = interpolate.interp1d(ell,Pell,**kwargs)
			Pl = power_interp(self.ell_map)
		
		else:
			Pl = power_func(self.ell_map,**kwargs)
		# plt.plot(l, Pl)
		# plt.xscale('log')
		assert Pl[Pl>=0.0].size == Pl.size

		#Generate real and imaginary parts
		real_part = np.sqrt(Pl) * np.random.normal(loc=0.0,scale=1.0,size=self.ell_map.shape) * lpix/(2.0*np.pi)
		imaginary_part = np.sqrt(Pl) * np.random.normal(loc=0.0,scale=1.0,size=self.ell_map.shape) * lpix/(2.0*np.pi)
		#make a noise map in fourier space
		ft_map = (real_part + imaginary_part*1.0j) * self.ell_map.shape[0]**2.0
		
		return ft_map
	
	def fromPowerSpectrum(self, power_func, seed=0,return_fourier=True, **kwargs):
	
		#Initialize random number generator
		if seed is not None:
			print("initializing seed")
			np.random.seed(seed)

		#Generate a random Fourier realization 
		ft_map = self.fourierMap(power_func,**kwargs)
		if return_fourier==True:
			return ft_map
		else:
			#return real map
			noise_map = np.fft.ifft2(ft_map)
			return noise_map.real

	def tSZ_noise(self, level='baseline', noise_dir=NOISE_DIR,
			deproj='standardILC', seed=0, filter_low_high_ell='', beam_FWHM_arcmin=1.4, **kwargs):

		"""
		This method generates a noise map for a tSZ field based on SO/S4 post-component noise curves.
		
		INPUT:
		level: specify between 'baseline' or 'goal' SO noise (str)
		noise_dir: directory to noise curves, defaults to /data/ folder in this directory
		deproj: relevant to SO noise, pick deprojection method 'standardILC','CMBdeproj', 'CIBdeproj' or 'CMBandCIBdeproj' (str)
		seed: random seed (int)
		kwargs: keyword arguments to be passed to power_func, or to the interpolate.interp1d routine

		OUTPUT:
		noise map: the same shape as specified when class object is initialized (numpy array of floats)
		"""

		if level=='baseline':
			noise_file="SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
		elif level=='goal':
			noise_file="SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"
		elif level=='S4':
			noise_file='S4_190604d_2LAT_T_default_noisecurves_deproj0_SENS0_mask_16000_ell_TT_yy.txt'
			column=2	
		else:
			sys.exit("pick baseline or goal SO noise")

		noise_ell_Cl_all=np.loadtxt(os.path.join(noise_dir,noise_file))
		
		if level=='S4':
			ind_80=np.where(noise_ell_Cl_all[:,0]>=80)[0]
			full_file=copy.deepcopy(noise_ell_Cl_all)
			noise_ell_Cl_all=full_file[ind_80,:]

		
		if level!='S4':

			#for SO noise, need to pick method
			if deproj=='standardILC':
				column=1
			elif deproj=='CMBdeproj':
				column=2
			elif deproj=='CIBdeproj':
				column=3
			elif deproj=='CMBandCIBdeproj':
				column=4
			else:
				sys.exit("pick 'standardILC','CMBdeproj', 'CIBdeproj' or 'CMBandCIBdeproj'")

		#beam
		beam_sq=gauss_beam(ell=noise_ell_Cl_all[:,0], FWHM_arcmin=beam_FWHM_arcmin)
		
		#apply filter directly to noise power spectra before generating maps
		if len(filter_low_high_ell)>0:
			print("using filter on noise")
			filter_dict=np.load(os.path.join(NOISE_DIR,filter_low_high_ell), allow_pickle=True).item()
			assert np.array_equal(filter_dict['ell'],noise_ell_Cl_all[:,0])
			noise_ell_Cl=np.vstack((noise_ell_Cl_all[:,0],noise_ell_Cl_all[:,column]*beam_sq*filter_dict['ell_filter']))

			self.fromPowerSpectrum(power_func=noise_ell_Cl,seed=seed, **kwargs)
		else:
			noise_ell_Cl=np.vstack((noise_ell_Cl_all[:,0],noise_ell_Cl_all[:,column]*beam_sq))
		
		return self.fromPowerSpectrum(power_func=noise_ell_Cl,seed=seed, **kwargs)
	
	@staticmethod
	def flat_ps(ell, noise_level):
		return noise_level**2.0*np.ones(len(ell))
	
	def tSZ_white_noise(self, noise_muK_arcmin, freq_GHz=150.0, seed=0):
		
		noise_y_rad=muK_arcmin_to_y_rad(mu_K_arcmin=noise_muK_arcmin,freq_GHz=freq_GHz)

		return self.fromPowerSpectrum(power_func=Noise.flat_ps, seed=seed, noise_level=noise_y_rad)
