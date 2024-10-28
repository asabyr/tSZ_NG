import sys
sys.path.append('../suite_code/')
import numpy as np
from scipy import linalg
from read_combined_stats import *
import copy

##################################################################
# this module computes Fisher forecasts from a set of measured 
# observables at the fiducial and +/-dtheta.
##################################################################

class Fisher:
    
    def __init__(self, stat_cov, theta,
                stat_high, stat_low, d_theta, deriv_type='central_diff', 
                from_sims=True, Nsims=5112, prefix='',tot_dim=20, Nsims_fid=34560, which_stats='',
                some_ind=[], exclude_ind=np.array([1,-1]), nfile_order=1, scale_area=1.):
        
        
        self.stat_cov=stat_cov #covariance at the fiducial parameters       
        self.exclude_ind=exclude_ind
        self.some_ind=some_ind #which indices of the observable to use
        if len(self.some_ind)>0:
            self.stat_cov=stat_cov[np.ix_(self.some_ind,self.some_ind)] #remove covariance elements accordingly
        
        self.theta=theta #array of parameter names (e.g. np.array(['Oc','s8']))
        assert list(self.theta)==list(d_theta.keys()) #assumed parameters check
        self.d_theta=d_theta #dictionary of dthetas (same length as theta)
        self.deriv_type=deriv_type #only central difference implemented right now
        assert self.deriv_type=='central_diff', "only central difference derivative scheme is currently implemented"
        self.from_sims=from_sims # to compute from sims/input is already an averaged stat, then set to True
        self.Nsims=Nsims #to compute average of the observable over this number of sims
        self.prefix=prefix #array of prefixes of the files, if computing from simulations
        self.tot_dim=tot_dim #vector length of the observable 
        self.Nsims_fid=Nsims_fid
        self.which_stats=which_stats #statistics that is computed (from cl, MF, moments, peaks, minima)
        #directory path to \pm dtheta
        #dictionary of the observable at theta+dtheta (same length as theta) if from_sims==False
        self.N_theta=len(theta) #number of parameters
       
        if "MF" in self.which_stats:
            self.out_prefix=self.prefix[0].replace("_MF","")
        elif "cl" in self.which_stats:
            self.out_prefix=self.prefix[0].replace("_cl","")
        elif "peaks" in self.which_stats:
            self.out_prefix=self.prefix[0].replace("_peaks","")
        elif "minima" in self.which_stats:
            self.out_prefix=self.prefix[0].replace("_minima","")
        elif "moments" in self.which_stats:
            self.out_prefix=self.prefix[0].replace("_moments","")

        self.nfile_order=nfile_order
        self.scale_area=scale_area
        #debias covariance
        self.corr_factor=(self.Nsims_fid-self.tot_dim-2)/(self.Nsims_fid-1)

        if self.from_sims==True: #specify directories where the vectors are stored
            self.stat_high_dir=stat_high 
            self.stat_low_dir=stat_low  
            self.compute_avg_stat() #compute avg statistics 
        else:
            #this option is not used in the end but you could specify 
            #the average observables directly 
            # (e.g. initially I was using interpolated values at dtheta from the suite
            # but combining interpolation+derivatives was too noisy)
            self.loop_N=1
            self.stat_high=np.expand_dims(stat_high,axis=1)
            self.stat_low=np.expand_dims(stat_low,axis=1)
        
        
    
    def compute_avg_stat_Nsim(self, dir_name, n_files):
        #compute average statistic for 1 directory and 1 value for sims
        
        read_cosmo=ReadCombinedStats(self.prefix, dir_name, nfiles=n_files,
                        tot_dim=self.tot_dim, which_stats=self.which_stats,
                        out_dir_prefix=1,
                        out_prefix=self.out_prefix, exclude_ind=self.exclude_ind, nfile_order=self.nfile_order)
        avg_cosmo=read_cosmo.process_dir()
        
        if len(self.some_ind)>0:
            return avg_cosmo['avg_'+self.out_prefix+"_"+self.which_stats][self.some_ind]
        return avg_cosmo['avg_'+self.out_prefix+"_"+self.which_stats]
    
        
    def compute_avg_stat(self):
        
        
        self.loop_N=1
        # initialize arrays for the statistics
        if isinstance(self.Nsims, np.ndarray):
            self.loop_N=len(self.Nsims)
        else:
            self.Nsims=np.array([self.Nsims])
        
        self.stat_high={}
        self.stat_low={}
        
        if len(self.some_ind)>0:
            self.vector_dim=len(self.some_ind)
        else:
            self.vector_dim=copy.deepcopy(self.tot_dim)
                       
        
        #loop for each param, +/- dtheta and Nsims
        # print(self.theta)
        for param in self.theta:
            self.stat_high[param]=np.zeros((self.vector_dim, self.loop_N))
            self.stat_low[param]=np.zeros((self.vector_dim, self.loop_N))
            for j in range(self.loop_N):
                    
                self.stat_high[param][:,j]=self.compute_avg_stat_Nsim(self.stat_high_dir[param], self.Nsims[j])
                self.stat_low[param][:,j]=self.compute_avg_stat_Nsim(self.stat_low_dir[param], self.Nsims[j])
        
        
    def compute_deriv(self, param, n_sims):
        #n_sims here is just an index corresponding to some number of sims
        if self.deriv_type=='central_diff':
            return (self.stat_high[param][:,n_sims]-self.stat_low[param][:,n_sims])/(2.0*self.d_theta[param])
        

    def compute_fisher(self):
        
        self.fisher_matrix=np.zeros((self.N_theta, self.N_theta, self.loop_N))
        self.theta_cov=np.zeros((self.N_theta, self.N_theta, self.loop_N))
        self.derivs=np.zeros((self.N_theta,self.vector_dim, self.loop_N))
        
        for n in range(self.loop_N):
            
            for i in range(self.N_theta):
                dsdi=self.compute_deriv(param=self.theta[i], n_sims=n)
                
                for j in range(self.N_theta):
                    dsdj=self.compute_deriv(param=self.theta[j], n_sims=n)
                    self.derivs[j,:,n]=copy.deepcopy(dsdj)
                    step_1=linalg.solve(self.stat_cov*self.scale_area,dsdj)
                    # step_1=np.dot(dsdj,linalg.inv(self.stat_cov)*self.scale_area)
                    self.fisher_matrix[i,j,n]=np.dot(dsdi,step_1*self.corr_factor)
                    
            self.theta_cov[:,:,n]=linalg.inv(self.fisher_matrix[:,:,n])
        
        if self.loop_N==1:
            return self.theta_cov[:,:,0]
        
        return self.theta_cov[:,:]
