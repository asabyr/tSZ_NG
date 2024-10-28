from read_combined_stats import *
import copy
import numpy as np
#convergence of specific covariance elements
class CovConv():

    def __init__(self, 
                 prefix, 
                 stat_dir, 
                 nspacing=[100,5000,500], 
                 max_nsims=5000,
                 inds_x=[],inds_y=[],
                 tot_dim=100, 
                 which_stats='cl_peaks_MF_V0_V1_V2_minima_moments_sigma0_sigma1_S0_S1_S2_K0_K1_K2_K3', 
                    out_prefix=''):
        
        #for reading stats
        self.prefix=prefix #prefix of the files to read
        self.stat_dir=stat_dir #directory for files
        
        #for conv
        self.nspacing=nspacing
        self.max_nsims=max_nsims
        self.inds_x=inds_x
        self.inds_y=inds_y
        
        #for reading stats
        self.tot_dim=tot_dim
        self.which_stats=which_stats
        
        self.out_prefix=out_prefix

        
    def prepare_arr(self):
        
        
        
        #how many sims to compute avg for
        min_sim, max_sim, dsim=self.nspacing
        self.num_files=np.arange(min_sim, max_sim, dsim)
        
        #initialize arrays
        self.stat_arr=np.ones((len(self.inds_x), len(self.num_files)+1))
        
        
        read_all=ReadCombinedStats(self.prefix,self.stat_dir,nfiles=self.max_nsims, 
                                  tot_dim=self.tot_dim, which_stats=self.which_stats, out_prefix=self.out_prefix)
        avg_stat=read_all.process_dir()
        self.full_final_cov=avg_stat['cov'+'_'+self.out_prefix+"_"+self.which_stats]
        
        
        #store total average
        for i in range(len(self.inds_x)):
            self.stat_arr[i, -1]=avg_stat['cov'+'_'+self.out_prefix+"_"+self.which_stats][self.inds_x[i],self.inds_y[i]]
            
            
        
    def compute_for_nsims(self):
        
        self.prepare_arr()
        
        for n in range(len(self.num_files)):
            
            read_nsim=ReadCombinedStats(self.prefix,self.stat_dir,nfiles=self.num_files[n], 
                                  tot_dim=self.tot_dim, which_stats=self.which_stats, out_prefix=self.out_prefix)
            avg_stat_nsim=read_nsim.process_dir()
            
            for i in range(len(self.inds_x)):
                self.stat_arr[i, n]=avg_stat_nsim['cov'+'_'+self.out_prefix+"_"+self.which_stats][self.inds_x[i],self.inds_y[i]]
             
        num_files_full_arr=np.append(self.num_files, self.max_nsims)
        
        return self.full_final_cov, self.stat_arr, num_files_full_arr
