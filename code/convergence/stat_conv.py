import sys
sys.path.append('../general/')
from read_stats import ReadStats
import copy
import numpy as np

##############################################################
# this module computes statistics for different number of sims
# to check their convergence.
##############################################################
class StatConv():

    def __init__(self, pick_stat, prefix, stat_dir,stat, 
                 nspacing=[10,5170,100], max_nsims=5184, inds=[], MF='v0', 
                 moments='sigma0', error=True, nfile_order=1):
        
        
        self.pick_stat=pick_stat #e.g. ell values
        self.prefix=prefix #prefix of the files to read
        self.stat_dir=stat_dir #directory for files
        self.stat=stat 
        self.inds=inds # inds to compute for
        self.nspacing=nspacing #n sims array (will use np.arange)
        self.max_nsims=max_nsims #maximum number of simulations available
        self.MF=MF #specify minkowski
        self.moments=moments #speficy moment
        self.error=error
        self.nfile_order=nfile_order

    def prepare_arr(self):
        
        #figure out x values
        if self.stat=='MF':
            read_all=ReadStats(self.prefix,self.stat_dir, MF_which=self.MF)
        elif self.stat=='moments':
            read_all=ReadStats(self.prefix,self.stat_dir, 
                               moments_which=self.moments)
        else:
            read_all=ReadStats(self.prefix,self.stat_dir)
        avg_stat=read_all.process_dir()
        
        if self.stat!='moments':
            x_fid=copy.deepcopy(avg_stat['x_value'+'_'+self.stat])
        
        if len(self.inds)==0 and self.stat!='moments':
            self.x_ind=np.in1d(x_fid, self.pick_stat).nonzero()[0]
            # print(self.x_ind)
        else:
            self.x_ind=copy.deepcopy(self.inds)
            # print(self.x_ind)

        #how many sims to compute avg for
        min_sim, max_sim, dsim=self.nspacing
        self.num_files=np.arange(min_sim, max_sim, dsim)
        
        #initialize arrays
        self.stat_arr=np.ones((len(self.x_ind), len(self.num_files)+1))
        self.errors=np.ones((len(self.x_ind), len(self.num_files)+1))
        self.var=np.ones((len(self.x_ind), len(self.num_files)+1))
 
        #store total average
        if self.stat=='moments':
            self.stat_arr[:, -1]=avg_stat['avg'+'_'+self.stat]
            self.errors[:, -1]=avg_stat['errors'+'_'+self.stat]
            self.var[:,-1]=np.sqrt(np.diag(avg_stat['cov'+'_'+self.stat]))
        else:
            self.stat_arr[:, -1]=avg_stat['avg'+'_'+self.stat][self.x_ind]
            self.errors[:, -1]=avg_stat['errors'+'_'+self.stat][self.x_ind]
            self.var[:,-1]=np.sqrt(np.diag(avg_stat['cov'+'_'+self.stat][self.x_ind]))
    
        # print('avg'+'_'+self.stat)
        
    def compute_for_nsims(self):
        
        self.prepare_arr()
        
        for n in range(len(self.num_files)):
            
            #calc avg
            if self.stat=='MF':
                read_nsim=ReadStats(self.prefix,self.stat_dir,nfiles=self.num_files[n], MF_which=self.MF,nfile_order=self.nfile_order)
            elif self.stat=='moments':
                read_nsim=ReadStats(self.prefix, self.stat_dir, nfiles=self.num_files[n],
                               moments_which=self.moments,nfile_order=self.nfile_order)
            else:   
                read_nsim=ReadStats(self.prefix,self.stat_dir,nfiles=self.num_files[n],nfile_order=self.nfile_order)
            
            avg_stat_nsim=read_nsim.process_dir()
            
            if self.stat=='moments':
                self.stat_arr[:,n]=avg_stat_nsim['avg'+'_'+self.stat]
                self.errors[:,n]=avg_stat_nsim['errors'+'_'+self.stat]
                self.var[:,n]=np.sqrt(np.diag(avg_stat_nsim['cov_'+self.stat]))
            else:
                self.stat_arr[:,n]=avg_stat_nsim['avg'+'_'+self.stat][self.x_ind]
                self.errors[:,n]=avg_stat_nsim['errors'+'_'+self.stat][self.x_ind]
                self.var[:,n]=np.sqrt(np.diag(avg_stat_nsim['cov_'+self.stat][self.x_ind]))
        
        num_files_full_arr=np.append(self.num_files, self.max_nsims)
        if self.error==True: 
            return self.stat_arr, self.errors, num_files_full_arr
        else:
            return self.stat_arr, self.var, num_files_full_arr
