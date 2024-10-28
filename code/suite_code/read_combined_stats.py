import numpy as np
import sys
import glob
import copy
import os
#####################################################################
# This module is for post-processing analysis --
# reads the computed multiple statistics from maps 
#####################################################################

class ReadCombinedStats:

    def __init__(self, prefix, stat_dir, nfiles=0,
                 remove_outliers=False, 
                 tot_dim=100, 
                 which_stats='cl_peaks_MF_V0_V1_V2_minima_moments_sigma0_sigma1_S0_S1_S2_K0_K1_K2_K3',
                 out_prefix='', out_dir_prefix=0, exclude_ind=np.array([1,-1]), nfile_order=1):

        self.prefix=prefix #prefix of the files to read (array)
        self.stat_dir=stat_dir #directory for files (array)
        self.nfiles=nfiles #how many files to read 
        self.remove_outliers=remove_outliers
        self.tot_dim=tot_dim
        self.which_stats=which_stats
        self.all_stats_y=[]
        self.all_stats_x=[]
        self.out_prefix=out_prefix
        self.out_dir_prefix=out_dir_prefix
        if not np.any(exclude_ind)==True:
            self.exclude_ind=np.array([0])
        else:
            self.exclude_ind=exclude_ind
        
        self.possible_moments=np.array(['sigma0','sigma1','S0','S1','S2','K0','K1','K2','K3'])
        self.possible_MF=np.array(['v0','v1','v2'])
        self.nfile_order=nfile_order

    def process_dir(self):
        #compute average stat and covariance in a directory
        self.prepare()
        self.compute_avg_stat()
        self.compute_cov_stat()
        return self.stat_dict
    
    
    def get_file_names(self, stat, n_prefix):
        
        if self.out_dir_prefix==1:
            stat_files_not_sort=np.array(glob.glob(self.stat_dir+f"/{self.prefix[n_prefix]}/"+self.prefix[n_prefix]+"*.npy"))
        else: 
            stat_files_not_sort=np.array(glob.glob(self.stat_dir+f"/{stat}/"+self.prefix[n_prefix]+"*.npy"))
            # print(self.stat_dir+f"/{stat}/"+self.prefix[n_prefix]+"*.npy")
        stat_files=np.sort(stat_files_not_sort)
        #print(stat_files)
        if self.nfile_order==1:
            self.all_file_names[:,n_prefix]=stat_files[:self.nfiles]
        elif self.nfile_order==0:
            self.all_file_names[:,n_prefix]=stat_files[-self.nfiles:]

        if self.out_dir_prefix==1:
            last_element_path=np.char.strip(self.all_file_names[:,n_prefix], chars=self.stat_dir+f"/{self.prefix[n_prefix]}/")
        else: 
            last_element_path=np.char.strip(self.all_file_names[:,n_prefix], chars=self.stat_dir+f"/{stat}/")
        
        self.all_file_names_clean[:,n_prefix]=np.array([x.replace(self.prefix[n_prefix]+"_","") for x in last_element_path])
        
    def prepare(self):

        n_prefix=0
        
        #number of stats
        possible_stats=np.array(['cl','peaks','MF','minima','moments'])
        match=np.isin(self.which_stats.split("_"), possible_stats)
        
        self.file_stats=np.array(self.which_stats.split("_"))[match]
        
        n_stats=match.sum()
        
        
        #add some code for the case where statistic is repeated
        repeat=1
        if n_stats==1:
            if len(self.prefix)>1:
                repeat=len(self.prefix)
                # print(f"repeat={repeat}")
        
        self.all_file_names_clean=np.empty((self.nfiles,int(n_stats*repeat)),dtype="<U500")
        self.all_file_names=np.empty((self.nfiles,int(n_stats*repeat)),dtype="<U500")

        for i in range(repeat):
            #power spectrum
            if "cl" in self.which_stats:
                self.all_stats_x.append("ell")
                self.all_stats_y.append("cl_scaled")
                self.get_file_names("cl", n_prefix)
                n_prefix+=1
            
            #peaks
            if "peaks" in self.which_stats:
                self.all_stats_x.append("peak_heights")
                self.all_stats_y.append("peak_counts")
                self.get_file_names("peaks", n_prefix)
                n_prefix+=1
            
            #MF
                
            if "V0" in self.which_stats:
                self.all_stats_x.append("midpoints")
                self.all_stats_y.append("v0")
            
            if "V1" in self.which_stats:
                self.all_stats_x.append("midpoints")
                self.all_stats_y.append("v1")
            
            if "V2" in self.which_stats:
                self.all_stats_x.append("midpoints")
                self.all_stats_y.append("v2")
            
            if "MF" in self.which_stats:
                self.get_file_names("MF", n_prefix)
                n_prefix+=1

            #minima
            if "minima" in self.which_stats:
                self.all_stats_x.append("minima_heights")
                self.all_stats_y.append("minima_counts")
                self.get_file_names("minima", n_prefix)
                n_prefix+=1
                
            #moments
            if "sigma0" in self.which_stats:
                self.all_stats_x.append("sigma0")
                self.all_stats_y.append("sigma0")
            if "sigma1" in self.which_stats:
                self.all_stats_x.append("sigma1")
                self.all_stats_y.append("sigma1")

            if "S0" in self.which_stats:
                self.all_stats_x.append("S0")
                self.all_stats_y.append("S0")
            if "S1" in self.which_stats:
                self.all_stats_x.append("S1")
                self.all_stats_y.append("S1")
            if "S2" in self.which_stats:
                self.all_stats_x.append("S2")
                self.all_stats_y.append("S2")

            if "K0" in self.which_stats:
                self.all_stats_x.append("K0")
                self.all_stats_y.append("K0")
            if "K1" in self.which_stats:
                self.all_stats_x.append("K1")
                self.all_stats_y.append("K1")
            if "K2" in self.which_stats:
                self.all_stats_x.append("K2")
                self.all_stats_y.append("K2")
            if "K3" in self.which_stats:
                self.all_stats_x.append("K3")
                self.all_stats_y.append("K3")
            
            if "moments" in self.which_stats:
                self.get_file_names("moments", n_prefix)
                n_prefix+=1
        
        for j in range(n_stats*repeat-1):

            if np.array_equal(self.all_file_names_clean[:,0], self.all_file_names_clean[:,j+1])==False:
                
                sys.exit("stat files don't correspond to the same maps")
        
        

    def compute_avg_stat(self):

        
        #compute full vector for each file
        self.y_values=np.zeros((self.tot_dim, self.nfiles))

        
        for i in range(self.nfiles):
            
            full_vector=np.array([])
            tot_stat=0
            count_moments=0
            count_MF=0
            #need to loop through files and through each MF/moment that was asked
            for k in range(len(self.all_stats_y)):
                
                stat_file=np.load(self.all_file_names[i,tot_stat], allow_pickle=True).item()
                
                if "cl" in self.all_file_names[i,tot_stat] and len(self.exclude_ind)>1:    
                    full_vector=np.append(full_vector,stat_file[self.all_stats_y[k]][self.exclude_ind[0]:self.exclude_ind[-1]])
                else:
                    full_vector=np.append(full_vector,stat_file[self.all_stats_y[k]])
                
                #count stats only if not repeating moments/MF
                if k+1!=len(self.all_stats_y):
                    
                    
                    if self.all_stats_y[k] in self.possible_moments:
                        if self.all_stats_y[k+1] in self.possible_moments:
                            # print("another moment stat")
                            tot_stat+=0
                        else:  
                            tot_stat+=1
                    elif self.all_stats_y[k] in self.possible_MF:
                        if self.all_stats_y[k+1] in self.possible_MF:
                            # print("another MF")
                            
                            if int(self.all_stats_y[k+1][-1])<int(self.all_stats_y[k][-1]):
                                tot_stat+=1
                            else:
                                tot_stat+=0
                        else:
                            tot_stat+=1
                    else:
                        tot_stat+=1
            if i==0:
                full_vector_sum=np.array(full_vector).flatten()
            else:
                full_vector_sum+=np.array(full_vector).flatten()
            #sys.exit(0)
            self.y_values[:,i]=copy.deepcopy(np.array(full_vector).flatten())

        #compute average
        self.avg_stat=np.sum(self.y_values, axis=1)/self.nfiles
        self.avg_stat=full_vector_sum/self.nfiles
        
        #final dictionary 
        self.stat_dict={}
        self.stat_dict['avg_'+self.out_prefix+"_"+self.which_stats]=copy.deepcopy(self.avg_stat)
        self.stat_dict['all_'+self.out_prefix+"_"+self.which_stats]=copy.deepcopy(self.y_values)
        

    def compute_cov_stat(self):

        diff=np.zeros((self.tot_dim, 1))
        cov=np.zeros((self.tot_dim, self.tot_dim))
        
        #compute covariance
        for i in range(len(self.y_values[0,:])):
            diff[:,0]=self.y_values[:,i]-self.avg_stat
            mult=np.dot(diff,diff.T)
            cov+=mult


        #save covariance and errors
        self.covariance=cov/(self.nfiles-1.)
        self.errors=np.sqrt(np.diag(self.covariance))/np.sqrt(self.nfiles)

        self.stat_dict['cov_'+self.out_prefix+"_"+self.which_stats]= self.covariance
        self.stat_dict['errors_'+self.out_prefix+"_"+self.which_stats]=self.errors
