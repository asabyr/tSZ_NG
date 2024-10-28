import numpy as np
import sys
import glob
import copy

#####################################################################
# This is an earlier module is for post-processing analysis --
# reads the computed statistics on maps (done via analyze_maps and/or calc_stats)
#####################################################################

class ReadStats:

    def __init__(self, prefix, stat_dir, nfiles=0,
                 MF_which='v0', moments_which='sigma0', 
                 remove_outliers=False, nfile_order=1):

        self.prefix=prefix #prefix of the files to read
        self.stat_dir=stat_dir #directory for files
        self.nfiles=nfiles #how many files to read
        self.moments_which=moments_which
        self.remove_outliers=remove_outliers
        self.nfile_order=nfile_order

        if "cl" in self.prefix:
            self.stat_name='cl'
            self.stat_x_arr='ell'
            self.stat_y_arr='cl_scaled'
        elif "namaster" in self.prefix:
            self.stat_name='cl'
            self.stat_x_arr='ell'
            self.stat_y_arr='cl_scaled'

        elif "peaks" in self.prefix:
            self.stat_name='peaks'
            self.stat_x_arr='peak_heights'
            self.stat_y_arr='peak_counts'

        elif "MF" in self.prefix:
            self.stat_name='MF'
            self.stat_x_arr='midpoints'
            self.stat_y_arr=MF_which
    
        elif "moments" in self.prefix:
            self.stat_name='moments'
            self.stat_y_arr=moments_which

        elif "minima" in self.prefix:
            self.stat_name='minima'
            self.stat_x_arr='minima_heights'
            self.stat_y_arr='minima_counts'

    def process_dir(self):
        #compute average stat and covariance in a directory
        self.compute_avg_stat()
        self.compute_cov_stat()
        return self.stat_dict

    def compute_avg_stat(self):

        #compute the average

        #get all file names
        self.all_stat_files_not_sort=glob.glob(self.stat_dir+"/"+self.prefix+"*.npy")
        self.all_stat_files_orig=np.sort(self.all_stat_files_not_sort)
        # print(self.all_stat_files[0])
        
        #figure out "x" dimension
        one_file=np.load(self.all_stat_files_orig[0], allow_pickle=True).item()

        if self.stat_name=='moments':
            n_x=1
        else:
            n_x=int(len(one_file[self.stat_x_arr]))

        #loop through all or just subset of files
        if self.nfiles>0:
            self.loopn=self.nfiles
            if self.nfile_order==1:
                self.all_stat_files=self.all_stat_files_orig[:self.nfiles]
            elif self.nfile_order==0:
                self.all_stat_files=self.all_stat_files_orig[-self.nfiles:]
        else:
            self.loopn=len(self.all_stat_files_orig)
            self.all_stat_files=copy.deepcopy(self.all_stat_files_orig)
            

        #collect x and y values
        x_values=np.zeros((n_x, self.loopn))
        self.y_values=np.zeros((n_x, self.loopn))

        #add stat and number of sims
        # I add number of sims based on previous set-up where I saved multiple sims in one file
        # decided to leave in case I go back to that

        for i in range(self.loopn):
            # print(self.all_stat_files[i])
            stat_file=np.load(self.all_stat_files[i], allow_pickle=True).item()
            
            #moments don't really have an x array
            if self.stat_name!='moments':
                x_values[:,i]=stat_file[self.stat_x_arr]

            if i==0:
                totsum=copy.deepcopy(stat_file[self.stat_y_arr])
                self.num_sims=1
            else:
                totsum+=copy.deepcopy(stat_file[self.stat_y_arr])
                self.num_sims+=1
            
            self.y_values[:,i]=copy.deepcopy(stat_file[self.stat_y_arr])

        #compute average
        print("total number of maps")
        print(self.num_sims)
        self.avg_stat=totsum/self.num_sims
        if self.stat_name!='moments':
            self.x_arr=x_values[:,0]
        print("calculated average statistic")

        #check that statistics were computed for the same bins/ells/thresholds
        if self.stat_name!='moments':
            for j in range(self.loopn-1):
                if np.array_equal(x_values[:,0],x_values[:,j+1])==False:
                    sys.exit("x values are not the same across files")

        #remove extreme points in moments computation in case of any failures
        #default is to not remove anything
        if  self.stat_name=='moments':
            self.y_values=self.remove_extremas(copy.deepcopy(self.y_values))
            print(np.shape(self.y_values))
            self.avg_stat=np.mean(self.y_values)

        #final dictionary 
        self.stat_dict={}
        if self.stat_name!='moments':
            self.stat_dict['x_value_'+self.stat_name]= copy.deepcopy(self.x_arr)
        self.stat_dict['avg_'+self.stat_name]=copy.deepcopy(self.avg_stat)
        self.stat_dict['all_'+self.stat_name]=copy.deepcopy(self.y_values)
        
    
    def remove_extremas(self, y_values):
    
        if self.remove_outliers==True:
            new_arr=copy.deepcopy(y_values)
            for i in range(4):
                new_arr=np.delete(new_arr,np.argmax(new_arr))
                new_arr=np.delete(new_arr,np.argmin(new_arr))
            return new_arr
        else:
            return copy.deepcopy(y_values[0])



    def compute_cov_stat(self):

        #initialize arrays
        if self.stat_name!='moments':
            vec_len=len(self.avg_stat)
        else:
            vec_len=1
        
        diff=np.zeros((vec_len, 1))
        cov=np.zeros((vec_len, vec_len))

        if self.stat_name!='moments':
            clean_y_values=len(self.y_values[0,:])
        else:
            clean_y_values=len(self.y_values[:])

        #compute covariance
        for i in range(clean_y_values):

            if self.stat_name!='moments':
                diff[:,0]=self.y_values[:,i]-self.avg_stat
            else:
                diff[:,0]=self.y_values[i]-self.avg_stat
                
            mult=np.dot(diff,diff.T)
            cov+=mult


        #save covariance and errors
        #self.covariance=cov/(self.num_sims-1.)
        self.covariance=cov/(clean_y_values-1.)
        #self.errors=np.sqrt(np.diag(self.covariance))/np.sqrt(self.num_sims)
        self.errors=np.sqrt(np.diag(self.covariance))/np.sqrt(clean_y_values)

        self.stat_dict['cov_'+self.stat_name]= self.covariance
        self.stat_dict['errors_'+self.stat_name]=self.errors
