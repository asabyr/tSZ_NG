import sys
import numpy as np
import glob

class RebinPeaks:
    #can be used for rebinning minima files too
    def __init__(self, prefix, stat_dir, new_prefix,new_thresholds,sigma=1):
        
        self.prefix=prefix #prefix of the files to read
        self.stat_dir=stat_dir #directory for files
        self.new_prefix=new_prefix
        self.new_thresholds=new_thresholds
        self.sigma=sigma
        if sigma!=1:
            self.new_thresholds=new_thresholds*sigma
            
        if "peaks" not in self.prefix and "minima" not in self.prefix:
            sys.exit("check the files; can only re-bin peak or minima files")
        if "peaks" in self.prefix:
            self.stat="peak"
        elif "minima" in self.prefix:
            self.stat="minima"
        self.process_dir()
        
    def process_dir(self):

        self.all_stat_files=glob.glob(self.stat_dir+"/"+self.prefix+"*.npy")
        self.new_stat_files=glob.glob(self.stat_dir+"/"+self.new_prefix+"*.npy")

        for i in range(len(self.all_stat_files)):
            file_name=self.all_stat_files[i].split("/")[-1].replace(self.prefix+'_','')
            new_out_file=self.stat_dir+"/"+self.new_prefix+'_'+file_name
            
            if new_out_file not in self.new_stat_files:

                one_file=np.load(self.all_stat_files[i], allow_pickle=True).item()
                peaks=one_file[self.stat+'_values']
                loc=one_file[self.stat+'_locs']
                counts, bin_edges=np.histogram(np.array(peaks), bins=self.new_thresholds)
                centers=(bin_edges[:-1] + bin_edges[1:]) / 2

                newdict={}
                newdict[self.stat+'_heights']=centers
                newdict[self.stat+'_counts']=counts
                newdict[self.stat+'_values']=np.array(peaks)
                newdict[self.stat+'_locs']=loc

                np.save(new_out_file,newdict)