import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import copy
sys.path.append('../general/')
N_PIXEL_ZERO=2

cosmo=sys.argv[1]
sims_path=sys.argv[2]
maps_path=os.path.join(sims_path,cosmo)

if len(sys.argv)==5:
    print("splitting into batches of map files")
    n_split=int(sys.argv[3])
    n_arr=int(sys.argv[4])
else:
    n_split=0

check_path=maps_path+"/check_all_maps/"
if not os.path.exists(check_path):
    os.mkdir(check_path)

if n_split>0:
    all_maps_total=glob.glob(maps_path+"/*bin")
    all_maps_split=np.array_split(all_maps_total, n_split)
    all_maps=all_maps_split[n_arr]
    fname=check_path+cosmo+f"_check_maps_{n_arr}_{n_split}.txt"
else:
    all_maps=glob.glob(maps_path+"/*bin")
    fname=check_path+cosmo+"_check_maps.txt"

#helper function to check for zeros
def check_zero_streaks(some_values):
    some_values=np.equal(some_values, 0).view(np.int8)
    some_values=np.concatenate((np.array([0]),some_values))
    some_values=np.concatenate((some_values,np.array([0])))

    absdiff=np.abs(np.diff(some_values))
    ranges = np.where(absdiff == 1)[0].reshape(-1,2)
    another_diff=np.diff(ranges)
    more_than=np.where(another_diff>=N_PIXEL_ZERO)[0]
    return more_than

def plot_map(map_file, out_dir):
    map_file_name=os.path.basename(map_file)
    map_data=np.fromfile(map_file,dtype=np.float32)
    Nside=int(np.sqrt(len(map_data)))
    map_arr=map_data.reshape(Nside,Nside)
    plt.figure()
    plt.imshow(np.log(map_arr))
    plt.colorbar()
    plt.savefig(out_dir+map_file_name.replace(".bin",".pdf"))
    plt.close()

np.savetxt(fname,["#will print map names that have streaks of zeros"],fmt='%s')  
#np.savetxt(check_path+cosmo+"_check_maps.txt",["#will print map names that have streaks of zeros"],fmt='%s')  
with open(fname,"a") as check_file:
  
    for i in range(len(all_maps)):
    
        sim_file=all_maps[i].replace(maps_path+'/','')
        print(sim_file)
        map_data=np.fromfile(all_maps[i], dtype=np.float32)

        #horizontal streaks
        map_data_zeros=check_zero_streaks(map_data)    
        if len(map_data_zeros)>0:
            np.savetxt(check_file, [sim_file],fmt='%s')
            plot_map(os.path.join(sims_path,cosmo,sim_file),check_path) 
        #vertical streaks
        Nside=int(np.sqrt(len(map_data))) 
        map_data_columns=map_data.reshape(Nside,Nside).flatten(order='F')
        
        map_data_columns_zeros=check_zero_streaks(map_data_columns) 
        if len(map_data_columns_zeros)>0:
            np.savetxt(check_file, [sim_file],fmt='%s')
            plot_map(os.path.join(sims_path,cosmo,sim_file), check_path)
    np.savetxt(check_file,['done with all files'],fmt='%s') 
