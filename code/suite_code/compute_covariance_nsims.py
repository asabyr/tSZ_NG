import os
import glob
import numpy as np
from read_combined_stats import *
import sys

prefix=np.asarray(sys.argv[1].split(','), dtype="<U200")
tot_dim=int(sys.argv[2])
which_stats=sys.argv[3]
out_prefix=sys.argv[4]
prefix_dir=int(sys.argv[5])
Nsims=int(sys.argv[6])
if len(sys.argv)==8:
    file_order=int(sys.argv[7])
else:
    file_order=1

if file_order==0:
    print("picking simulations from bottom to top of the list")

if "M15" in prefix[0]:
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/0pt5_M15/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/0pt5_M15/"
elif "M13" in prefix[0]:
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M13/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M13/"
elif "M12" in prefix[0]:
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M12/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/M12/"
else:
    SIM_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/"
    AVG_DIR="/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/"

fid_cosmo='Oc_0.264_s8_0.811'
avg_stat_file=os.path.join(AVG_DIR, which_stats, "Nsims_"+str(Nsims)+"_"+out_prefix+"_"+which_stats+"_"+fid_cosmo)

read_cosmo=ReadCombinedStats(prefix,f'{SIM_DIR}{fid_cosmo}/', nfiles=Nsims,
                        tot_dim=tot_dim, which_stats=which_stats, out_prefix=out_prefix, out_dir_prefix=prefix_dir, nfile_order=file_order)
avg_cosmo=read_cosmo.process_dir()

print(f"will save avg stat to {avg_stat_file}")
np.save(avg_stat_file, avg_cosmo)
