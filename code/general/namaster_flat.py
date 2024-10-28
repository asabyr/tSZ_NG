import sys
import numpy as np
import pymaster as nmt
import copy
import os
class NamasterFlat:

    def __init__(self, map_arr, mask_arr, Nside, res, 
                 ledges, w00_file):
        
        self.map_arr=map_arr
        self.mask_arr=mask_arr
        self.Nside=Nside
        self.res=res #in arcmin
        self.ledges=ledges #these are actually centers, need to rename
        self.w00_file=w00_file

    def compute_namaster(self):
        Lx = self.Nside*self.res/60.0 * np.pi/180.0 #in radians
        Ly = self.Nside*self.res/60.0 * np.pi/180.0

        f0_flat = nmt.NmtFieldFlat(Lx, Ly, self.mask_arr, [self.map_arr])

        l0_bins = copy.deepcopy(self.ledges)
        lf_bins = copy.deepcopy(self.ledges)

        b_flat = nmt.NmtBinFlat(l0_bins, lf_bins)

        self.ells = b_flat.get_effective_ells()

        if os.path.exists(self.w00_file)==True:
            w00_flat = nmt.NmtWorkspaceFlat()
            w00_flat.read_from(self.w00_file)
        else:
            w00_flat = nmt.NmtWorkspaceFlat()
            w00_flat.compute_coupling_matrix(f0_flat, f0_flat, b_flat)
            w00_flat.write_to(self.w00_file)
            #sys.exit(0)
        
        cl00_coupled_flat = nmt.compute_coupled_cell_flat(f0_flat, f0_flat, b_flat)
        self.cl00_uncoupled_flat = w00_flat.decouple_cell(cl00_coupled_flat)

        cl_factor=self.ells*(self.ells+1)*10.0**12/(2*np.pi)
        cl_dict={}
        cl_dict['ell']=self.ells
        cl_dict['cl']=self.cl00_uncoupled_flat[0]
        cl_dict['cl_scaled']=self.cl00_uncoupled_flat[0]*cl_factor
        
        return cl_dict


        
        
