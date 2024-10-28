import numpy as np
import os
import sys
import copy
#####################################################################
# This module splits one large map into smaller ones,
# while trimming the edges.
# ! This code doesn't automatically compute how many pixels
# can be trimmed/number of maps. You need to know and specify this apriori!
#####################################################################

class SplitMaps:

    def __init__(self, fname, trim_major, n_small_maps, trim_minor,
                  data_type=np.float32, one_side_trim=False):

        self.fname=fname #assumes .bin format
        self.trim_major=trim_major #large map trim
        self.n_small_maps=n_small_maps
        self.trim_minor=trim_minor #smaller maps trim
        self.data_type=data_type #np.float32 for "float" in C and np.float64 for "double" in C
        self.one_side_trim=one_side_trim
        

    @staticmethod
    def trim_frame(one_map, trim):
        return one_map[trim:-trim,trim:-trim]

    def process_maps(self):

        #read map file and reshape
        self.map_data=np.fromfile(self.fname, dtype=self.data_type)
        self.Nside=int(np.sqrt(len(self.map_data)))
        self.map_arr=self.map_data.reshape(self.Nside,self.Nside)

        #trim edges
            
        #trimming two sides differently
        if isinstance(self.trim_major, np.ndarray)==True: 
            if len(self.trim_major)==2: #check that only 2 numbers specified
                self.big_map_trimmed=self.map_arr[self.trim_major[0]:-self.trim_major[-1],self.trim_major[0]:-self.trim_major[-1]]
            else:
                sys.exit("check large trim specification")
        else: #one number for large map trim
            if self.trim_major==0:
                self.big_map_trimmed=copy.deepcopy(self.map_arr)
            elif self.one_side_trim==True:
                self.big_map_trimmed=self.map_arr[self.trim_major:,self.trim_major:]
            else:
                self.big_map_trimmed=self.trim_frame(self.map_arr, self.trim_major)
        
        if self.n_small_maps>1:
            #split into smaller maps
            self.small_maps=self.split_tiles()
            #trim smaller maps
            self.small_maps_trimmed=self.trim_tiles()
            return self.small_maps_trimmed
        else:
            return np.array(self.big_map_trimmed)



    def save_small_maps(self):

        self.small_maps_trimmed=self.process_maps()
        split_path=os.path.split(self.fname)
        small_dir_path=os.path.join(split_path[0],"small_maps/")

        for i in range(self.small_maps_trimmed.shape[0]):
            small_map_name=split_path[1].replace('.bin', "_"+str(i)+'.bin')
            immutable_bytes=bytes(self.small_maps_trimmed[i].astype(np.float32))
            print("saving small to "+small_dir_path+small_map_name)
            with open(small_dir_path+small_map_name, "wb") as binary_file:
                binary_file.write(immutable_bytes)

    def split_tiles(self):
        #efficient way of splitting https://stackoverflow.com/questions/34940529/fastest-method-of-splitting-image-into-tiles
        map_height, map_width = self.big_map_trimmed.shape
        tile_height=map_height/np.sqrt(self.n_small_maps)
        tile_width=map_width/np.sqrt(self.n_small_maps)

        tiles=self.big_map_trimmed.reshape(int(np.sqrt(self.n_small_maps)), int(tile_height), int(np.sqrt(self.n_small_maps)), int(tile_width))
        tiles_swaped=tiles.swapaxes(1,2)

        tiles_flattened=tiles_swaped.reshape(self.n_small_maps,tiles_swaped.shape[-2],tiles_swaped.shape[-1])

        return tiles_flattened

    def trim_tiles(self):

        new_n_pix=self.small_maps.shape[-1]-self.trim_minor*2. #final number of pixels for each map
        
        new_small_maps=np.zeros((self.small_maps.shape[0],int(new_n_pix), int(new_n_pix))) #initialize array

        for i in range(self.small_maps.shape[0]):

            new_small_maps[i,:,:]=self.trim_frame(self.small_maps[i,:,:], int(self.trim_minor))

        return new_small_maps
