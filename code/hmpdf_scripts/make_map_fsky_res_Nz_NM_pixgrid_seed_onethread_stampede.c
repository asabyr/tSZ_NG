/*! [compile] */
/*!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/07833/tg871330/stampede3/software/hmpdf  */
/*!gcc --std=gnu99 -I$TACC_GSL_INC -I/work/07833/tg871330/stampede3/software/hmpdf/include -o make_map_fsky_res_Nz_NM_pixgrid_seed_onethread_stampede make_map_fsky_res_Nz_NM_pixgrid_seed_onethread_stampede.c -L/work/07833/tg871330/stampede3/software/hmpdf/ -lhmpdf */
/*!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/07833/tg871330/software_scratch/hmpdf */
/*!gcc --std=gnu99 -I$TACC_GSL_INC -I/scratch/07833/tg871330/software_scratch/hmpdf/include -o make_map_fsky_res_Nz_NM_pixgrid_seed_onethread_stampede make_map_fsky_res_Nz_NM_pixgrid_seed_onethread_stampede.c -L/scratch/07833/tg871330/software_scratch/hmpdf/ -lhmpdf */
/* old gsl: */
/* gcc --std=gnu99 -I/work/07833/tg871330/stampede3/software/gsl-2.6/include -I/work/07833/tg871330/stampede3/software/hmpdf/include -o make_map_fsky_res_Nz_NM_pixgrid_seed_onethread_stampede make_map_fsky_res_Nz_NM_pixgrid_seed_onethread_stampede.c -L/work/07833/tg871330/stampede3/software/hmpdf/ -lhmpdf */
/*! [compile] */

#include <stdio.h>
#include "/scratch/07833/tg871330/software_scratch/hmpdf/include/utils.h"
#include "/scratch/07833/tg871330/software_scratch/hmpdf/include/hmpdf.h"

int make_tSZ_map(double map_fsky, double pixel_side, int N_z, int N_Mass,int pixelgrid, int map_seed, char* out_directory, char* out_suffix, int n_maps)
{
    /* get a new hmpdf_obj */
    hmpdf_obj *d = hmpdf_new();
    if (!(d))
        return -1;

    /* initialize */
    if (hmpdf_init(d,"/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/num_convergence/fid_cosmo.ini" , hmpdf_tsz,
                   hmpdf_map_fsky, map_fsky,
               hmpdf_pixel_side, pixel_side,
               hmpdf_N_z, N_z,
               hmpdf_N_M, N_Mass,
               hmpdf_map_pixelgrid, pixelgrid,
                hmpdf_map_seed, map_seed,
                hmpdf_N_threads, 1,
                hmpdf_z_min, 0.005,
               hmpdf_verbosity, 5))
        return -1;

    /* get n_maps and save with float precision */
    
    for (int ii=0; ii<n_maps; ii++){
    double *map;
    long Nside;
   
     if (hmpdf_get_map(d, &map, &Nside, 1))
          return -1;
    
    /* save with float precision */
    float *float_map= malloc(Nside*Nside*sizeof(float));
    for (int jj=0; jj<Nside*Nside; jj++){
         float_map[jj]=(float)map[jj];
         }
    
    /* save to a binary file */
    char map_file_name[512];
    sprintf(map_file_name,"%s/%s_%s_%d_%d%s", out_directory, "fid", out_suffix, map_seed, ii, ".bin");
    printf(map_file_name);
    
    FILE *fp = fopen(map_file_name, "wb");
    fwrite(float_map, sizeof(float), Nside*Nside, fp);
    fclose(fp);
    
    /* free both arrays */
    free(map);
    free(float_map);
      }
    
    /* free memory associated with the hmpdf_obj */
    if (hmpdf_delete(d))
        return -1;
    return 0;
    }

int main(int argc, char **argv){

    if (argc!=10){
        printf("You need to specify fsky, pixel resolution, Nz, NM, pixgrid, random seed, out directory, suffix and number of maps!\n");
        return -1;
    }
    double fsky=atof(argv[1]);
    double pixel_res=atof(argv[2]);
    int Nz=atoi(argv[3]);
    int NMass=atoi(argv[4]);
    int pixgrid=atoi(argv[5]);
    int map_seed=atoi(argv[6]);
    char *out_dir=argv[7];
    char *out_suff=argv[8];
    int n=atoi(argv[9]);

    if (make_tSZ_map(fsky, pixel_res, Nz, NMass, pixgrid, map_seed, out_dir, out_suff, n)){
    fprintf(stderr, "failed\n");
    return -1;
    }
    else{
    return 0;
    }
}
