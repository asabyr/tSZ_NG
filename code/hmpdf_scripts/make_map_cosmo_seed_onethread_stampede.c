/*! [compile] */
/*!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/07833/tg871330/software_scratch/hmpdf */
/*!gcc --std=gnu99 -I$TACC_GSL_INC -I/scratch/07833/tg871330/software_scratch/hmpdf/include -o make_map_cosmo_seed_onethread_stampede make_map_cosmo_seed_onethread_stampede.c -L/scratch/07833/tg871330/software_scratch/hmpdf/ -lhmpdf */
/*!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/07833/tg871330/stampede3/software/hmpdf  */
/*!gcc --std=gnu99 -I$TACC_GSL_INC -I/work/07833/tg871330/stampede3/software/hmpdf/include -o make_map_cosmo_seed_onethread_stampede make_map_cosmo_seed_onethread_stampede.c -L/work/07833/tg871330/stampede3/software/hmpdf/ -lhmpdf */
/* old gsl: */
/* gcc --std=gnu99 -I/work/07833/tg871330/stampede3/software/gsl-2.6/include -I/work/07833/tg871330/stampede3/software/hmpdf/include -o make_map_cosmo_seed_onethread_stampede make_map_cosmo_seed_onethread_stampede.c -L/work/07833/tg871330/stampede3/software/hmpdf/ -lhmpdf */
/*! [compile] */

#include <stdio.h>
#include "/scratch/07833/tg871330/software_scratch/hmpdf/include/utils.h"
#include "/scratch/07833/tg871330/software_scratch/hmpdf/include/hmpdf.h"

int make_tSZ_map(char *cosmo, int map_seed, char *out_directory, int n_maps)
{
    // ini file
    char *ini_dir = "/scratch/07833/tg871330/tSZ_maps/hmpdf_maps/Oc_s8/ini_files/";
    char cosmo_ini[512];    
    sprintf(cosmo_ini, "%s%s%s", ini_dir, cosmo, ".ini");
    printf(cosmo_ini);

    /* get a new hmpdf_obj */
    hmpdf_obj *d = hmpdf_new();
    if (!(d))
        return -1;

    /* initialize */
    if (hmpdf_init(d, cosmo_ini,
                   hmpdf_tsz,
                   hmpdf_map_fsky, 0.1,
                   hmpdf_pixel_side, 0.1,
                   hmpdf_N_z, 150,
                   hmpdf_N_M, 150,
                   hmpdf_map_pixelgrid, 3,
                   hmpdf_map_seed, map_seed,
                   hmpdf_N_threads, 1,
                   hmpdf_z_min, 0.005,
                   hmpdf_verbosity, 5))
        return -1;

    /* get the map and save with float precision */

    for (int ii = 0; ii < n_maps; ii++)
    {
        double *map;
        long Nside;
        if (hmpdf_get_map(d, &map, &Nside, 1))
            return -1;
        
        /* convert map to float */
        float *float_map= malloc(Nside*Nside*sizeof(float));        
        for (int jj = 0; jj < Nside * Nside; jj++)
        {
            float_map[jj] = (float)map[jj];
        }

        /* save map */
        char *num_settings = "NzNM150_0pt1fsky_0pr1arcmin";
        char map_file_name[512];
        sprintf(map_file_name, "%s/%s_%s_%d_%d%s", out_directory, cosmo, num_settings, map_seed, ii, ".bin");
        printf(map_file_name);

        FILE *fp = fopen(map_file_name, "w");
        fwrite(float_map, sizeof(float), Nside * Nside, fp);
        fclose(fp);

        /* free maps */
        free(map);
        free(float_map);
    }

    /* free memory associated with the hmpdf_obj */
    if (hmpdf_delete(d))
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    // input arguments
    if (argc != 5)
    {
        printf("You need to specify cosmology, seed, out directory, and number of maps!\n");
        return -1;
    }

    char *cosmo = argv[1];         // cosmo ini file
    int map_seed = atoi(argv[2]); // random seed
    char *out_dir = argv[3];      // directory for maps
    int n = atoi(argv[4]);        // number of maps

    // make a map
    if (make_tSZ_map(cosmo, map_seed, out_dir, n))
    {
        fprintf(stderr, "failed\n");
        return -1;
    }
    else
    {
        return 0;
    }
}
