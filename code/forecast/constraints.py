import numpy as np
from scipy.interpolate import *
import glob
import os
import copy
from inference_funcs import *
import sys
this_dir=os.path.dirname(os.path.abspath(__file__))
general_dir=this_dir.replace('forecast','general')
suite_code_dir=this_dir.replace('forecast','suite_code')
sys.path.append(general_dir)
sys.path.append(suite_code_dir)
from scipy.optimize import minimize
from help_funcs import get_cosmologies

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class Constraints:

    def __init__(self, prefix, stat, stat_dir,
                 interpolator=CloughTocher2DInterpolator,
                 fid_Nsims=34560,
                test_interpolator=False,
                fid_covariance_file='',
                some_ind=[],
                interpolate_logL=False,
                n_grid_points=1000, scale=1.0,
                cosmo_ini='',
                res_type='sigma', GP_params={}, fid_cov_mean=False,
                fid_string='Oc_0.264_s8_0.811',
                x_prior=np.array([0.21, 0.32]),
                y_prior=np.array([0.6, 1.0])):

        self.prefix=prefix #file prefix for stat files
        self.stat=stat #statistic (i.e part or full string: cl_peaks_MF_V0_V1_V2_minima_moments_sigma0_sigma1_S0_S1_S2_K0_K1_K2_K3)
        self.stat_dir=stat_dir #directory where the stat files are
        self.n_grid_points=n_grid_points #how many points on a grid over which to interpolate
        self.interpolator=interpolator #interpolator function
        self.fid_Nsims=fid_Nsims #number of fiducial sims for covariance
        self.test_interpolator=test_interpolator #whether to test how well interpolation works
        self.fid_covariance_file=fid_covariance_file #supply covariance file if different in naming format from other files (e.g if you're testing a different number of simulations )
        self.some_ind=some_ind #which indices to use for the stats (i.e. if you want contraints from only some part of the vector)
        self.interpolate_logL=interpolate_logL #interpolate logL instead of observable
        self.scale=scale #scale logLs (e.g. for realistic survey size)
        self.cosmo_ini=cosmo_ini #directory to .ini files, if need to use only specific cosmologies
        self.res_type=res_type #which residuals in interpolation to compute (sigma or percent)
        self.GP_params=GP_params #for GP, not currently used outside of just for fiducial interpolation test
        self.fid_cov_mean=fid_cov_mean # bool, True if the same file is used for the mean and covariance at fiducial cosmology

        self.fid_string=fid_string
        str_arr=np.array(fid_string.split("_"))
        self.param_x=str_arr[0]
        self.param_y=str_arr[2]
        self.fid_x=np.float(str_arr[1])
        self.fid_y=np.float(str_arr[3])

        self.x_min=x_prior[0]
        self.x_max=x_prior[1]
        self.y_min=y_prior[0]
        self.y_max=y_prior[1]

        ####################################################
        #              Set up grid/interpolator            #
        ####################################################

        #load fiducial value and covariance
        if len(self.fid_covariance_file)>0:
            #if specific file for fiducial covariance was specified
            fid_file_cov=np.load(fid_covariance_file, allow_pickle=True).item()

            #find the correct key
            for key, value in fid_file_cov.items():
                if "cov" in key:
                    cov_key=key
                    break

            #specifies if mean and covariance are the same file or not
            if self.fid_cov_mean==True:
                fid_file=np.load(fid_covariance_file, allow_pickle=True).item()
            else:

                fid_file=np.load(self.stat_dir+self.prefix+"_"+stat+f"_{self.fid_string}.npy", allow_pickle=True).item()

        else:
            fid_file_cov=np.load(self.stat_dir+self.prefix+"_"+stat+f"_{self.fid_string}.npy", allow_pickle=True).item()
            cov_key='cov_'+self.prefix+"_"+self.stat
            fid_file=np.load(self.stat_dir+self.prefix+"_"+stat+f"_{self.fid_string}.npy", allow_pickle=True).item()

        #if only using some part of the observable
        if len(self.some_ind)>0:
            self.fid_value=fid_file['avg_'+self.prefix+"_"+self.stat][self.some_ind]
            self.fid_covariance=fid_file_cov[cov_key][np.ix_(self.some_ind,self.some_ind)]
            self.fid_sigma=np.sqrt(np.diag(self.fid_covariance))
        else:
            #just using the entire vector
            self.fid_value=fid_file['avg_'+self.prefix+"_"+self.stat]
            self.fid_covariance=fid_file_cov[cov_key]
            self.fid_sigma=np.sqrt(np.diag(self.fid_covariance))
        # self.fid_covariance*=self.scale
        #get all stats & cosmologies
        if self.interpolate_logL==True:
            dx=delta_x(self.fid_value,self.fid_value)
            self.fid_logL=loglike(cov=self.fid_covariance, delta_x=dx,
                                 n_stat=len(self.fid_value), N_sims=self.fid_Nsims)
        self.compute_stat_suite()

        #make interpolator
        if self.interpolator!=GaussianProcessRegressor: #not set-up for GP
            self.interp_cosmo=self.interpolator(self.cosmologies, self.stat_values)

        if self.test_interpolator==True:
            self.test_interp_fid()
        elif self.test_interpolator=='all':
            self.test_interp_all()
        elif self.test_interpolator==False:
            pass
        else:
            sys.exit("choose testing intepolator for fiducial or all cosmologies")

    def likelihood_grid(self):

        self.set_up_grid()
        self.interpolate_grid()

        if self.interpolate_logL==True:
            self.logLs_arr=copy.deepcopy(self.grid_interp_stat)*self.scale
            self.Ls=np.exp(-0.5*self.logLs_arr)
        else:
            self.Ls()


    ####################################################
    #              Set up grid/interpolator            #
    ####################################################
    def compute_stat_suite(self):

        #get all cosmologies and stats
        if len(self.cosmo_ini)==0:
            all_stat_files=glob.glob(os.path.join(self.stat_dir,self.prefix+"_"+self.stat+"*npy"))
        else:
            cosmo_names=get_cosmologies(self.cosmo_ini)
            all_stat_files=[]
            for cosmo in cosmo_names:
                all_stat_files.append(os.path.join(self.stat_dir, self.prefix+"_"+self.stat+"_"+cosmo+".npy"))
        print(f"using {len(all_stat_files)} total cosmologies")
        self.N_cosmo=len(all_stat_files)

        #if covariance file is different, make sure the correct one is read if mean/covariance use the same file
        #essentially replace in the array of all files, the one correponding to fiducial with
        #self.fid_covariance_file
        if self.fid_cov_mean==True:
            assert(len(self.fid_covariance_file)>0)
            fid_file_existing=self.stat_dir+self.prefix+'_'+self.stat+f'_{self.fid_string}.npy'
            ind_fid=np.where(np.array(all_stat_files)==fid_file_existing)[0][0]
            all_stat_files[ind_fid]=copy.deepcopy(self.fid_covariance_file)

        #get dimensions
        avg_stat_file=np.load(all_stat_files[0], allow_pickle=True).item()
        if len(self.some_ind)>0:
            self.x_dim=len(avg_stat_file['avg_'+self.prefix+"_"+self.stat][self.some_ind])
        else:
            self.x_dim=len(avg_stat_file['avg_'+self.prefix+"_"+self.stat])

        self.cosmologies=np.zeros((len(all_stat_files),2))
        self.stat_values=np.zeros((len(all_stat_files),self.x_dim))
        self.stat_sigma=np.zeros((len(all_stat_files),self.x_dim))

        if self.interpolate_logL==True:
            self.stat_values=np.zeros((len(all_stat_files),1))

        #save average statistic
        for i in range(len(all_stat_files)):

            #get stat
            avg_stat_file=np.load(all_stat_files[i], allow_pickle=True).item()
            if len(self.some_ind)>0:
                obs_values=avg_stat_file['avg_'+self.prefix+"_"+self.stat][self.some_ind]
                obs_cov=avg_stat_file['cov_'+self.prefix+"_"+self.stat][self.some_ind]
                obs_sigma=np.sqrt(np.diag(obs_cov))
            else:
                obs_values=avg_stat_file['avg_'+self.prefix+"_"+self.stat]
                obs_cov=avg_stat_file['cov_'+self.prefix+"_"+self.stat]
                obs_sigma=np.sqrt(np.diag(obs_cov))

            #logL or observable
            if self.interpolate_logL==False:
                self.stat_values[i,:]=copy.deepcopy(obs_values)
                self.stat_sigma[i,:]=copy.deepcopy(obs_sigma)
            else:
                dx=delta_x(obs_values,self.fid_value)
                self.stat_values[i]=loglike(cov=self.fid_covariance, delta_x=dx,
                                 n_stat=self.x_dim, N_sims=self.fid_Nsims)

            file_name=os.path.basename(all_stat_files[i])
            X_Y_string=np.array(file_name.replace(".npy","").split("_"))
            X_ind=np.where(X_Y_string==f'{self.param_x}')[0]

            #get cosmo
            self.cosmologies[i,0]=float(X_Y_string[int(X_ind)+1]) #param x
            self.cosmologies[i,1]=float(X_Y_string[-1]) #param Y

    def set_up_grid(self):

        self.X_extent=self.x_max-self.x_min
        self.Y_extent=self.y_max-self.y_min
        self.tot_area=self.X_extent*self.Y_extent

        self.X_all=np.linspace(self.x_min, self.x_max, self.n_grid_points)
        self.Y_all=np.linspace(self.y_min, self.y_max, self.n_grid_points)
        self.grid_X, self.grid_Y=np.meshgrid(self.X_all, self.Y_all)

    def interpolate_grid(self):
        if self.interpolator!=GaussianProcessRegressor:
            self.grid_interp_stat=self.interp_cosmo(self.grid_X, self.grid_Y)
        #not set-up for GP


    def test_interp_fid(self):

        #test removing fiducial values & interpolating them instead
        ind_fid=np.where(self.cosmologies[:,0]==self.fid_x)

        rest_cosmologies=np.delete(self.cosmologies,ind_fid, axis=0)
        rest_stat_values=np.delete(self.stat_values,ind_fid, axis=0)

        # print(rest_cosmologies)
        if self.interpolator==CloughTocher2DInterpolator:
            test_interp_cosmo=self.interpolator(rest_cosmologies, rest_stat_values)
            fid_interp=test_interp_cosmo(self.fid_x, self.fid_y)
        elif self.interpolator==interp2d:
            test_interp_cosmo=self.interpolator(rest_cosmologies, rest_stat_values, kind='cubic')
            fid_interp=test_interp_cosmo(self.fid_x, self.fid_y)
        elif self.interpolator==RBFInterpolator:
            test_interp_cosmo=self.interpolator(rest_cosmologies, rest_stat_values)
            fid_interp=test_interp_cosmo(np.array([self.fid_x, self.fid_y]).reshape(2, -1).T)
        elif self.interpolator==GaussianProcessRegressor:
            kernel = self.GP_params['sigma'] * RBF(length_scale=self.GP_params['l_scale'],
                                                   length_scale_bounds=self.GP_params['length_scale_bounds'])
            print(self.GP_params['n_restarts_optimizer'])
            pipe = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=self.GP_params['n_restarts_optimizer']))
            pipe.fit(rest_cosmologies, rest_stat_values)  # apply scaling on training data
            fid_interp=pipe.predict(np.array([self.fid_x, self.fid_y]).reshape(1,2))



        #check residuals
        if self.interpolate_logL==False:
            if self.res_type=='sigma':
                self.residual=np.abs((fid_interp-self.fid_value)/self.fid_sigma)
            elif self.res_type=='percent':
                self.residual=(fid_interp-self.fid_value)/self.fid_value*100.
            print(f"residual:{self.residual}")
            print(f"max:{np.max(self.residual)}")
            print(f"mean:{np.mean(self.residual)}")
            print(f"median:{np.median(self.residual)}")
        elif self.interpolate_logL==True:
            print(self.fid_logL)
            print(fid_interp)
            print(f"residual:{(fid_interp/self.fid_logL-1)*100.0}")

    def test_interp_all(self):

        self.residuals=np.ones((self.N_cosmo,self.x_dim))*10**6

        for i in range(self.N_cosmo):

            rest_cosmologies=np.delete(self.cosmologies,i, axis=0)
            rest_stat_values=np.delete(self.stat_values,i, axis=0)

            if self.interpolator==RBFInterpolator:
                test_interp_cosmo=self.interpolator(rest_cosmologies, rest_stat_values)
                x=np.array([self.cosmologies[i,0], self.cosmologies[i,1]]).reshape(2, -1).T
                interp_value=test_interp_cosmo(x)
            elif self.interpolator==GaussianProcessRegressor:
                kernel = self.GP_params['sigma'] * RBF(length_scale=self.GP_params['l_scale'],
                                                    length_scale_bounds=self.GP_params['length_scale_bounds'])

                pipe = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel,
                                                                                n_restarts_optimizer=self.GP_params['n_restarts_optimizer']))
                pipe.fit(rest_cosmologies, rest_stat_values)  # apply scaling on training data
                x=np.array([self.cosmologies[i,0], self.cosmologies[i,1]]).reshape(1,2)
                interp_value=pipe.predict(x)
            else:
                test_interp_cosmo=self.interpolator(rest_cosmologies, rest_stat_values)
                interp_value=test_interp_cosmo(self.cosmologies[i,0], self.cosmologies[i,1])

            if self.res_type=='sigma':
                self.residuals[i,:]=(interp_value-self.stat_values[i])/self.stat_sigma[i]
            elif self.res_type=='percent':
                self.residuals[i,:]=(interp_value-self.stat_values[i])/self.stat_values[i]*100.0

        self.residuals=np.nan_to_num(self.residuals)

    ####################################################
    #              Now compute constraints             #
    ####################################################

    def Ls(self):

        #compute likelihood for every grid point
        self.logLs_arr=np.empty((self.n_grid_points,self.n_grid_points))
        self.Ls=np.empty((self.n_grid_points,self.n_grid_points))

        for j in range(self.n_grid_points):
            for i in range(self.n_grid_points):

                #if NaN just set loglikelihood to super high value
                if np.isnan(np.sum(self.grid_interp_stat[j,i,:]))==False:

                    dx=delta_x(self.grid_interp_stat[j,i,:],self.fid_value)
                    logL=loglike(cov=self.fid_covariance, delta_x=dx,
                                 n_stat=self.x_dim, N_sims=self.fid_Nsims)
                    self.logLs_arr[j,i]=logL*self.scale
                    self.Ls[j,i]=np.exp(-0.5*logL*self.scale)
                else:
                    # print("found NaN in stat")
                    self.logLs_arr[j,i]=10**6
                    self.Ls[j,i]=0.0

    def get_marginal_like(self):

        self.Ls_X=np.zeros(self.n_grid_points)
        self.Ls_Y=np.zeros(self.n_grid_points)

        for i in range(self.n_grid_points):
            self.Ls_X[i]=np.trapz(self.Ls[:,i],self.Y_all)
            self.Ls_Y[i]=np.trapz(self.Ls[i,:],self.X_all)

    def get_marginal_like_deg(self):

        self.Ls_X_deg=np.zeros(self.n_grid_points)
        self.Ls_Y_deg=np.zeros(int(self.n_grid_points**2))

        for i in range(self.n_grid_points):
            self.Ls_X_deg[i]=np.trapz(self.Sigma_Ls[:,i],self.Sigma_all)
        for j in range(int(self.n_grid_points**2)):
            self.Ls_Y_deg[j]=np.trapz(self.Sigma_Ls[j,:],self.X_all)

    def compute_profiles(self, plot=True):
        #unmarginalized error tests
        diff_x=np.abs(self.X_all-self.fid_x)
        diff_y=np.abs(self.Y_all-self.fid_y)

        ind_x=np.argmin(diff_x)
        ind_y=np.argmin(diff_y)

        self.Ls_X_profile=self.Ls[ind_y, :]
        self.Ls_Y_profile=self.Ls[:, ind_x]

        lims_X, _, _=get_like_level(self.Ls_X_profile)
        X_low, X_high=get_marginal_param_limits(self.Ls_X_profile, lims_X, self.grid_X[0,:])

        lims_Y, _, _=get_like_level(self.Ls_Y_profile)
        Y_low, Y_high=get_marginal_param_limits(self.Ls_Y_profile, lims_Y, self.grid_Y[:,0])

        if plot==True:
            plot_marginal(self.grid_X[0,:], self.Ls_X_profile, lims_X,
            X_low, X_high, self.fid_x)

            plot_marginal(self.grid_Y[:,0], self.Ls_Y_profile, lims_Y,
            Y_low, Y_high,self.fid_y)

        constraints_dict={}

        constraints_dict['1sigma_x']=np.array([self.fid_x-X_low[0], X_high[0]-self.fid_x])
        constraints_dict['2sigma_x']=np.array([self.fid_x-X_low[1], X_high[1]-self.fid_x])
        constraints_dict['1sigma_y']=np.array([self.fid_y-Y_low[0], Y_high[0]-self.fid_y])
        constraints_dict['2sigma_y']=np.array([self.fid_y-Y_low[1], Y_high[1]-self.fid_y])

        return constraints_dict


    def compute_marginal(self, plot=True):

        self.get_marginal_like()
        #compute marginal X
        lims_X, _, _=get_like_level(self.Ls_X)
        X_low, X_high=get_marginal_param_limits(self.Ls_X, lims_X, self.grid_X[0,:])

        X_centroid=get_centroid(self.Ls_X, self.grid_X[0,:])

        #compute marginal Y
        lims_Y, _, _=get_like_level(self.Ls_Y)
        Y_low, Y_high=get_marginal_param_limits(self.Ls_Y, lims_Y, self.grid_Y[:,0])

        Y_centroid=get_centroid(self.Ls_Y, self.grid_Y[:,0])

        if plot==True:
            plot_marginal(self.grid_X[0,:], self.Ls_X, lims_X,
            X_low, X_high, X_centroid)

            plot_marginal(self.grid_Y[:,0], self.Ls_Y, lims_Y,
            Y_low, Y_high, Y_centroid)

        constraints_dict={}
        constraints_dict['centroid_x']=X_centroid
        constraints_dict['centroid_y']=Y_centroid

        constraints_dict['1sigma_x']=np.array([X_centroid-X_low[0], X_high[0]-X_centroid])
        constraints_dict['2sigma_x']=np.array([X_centroid-X_low[1], X_high[1]-X_centroid])
        constraints_dict['1sigma_y']=np.array([Y_centroid-Y_low[0], Y_high[0]-Y_centroid])
        constraints_dict['2sigma_y']=np.array([Y_centroid-Y_low[1], Y_high[1]-Y_centroid])

        return constraints_dict

    def compute_contours(self, area=True):

        lims, k, ind_sort=get_like_level(self.Ls)

        sort_param_grid_x=self.grid_X.flatten()[ind_sort][::-1]
        sort_param_grid_y=self.grid_Y.flatten()[ind_sort][::-1]

        #compute 2D contour levels

        param_dict={}
        param_dict['1sigma_x']=sort_param_grid_x[:k[0]]
        param_dict['2sigma_x']=sort_param_grid_x[:k[1]]
        param_dict['1sigma_y']=sort_param_grid_y[:k[0]]
        param_dict['2sigma_y']=sort_param_grid_y[:k[1]]

        if area==True:
            self.contour_area_1sigma=k[0]*self.tot_area/self.n_grid_points**2.0
            self.contour_area_2sigma=k[1]*self.tot_area/self.n_grid_points**2.0

        return lims, param_dict

    def compute_deg_constraints(self, alpha=0.0, plot=True, path_to_figs=""):

        #not used
        # if alpha==0:

        #     alpha_fit=minimize(alpha_param,alpha_guess,
        #                        args=(self.fid_x, self.grid_X, self.grid_Y, self.Ls))
        #     alpha=alpha_fit.x

        self.Sigma=self.grid_Y*(self.grid_X/self.fid_x)**alpha
        self.Sigma_all=np.sort(self.Sigma.flatten())

        #get x and y params, and build interpolator
        params=np.zeros((int(self.n_grid_points**2.0),2))
        params[:,0]=self.grid_X.flatten()
        params[:,1]=self.Sigma.flatten()
        interp_Ls=self.interpolator(params, self.Ls.flatten())

        #make a new, asymmetric grid and interpolate Ls
        self.deg_grid_X,self.deg_grid_Y=np.meshgrid(self.X_all, self.Sigma_all)
        self.Sigma_Ls=np.nan_to_num(interp_Ls(self.deg_grid_X,self.deg_grid_Y))
        self.Sigma_Ls[self.Sigma_Ls<0]=0.0

        #get marginal likelihood
        self.get_marginal_like_deg()

        #compute marginal Y
        lims_Y_deg, _, _=get_like_level(self.Ls_Y_deg)
        Sigma_low, Sigma_high=get_marginal_param_limits(self.Ls_Y_deg, lims_Y_deg, self.deg_grid_Y[:,0])
        Sigma_centroid=get_centroid(self.Ls_Y_deg, self.Sigma_all)

        #compute marginal X
        lims_X_deg, _, _=get_like_level(self.Ls_X_deg)
        X_low, X_high=get_marginal_param_limits(self.Ls_X_deg, lims_X_deg, self.deg_grid_X[0,:])
        X_centroid=get_centroid(self.Ls_X_deg, self.X_all)

        if len(self.some_ind)>0:
            inds_str=str(self.some_ind[0])+"_"+str(self.some_ind[-1])
        else:
            inds_str='all_inds'

        if plot==True:
            plot_marginal(self.deg_grid_X[0,:], self.Ls_X_deg, lims_X_deg,
            X_low, X_high, X_centroid, save_plot=path_to_figs+"X_marginal"+self.prefix+"_"+self.stat+"_"+inds_str)
            plot_marginal(self.deg_grid_Y[:,0], self.Ls_Y_deg, lims_Y_deg,
            Sigma_low, Sigma_high, Sigma_centroid, save_plot=path_to_figs+"Sigma_marginal"+self.prefix+"_"+self.stat+"_"+inds_str)

        constraints_dict={}
        constraints_dict['alpha']=alpha
        constraints_dict['centroid_Sigma']=Sigma_centroid
        constraints_dict['1sigma_Sigma']=np.array([Sigma_centroid-Sigma_low[0], Sigma_high[0]-Sigma_centroid])
        constraints_dict['2sigma_Sigma']=np.array([Sigma_centroid-Sigma_low[1], Sigma_high[1]-Sigma_centroid])

        constraints_dict['centroid_X']=Sigma_centroid
        constraints_dict['1sigma_X']=np.array([X_centroid-X_low[0], X_high[0]-X_centroid])
        constraints_dict['2sigma_X']=np.array([X_centroid-X_low[1], X_high[1]-X_centroid])

        return constraints_dict
