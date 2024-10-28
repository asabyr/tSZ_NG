import numpy as np
import os

THETA_FID={'Oc':0.264,'s8':0.811}

def get_Oc(folder_name):
    elements=np.array(folder_name.split('_'))
    return float(elements[1])
def get_s8(folder_name):
    elements=np.array(folder_name.split('_'))
    return float(elements[-1])
def get_dtheta(high, low):
    
    high_Oc=get_Oc(os.path.basename(os.path.normpath(high['Oc'])))
    low_Oc=get_Oc(os.path.basename(os.path.normpath(low['Oc'])))
    
    high_s8=get_s8(os.path.basename(os.path.normpath(high['s8'])))
    low_s8=get_s8(os.path.basename(os.path.normpath(low['s8'])))
    
    diff_Oc=np.round(high_Oc-THETA_FID['Oc'],5)
    diff_s8=np.round(high_s8-THETA_FID['s8'],5)
    assert diff_Oc==np.round(THETA_FID['Oc']-low_Oc,5)
    assert diff_s8==np.round(THETA_FID['s8']-low_s8,5)
    
    dtheta={}
    dtheta['Oc']=diff_Oc
    dtheta['s8']=diff_s8
    return dtheta
    
def get_fisher_params(cov_x_y, chi2=2.3):

    sigma_x=np.sqrt(cov_x_y[0,0])
    sigma_y=np.sqrt(cov_x_y[1,1])
    sigma_x_y=cov_x_y[0,1]

    sqrt_element=np.sqrt((sigma_x**2-sigma_y**2)**2/4+cov_x_y[0,1]**2)

    a2=(sigma_x**2+sigma_y**2)/2.0+sqrt_element
    b2=(sigma_x**2+sigma_y**2)/2.0-sqrt_element
    tan2_theta=2*sigma_x_y/(sigma_x**2-sigma_y**2)
    
    #return a,b,theta,area
    return np.sqrt(a2), np.sqrt(b2), np.arctan(tan2_theta)/2.0, np.pi*chi2*np.sqrt(a2)*np.sqrt(b2)


def get_fisher_dirs(high_Oc='Oc_0.26664_s8_0.811', high_s8='Oc_0.264_s8_0.81911',
                            low_Oc='Oc_0.26136_s8_0.811', low_s8='Oc_0.264_s8_0.80289', 
                            root='/scratch/07833/tg871330/tSZ_maps/hmpdf_maps'):

    
    high_folder={}
    high_folder['Oc']=f'{root}/{high_Oc}/'
    high_folder['s8']=f'{root}/{high_s8}/'

    low_folder={}
    low_folder['Oc']=f'{root}/{low_Oc}/'
    low_folder['s8']=f'{root}/{low_s8}/'

    dtheta_values=get_dtheta(high_folder, low_folder)

    return high_folder, low_folder, dtheta_values
