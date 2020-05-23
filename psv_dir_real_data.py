import numpy as np
from numpy import random

def Med_desvxy_sim_corr(lista_xy,amedio):
    """
    It calculates the mean x and y and the stardard deviation of x and y using a list of results
    in x,y coordinates (equal area coordinates) from a rotated to the vertical distribution (with the expected direction in the center of projection)
    amedio is the mean alpha95 from the list of paleomagnetic data 
    """
    xm = np.average(lista_xy[:,0])
    ym = np.average(lista_xy[:,1])
    
    xstd = np.std(lista_xy[:,0],ddof=1)
    ystd = np.std(lista_xy[:,1],ddof=1)
    
    sw_n=(81./140)*amedio
    r = np.sqrt(1-np.cos(np.deg2rad(sw_n)))
    sig_w = r/np.sqrt(2)

    xstd_c = np.sqrt(xstd**2-sig_w**2)
    ystd_c = np.sqrt(ystd**2-sig_w**2)
    return xm,ym,xstd,ystd,xstd_c,ystd_c,sig_w

def std_boot_corr(lista_xy_a, NB):
    """ Calculates the bootstraped limits for std error with a column of alpha 95s
    """
    low = int(.025 * NB)
    high = int(.975 * NB)
    xstd_b = np.zeros(NB)
    ystd_b = np.zeros(NB)
    for i in range(NB):
        new = resampling(lista_xy_a)
        amedio = np.nanmean(new[:,2])
        if np.isnan(amedio)==True:
            amedio = 0.
        xm,ym,xstd,ystd,xstd_c,ystd_c,sig_w = Med_desvxy_sim_corr(new[:,0:2],amedio)
        xstd_b[i] = xstd_c   
        ystd_b[i] = ystd_c       
    xstd_b.sort()
    ystd_b.sort()
    return xstd_b[low],xstd_b[high],ystd_b[low],ystd_b[high]
	
def Adir_sim_corr(lista_xy,amedio):
    """
    It calculates the area of distribution Adir using a list of results
    in x,y coordinates (equal area coordinates) from a rotated to the vertical distribution (with the expected direction in the center of projection)
    """
    xm,ym,xstd,ystd,xstd_c,ystd_c,sig_w = Med_desvxy_sim_corr(lista_xy,amedio)
    Adir = np.sqrt((xstd_c**2)*(ystd_c**2))
    
    return Adir
	
def Elon_linha_sim_corr(lista_xy,amedio):
    """
    It calculates the new elongation using the ratio of y standard deviation^2 /  x standard deviation^2 using a list of results
    in x,y coordinates (equal area coordinates) from a rotated to the vertical distribution (with the expected direction in the center of projection)
    corrected from experimental error

    This is new after review of 10Ma paper
    """
   
    xm,ym,xstd,ystd,xstd_c,ystd_c,sig_w = Med_desvxy_sim_corr(lista_xy,amedio)
	
       
    return ystd_c**2/xstd_c**2
	
def Adir_boot_corr(lista_xy_a, NB):
    """Calculates de bootstrap 95 percent of confidence regions of the 
	Area of distribution Adir lista xy and a collumn of alphas"""
    low = int(.025 * NB)
    high = int(.975 * NB)
    adir_b = np.zeros(NB)
    for i in range(NB):
        new = resampling(lista_xy_a)
        amedio = np.nanmean(new[:,2])
        if np.isnan(amedio)==True:
            amedio = 0.
        adir_b[i] = Adir_sim_corr(new[:,0:2],amedio)        
    adir_b.sort()
    #print(det_b)
    return adir_b[low],adir_b[high]

def Elon_linha_boot_corr(lista_xy_a, NB):
    """Calculates de bootstrap 95 percent of confidence regions of the 
    Elongation prime = sigmay^2/sigmax^2 """
    low = int(.025 * NB)
    high = int(.975 * NB)
    E_b = np.zeros(NB)
    for i in range(NB):
        new = resampling(lista_xy_a)
        amedio = np.nanmean(new[:,2])
        if np.isnan(amedio)==True:
            amedio = 0.
        E_b[i] = Elon_linha_sim_corr(new[:,0:2],amedio)        
    E_b.sort()
    #print(E_b)
    return E_b[low],E_b[high]

def resampling(dataset):
    """For any array with N lines, it returns the an array of the same shape with random sampling of the N lines"""
    new_dat = np.zeros(np.shape(dataset)) 
    for k in range(np.shape(dataset)[0]):
        random.seed()
        ind = random.randint(0,np.shape(dataset)[0])
        new_dat[k,:]=dataset[ind,:]
    return new_dat

def mean_boot(x, NB):
    """ Calculates the bootstraped limits for mean value <x> of one dimensional data - x is an array
    """
    low = int(.025 * NB)
    high = int(.975 * NB)
    mean_b = np.zeros(NB)
    for i in range(NB):
        x_B = np.zeros(len(x))
        for k in range(len(x)):
            random.seed()
            ind = random.randint(len(x))
            x_B[k]=x[ind]
        mean_b[i] = np.mean(x_B)
    mean_b.sort()
    return mean_b[low],mean_b[high]
