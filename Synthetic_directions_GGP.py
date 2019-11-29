import numpy as np
import matplotlib.pyplot as plt
import PSV_dir_predictions as psv
import pmagpy.pmag as pmag
from numpy import random



def simulations_GGP(lat,GGPmodel,Nsim,plot=True):
    """ Gerenates four arrays that are directions (dec, inc), x and y equal area coordinates of rotated directions (x,y), 
    rotated directions (dec,inc), x and y equal area coordinates of rotated directions

    input 
    - lat latitude you want generate the directions list
    - GGPmodel: dictionary like GGP_QC96_GAD = {"g10" : -30, "g20" : 0.0, "g30" : 0.0, "sig10" : 3.0, "sig11": 3.0,
            "sig20" : 1.3 , "sig21" : 4.3, "sig22" : 1.3, "alpha": 27.7, "beta": 1.0, "name":'QC96_GAD'}
    - Nsim number of simulations you want

    output:
    lista_sim, lista_xy, lista_r, lista_xyr """
    
    dec_r=180
    pol=-GGPmodel['g10']/abs(GGPmodel['g10'])
    if pol<0:
        dec_r=0
    lista_sim = dii_fromGGPmodels(Nsim,lat,GGPmodel)
    m = psv.m_TAF(GGPmodel, lat)
    Binc = np.rad2deg(np.arctan(m[2]/abs(m[0])))
    lista_r = np.zeros(np.shape(lista_sim))
    
    for j in range(len(lista_sim)):
        lista_r[j,0],lista_r[j,1] = pmag.dotilt(lista_sim[j,0],lista_sim[j,1],dec_r,(90-Binc))
    lista_xyr = np.zeros([len(lista_sim),2])
    lista_xy = np.zeros([len(lista_sim),2])
    for j in range(len(lista_sim)):
        lista_xyr[j,0:2] = pmag.dimap(lista_r[j,0],lista_r[j,1])
        lista_xy[j,0:2] = pmag.dimap(lista_sim[j,0],lista_sim[j,1])
    if plot==True:
        plot_syn_directions(lista_sim, lista_xy, lista_r, lista_xyr,GGPmodel['name'],lat)
    return lista_sim, lista_xy, lista_r, lista_xyr

def dii_fromGGPmodels(N,L,GGPmodel): 
    
    
    """ 
    generates a set of vectors (declination, inclination and intensity)
    Modified from PMAGPY (tk03.py) - Generates set of directions with N data for a value of latitude L.
	N = number of directions, L = Latitude
	g10 = GGPmodel['g10']
    g20 = GGPmodel['g20']
    g30 = GGPmodel['g30']
    sig10 = GGPmodel['sig10']
    sig11 = GGPmodel['sig11']
    sig20 = GGPmodel['sig20']
    sig21 = GGPmodel['sig21']
    sig22 = GGPmodel['sig22']
    
    alpha = GGPmodel['alpha']
    beta = GGPmodel['beta']
    
    """

    D,R=0.,0
    cnt=1
    Imax=0
    d_i_i= np.zeros((N,3))
    for k in range(N): 
        gh=GGPgh(8,GGPmodel)[0] #this function returns gh and sigmas, then we take only gh 
        long=random.randint(0,360) # get a random longitude, between 0 and 359
        vec= pmag.getvec(gh,L,long)  # send field model and lat to getvec - the vector has also the values of intensity
        if vec[2]>=Imax:
            vec[0]+=D
            if k%2==0 and R==1:
                vec[0]+=180.
                vec[1]=-vec[1]
            if vec[0]>=360.:vec[0]-=360.
            d_i_i[k,0] = vec[0] #dec
            d_i_i[k,1] = vec[1] #inc
            d_i_i[k,2] = vec[2] #intensity
    return d_i_i      

def GGPgh(terms,GGPmodel):
    
    """
    Modified from pmagpy (mktk03(terms,seed,G2,G3))
    generates a list of gauss coefficients drawn from the a given GGP model
    where:
    g10 = GGPmodel['g10']
    g20 = GGPmodel['g20']
    g30 = GGPmodel['g30']
    sig10 = GGPmodel['sig10']
    sig11 = GGPmodel['sig11']
    sig20 = GGPmodel['sig20']
    sig21 = GGPmodel['sig21']
    sig22 = GGPmodel['sig22']
    
    alpha = GGPmodel['alpha']
    beta = GGPmodel['beta']
    
    """
    g10 = GGPmodel['g10']
    g20 = GGPmodel['g20']
    g30 = GGPmodel['g30']
    sig10 = GGPmodel['sig10']
    sig11 = GGPmodel['sig11']
    sig20 = GGPmodel['sig20']
    sig21 = GGPmodel['sig21']
    sig22 = GGPmodel['sig22']
    
    alpha = GGPmodel['alpha']
    beta = GGPmodel['beta']

    gh=[]
    all_s = []
    
    s = s_lm(1,0,alpha,beta,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)
    gnew=random.normal(g10,s)
    
    gh.append(gnew)
    all_s.append(s)
    
    s = s_lm(1,1,alpha,beta,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)
    #mean g11 = 0
    gh.append(random.normal(0,s))
    all_s.append(s)
    gnew=gh[-1]
    #mean h11 = 0
    gh.append(random.normal(0,s))
    all_s.append(s)
    hnew=gh[-1]
    for l in range(2,terms+1):
        for m in range(l+1):
            OFF=0.0
            if l==2 and m==0:OFF=g20
            if l==3 and m==0:OFF=g30
            s = s_lm(l,m,alpha,beta,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)
            gh.append(random.normal(OFF,s))
            all_s.append(s)
            gnew=gh[-1]
            if m==0:
                hnew=0
            else:
                gh.append(random.normal(0,s))
                all_s.append(s)
                hnew=gh[-1]
    return gh, all_s
def s_lm(l,m,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
	
    """
    Standard deviation for each gauss coefficient 
    l,m are the degree
    alpha is the alpha factor since CP88
    beta is the factor from TK03
    sig10_2 is the squared standard deviation for g10, which for CP88 is different
    sig10 = 3 then sig10_2=3*3=9 for CP88 model
    if sig10_2 is zero, then it will be calculated by the same equation as non-dipolar coefficients
    """
    
    c_a = 0.547
    if ((l-m)/2. - int((l-m)/2)) == 0:
        
        s_lm2 = ((c_a**(2*l))*(alpha**2))/((l+1)*(2*l+1))
        
    else:
        
        s_lm2 = (c_a**(2*l))*((alpha*beta)**2)/((l+1)*(2*l+1))
    if (l==1 and m==0):
        if (sig10_2>0):
            s_lm2 = sig10_2
    if (l==1 and m==1):
        if (sig11_2>0):
            s_lm2 = sig11_2
    if (l==2 and m==0):
        if (sig20_2>0):
            s_lm2 = sig20_2
    if (l==2 and m==1):
        if (sig21_2>0):
            s_lm2 = sig21_2
    if (l==2 and m==2):
        if (sig22_2>0):
            s_lm2 = sig22_2
        
    return np.sqrt(s_lm2)

def plot_syn_directions(lista_sim, lista_xy, lista_r, lista_xyr,name = 'Dir',lat = ' '):
    
    fig = plt.figure(figsize=[13,6], dpi=80)
            
    plt.subplot(121)
    xb=np.arange(-1,1.001,0.001)
    ybn=-np.sqrt(abs(1-xb**2))
    ybp=np.sqrt(abs(1-xb**2))
    plt.plot(xb,ybn, '--',c='0.5')
    plt.plot(xb,ybp,'--',c='0.5')
    plt.plot(0,0,'+',c='k')
    plt.axis('off', aspect='equal')
    
    for i in range(len(lista_sim)):
        if lista_sim[i,1]>=0:
            plt.plot(lista_xy[i,0],lista_xy[i,1],'o',c='0.5',fillstyle='full', markersize=5)
    for i in range(len(lista_sim)):
        if lista_sim[i,1]<0:
            plt.plot(lista_xy[i,0],lista_xy[i,1],'o',c='0.5',fillstyle='none', markersize=5)
    
    plt.text(0.0,-0.95,'Lat=%i$^\circ$'%lat,horizontalalignment='center')
    plt.text(0.0,-0.80,name,horizontalalignment='center')
    plt.text(0.0,-0.70,'Synthetic Directions', horizontalalignment='center')
                
         
    plt.subplot(122)
    plt.plot(xb,ybn, '--',c='0.5')
    plt.plot(xb,ybp,'--',c='0.5')
    plt.plot(0,0,'+',c='k')
    plt.axis('off', aspect='equal')
    for i in range(len(lista_r)):
        if lista_r[i,1]>=0:
            plt.plot(lista_xyr[i,0],lista_xyr[i,1],'o',c='0.5',fillstyle='full', markersize=5)
    for i in range(len(lista_r)):
        if lista_r[i,1]<0:
            plt.plot(lista_xyr[i,0],lista_xyr[i,1],'o',c='0.5',fillstyle='none', markersize=5)
    plt.text(0.0,-0.95,'Lat=%i$^\circ$'%lat,horizontalalignment='center')
    plt.text(0.0,-0.80,name,horizontalalignment='center')
    plt.text(0.0,-0.70,'Rotated', horizontalalignment='center')         
     
    plt.show()        

def Med_desvxy_sim(lista_xy):
    """
    It calculates the mean x and y and the stardard deviation of x and y using a list of results
    in x,y coordinates (equal area coordinates)
    """
    xm = np.average(lista_xy[:,0])
    ym = np.average(lista_xy[:,1])
    
    xstd = np.std(lista_xy[:,0],ddof=1)
    ystd = np.std(lista_xy[:,1],ddof=1)
    
    return xm,ym,xstd,ystd

def covvxy_sim(lista_xy):
    """
    It calculates the covariance between x and y a using a list of results
    in x,y coordinates (equal area coordinates)
    """
    xm = np.average(lista_xy[:,0])
    ym = np.average(lista_xy[:,1])
    cov_xys = (sum((lista_xy[:,0]-xm)*(lista_xy[:,1]-ym))/len(lista_xy))
    
    return cov_xys
	
def det_cov_xy_sim(lista_xy):
    """
    It calculates the determinant of covariance matrix between x and y a using a list of results
    in x,y coordinates (equal area coordinates)
    """
    xm = np.average(lista_xy[:,0])
    ym = np.average(lista_xy[:,1])
	
    xstd = np.std(lista_xy[:,0],ddof=1)
    ystd = np.std(lista_xy[:,1],ddof=1)
	
    cov_xys = (sum((lista_xy[:,0]-xm)*(lista_xy[:,1]-ym))/len(lista_xy))
	
    det = (xstd**2)*(ystd**2) - cov_xys**2
    
    return det

def Adir(lista_xy):
    return(np.sqrt(det_cov_xy_sim(lista_xy)))
	
def Elon_linha_sim(lista_xy):
    """
    It calculates the new elongation using the ratio of y standard deviation /  x standard deviation using a list of results
    in x,y coordinates (equal area coordinates)
    """
   
    xstd = np.std(lista_xy[:,0],ddof=1)
    ystd = np.std(lista_xy[:,1],ddof=1)
	
       
    return ystd/xstd
	
def detcov_boot(lista_xy, NB):
    """Calculates de bootstrap 95 percent of confidence regions of the 
	Area of distribution det(cov)"""
    low = int(.025 * NB)
    high = int(.975 * NB)
    det_b = np.zeros(NB)
    for i in range(NB):
        new = resampling(lista_xy)
        det_b[i] = det_cov_xy_sim(new)        
    det_b.sort()
    #print(det_b)
    return det_b[low],det_b[high]

def Adir_boot(lista_xy, NB):
    detlow,detup = detcov_boot(lista_xy, NB)
    return np.sqrt(detlow),np.sqrt(detup)
    

def Elon_linha_boot(lista_xy, NB):
    """Calculates de bootstrap 95 percent of confidence regions of the 
    Elongation prime = sigmay/sigmax """
    low = int(.025 * NB)
    high = int(.975 * NB)
    E_b = np.zeros(NB)
    for i in range(NB):
        new = resampling(lista_xy)
        E_b[i] = Elon_linha_sim(new)        
    E_b.sort()
    #print(E_b)
    return E_b[low],E_b[high]

def std_boot(x, NB):
    """ Calculates the bootstraped limits for std error
    """
    low = int(.025 * NB)
    high = int(.975 * NB)
    std_b = np.zeros(NB)
    for i in range(NB):
        x_B = np.zeros(len(x))
        for k in range(len(x)):
            random.seed()
            ind = random.randint(len(x))
            x_B[k]=x[ind]
        std_b[i] = np.std(x_B,ddof=1)
    std_b.sort()
    return std_b[low],std_b[high]

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