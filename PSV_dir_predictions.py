import numpy as np
import math
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt


def P_lm(l,m,theta):
    """ this function calculates the Associated Legendre Functions
    3M - page: 441, equation B. 3. 2
    theta in radians - co-latitude"""
    A = np.sin(theta)**m/2**l
    sum_p_lm = 0.
    for t in range(0,int((l-m)/2)+1):
        #t vai de zero ate o ultimo inteiro menor que l-m /2   
        
        B = (-1)**t*math.factorial(2*l-2*t)
        C = math.factorial(t)*math.factorial(l-t)*math.factorial(l-m-2*t)
        D = l - m - 2*t
        sum_p_lm += (B/C)*np.cos(theta)**D
    return A*sum_p_lm
      
	
def dP_lm_dt(l,m,theta):
    """ this function calculates the derivative of Associated Legendre Functions related to theta
	theta in radians - co-latitude
    """
    A = (np.sin(theta)**m)/(2.**l)

    if m == 0:
        A2 = 0.
    else:
        A2 = m*(np.sin(theta)**(m-1))*np.cos(theta)/2**l
    sum_p_lm = 0.
    sum_p_lm2 = 0.
      
    for t in range(0,int((l-m)/2)+1):
        #t vai de zero ate o ultimo inteiro menor que l-m /2   
        B = ((-1)**t)*math.factorial(2*l-2*t)
        C = math.factorial(t)*math.factorial(l-t)*math.factorial(l-m-2*t)
        D = l - m - 2*t
        
        sum_p_lm += (B/C)*(np.cos(theta)**D)
    
    for t in range(0,int((l-m)/2)+1):
        #t vai de zero ate o ultimo inteiro menor que l-m /2   
        
        B = ((-1)**t)*math.factorial(2*l-2*t)
        C = math.factorial(t)*math.factorial(l-t)*math.factorial(l-m-2*t)
        D = l - m - 2*t
        
        sum_p_lm2 += np.sin(theta)*(B*D/C)*((np.cos(theta))**(D-1))
		
    return (A2*sum_p_lm - A*sum_p_lm2)
	
def s_lm2(l,m,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
	
    """
    Variance each gauss coefficient 
    l,m are the degree
    alpha is the alpha factor since CP88
    beta is the factor from TK03 (it was 3.8 for TK03, but can be used as a generic one)
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
            #print('sig10=%.2f' % np.sqrt(sig10_2))
            s_lm2 = sig10_2
    if (l==1 and m==1):
        if (sig11_2>0):
            #print('sig11=%.2f' % np.sqrt(sig11_2))
            s_lm2 = sig11_2
    if (l==2 and m==0):
        if (sig20_2>0):
            #print('sig20=%.2f' % np.sqrt(sig20_2))
            s_lm2 = sig20_2
    if (l==2 and m==1):
        if (sig21_2>0):
            #print('sig21=%.2f' % np.sqrt(sig21_2))
            s_lm2 = sig21_2
    if (l==2 and m==2):
        if (sig22_2>0):
            #print('sig22=%.2f' % np.sqrt(sig22_2))
            s_lm2 = sig22_2
        
    return s_lm2


def sig_br2(l,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    l is the maximum degree for calculating, theta in radians - co-latitude
    The variance in r direction - propagation of error following that each component of the magenetic field
    is a linear combination of the gauss coeficients glm and hlm
    """
    sum_l = 0
    
    #print(l,alpha,beta,theta)
    for i in range(1,l+1):
        #print(i)
        A = ((i+1)**2)*s_lm2(i,0,alpha,beta, sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*(P_lm(i,0,theta)**2)
        
        #print(A)
        sum_m=0.
        for j in range(1,i+1): 
            #print (j)
            B = ((math.factorial(i-j))/(math.factorial(i+j)))*s_lm2(i,j,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*P_lm(i,j,theta)**2
            sum_m = sum_m + B
        #print(((i+1)**2)*2*sum_m)
        sum_l = sum_l + A + ((i+1)**2)*2*sum_m
    return sum_l

def sig_bt2(l,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    l is the maximum degree for calculating, theta in radians - co-latitude   
    The variance in Theta theta in radians - co-latitude)direction (through North-South direction) - propagation of error following that each component of the magenetic field
    is a linear combination of the gauss coeficients glm and hlm
    """
    sum_l = 0
    
    #print(l,alpha,beta,theta)
    for i in range(1,l+1):
        #print(i)
        A = s_lm2(i,0,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*(dP_lm_dt(i,0,theta)**2)
        
        #print(A)
        sum_m=0.
        for j in range(1,i+1): 
            #print (j)
            B = ((math.factorial(i-j))/(math.factorial(i+j)))*s_lm2(i,j,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*dP_lm_dt(i,j,theta)**2
            sum_m = sum_m + B
        #print(((i+1)**2)*2*sum_m)
        sum_l = sum_l + A + 2*sum_m
    return sum_l
	
def sig_bph2(l,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    l is the maximum degree for calculating, theta in radians - co-latitude
    The variance in Phi direction (through the East-West direction) - propagation of error following that each component of the magenetic field
    is a linear combination of the gauss coeficients glm and hlm
    """

    sum_l = 0
    if theta == 0:
        print('lat=90 will be aproximated to 89.999999')
        theta = np.deg2rad(90-89.999999)
    for i in range(1,l+1):
        sum_m=0.
        for j in range(1,i+1): 
            B = (j**2)*((math.factorial(i-j))/(math.factorial(i+j)))*s_lm2(i,j,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*(P_lm(i,j,theta)**2)
            sum_m = sum_m + B
        #print(((i+1)**2)*2*sum_m)
        sum_l = sum_l + 2*sum_m/(np.sin(theta)**2)
    return sum_l
def cov_br_bt(l,alpha, beta, theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
	"""
    l is the maximum degree for calculating, theta in radians - co-latitude
    Calculates the covaricance between Br and Btheta"""
	sum_l = 0.
	for i in range(1,l+1):
		A = s_lm2(i,0,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*P_lm(i,0,theta)*dP_lm_dt(i,0,theta)
		sum_m = 0.
		for j in range(1,i+1):
			B = (math.factorial(i-j)/math.factorial(i+j))*s_lm2(i,j,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*P_lm(i,j,theta)*dP_lm_dt(i,j,theta)
			sum_m += B
		sum_l = sum_l -(i+1)*(A+2*sum_m)
	return sum_l 

def Cov(alpha,beta,lat, degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    Covariance Matrix for a given GGP model in a latitude = lat in degrees
    degree is the maximum degree of gaussian coeficients"""
    theta = np.deg2rad(90-lat)
    Cov = np.zeros([3,3])
    Cov[0,0] = sig_bt2(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[0,2] = cov_br_bt(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[1,1] = sig_bph2(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    Cov[2,0] = cov_br_bt(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[2,2] = sig_br2(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    
    return Cov

def Cov_modelo(GGPmodel,lat, degree):
    """ calculates the covariance matrix reading the Model dictionary and Latitude in degrees"""
    g10 = GGPmodel['g10']
    g20 = GGPmodel['g20']
    g30 = GGPmodel['g30']
    sig10_2 = GGPmodel['sig10']**2
    sig11_2 = GGPmodel['sig11']**2
    sig20_2 = GGPmodel['sig20']**2
    sig21_2 = GGPmodel['sig21']**2
    sig22_2 = GGPmodel['sig22']**2
    
    alpha = GGPmodel['alpha']
    beta = GGPmodel['beta']
    
    theta = np.deg2rad(90-lat)
    Cov = np.zeros([3,3])
    Cov[0,0] = sig_bt2(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[0,2] = cov_br_bt(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[1,1] = sig_bph2(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    Cov[2,0] = cov_br_bt(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[2,2] = sig_br2(degree,alpha,beta,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    return Cov
	
def xy_eq2u_3D(x,y,hem=1):
    """From a given x, y pair from an equal area plot I find u from 3D coordinate system
    the vector u = (u1,u2,u3) is unitary (size 1) and points to Z vertical
    this function returns one array of lenght 3"""
    u=np.zeros(3)
    if x==0 and y==0:
        u[0] = 0.
        u[1] = 0.
    else:
        u[0]=y*np.sqrt(1 - (1 - x**2 - y**2)**2)/np.sqrt(x**2 + y**2)
        u[1]=x*np.sqrt(1 - (1 - x**2 - y**2)**2)/np.sqrt(x**2 + y**2)
    if hem==1:
        u[2] = 1 - x**2 - y**2
    if hem == -1:
        u[2] = -(1 - x**2 - y**2)
    return u

def u3D_2_xyeq(v):
    """
    
    for a given vector in 3D u = (u0,u1,u2)
    it returns the x and y of equal area projection
    
    """
    u = v/np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    x = u[1]*np.sqrt(1 - np.abs(u[2]))/np.sqrt(u[0]**2 + u[1]**2)
    y = u[0]*np.sqrt(1 - np.abs(u[2]))/np.sqrt(u[0]**2 + u[1]**2)
    
    return x,y

def su(x,y,hem,Lamb,m_norm,m):
    """
	Returns the density function su from Khokhlov et al 2006 for a given x and y
	GGPmodel is a dictionary with the parameters of a zonal GGP
	degree - is the degree in which the covariance is calculated
	dx and dy are the space in X and Y axes of the equal area projection
	hem is the hemisphere side (1 means positive, -1 means negative) 
	"""
    u = np.zeros(3)
    if x==0 and y==0:
        u[0] = 0.
        u[1] = 0.
    else:
        u[0] = y*np.sqrt(1 - (1-x**2 - y**2)**2) / np.sqrt(x**2 + y**2)
        u[1] = x*np.sqrt(1 - (1-x**2 - y**2)**2) / np.sqrt(x**2 + y**2)
    if hem==1:
        u[2] = 1 - x**2 - y**2
    if hem == -1:
        u[2] = -(1 - x**2 - y**2)

    u_norm = np.sqrt(np.dot(np.matmul(Lamb,u),u))

    z = (np.dot(np.matmul(Lamb,m),u))/u_norm
    
    s_u1=np.exp(-0.5*m_norm**2)*np.sqrt(np.linalg.det(Lamb))/(4*np.pi*u_norm**3)
    
    s_u2=z*np.sqrt(2/np.pi) + np.exp(0.5*z**2)*(1+z**2)*(1+sp.special.erf(z/np.sqrt(2)))
    
    su = s_u1*s_u2
    
    return su

	
def su_GGPmodel(GGPmodel,lat,degree,dx,dy,hem):
    """
	Returns a map of the density function su from Khokhlov et al 2013
	GGPmodel is a dictionary with the parameters of a zonal GGP
	degree - is the degree in which the covariance is calculated
	dx and dy are the space in X and Y axes of the equal area projection
	hem is the hemisphere side (1 means positive, -1 means negative) 
	"""
    m = m_TAF(GGPmodel, lat)
    Cov = Cov_modelo(GGPmodel,lat,degree)
    Lamb = np.linalg.inv(Cov)

    m_norm = np.sqrt(np.dot(np.matmul(Lamb,m),m))
    
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)

    XX,YY = np.meshgrid(X,Y)
    s = np.zeros(np.shape(XX))
    
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(XX)[1]):
            if (XX[i,j]**2 + YY[i,j]**2)<=1 :
                s[i,j] = su(XX[i,j],YY[i,j],hem,Lamb,m_norm,m)
    
    return m,X,Y,XX,YY,s

def Rt(theta):
    """
    Rotation matrix of theta degrees ONLY along the plane north and vertical

    """
    R = np.zeros([3,3])
    R[0,0] = np.cos(np.deg2rad(theta))
    R[0,2] = np.sin(np.deg2rad(theta))
    R[1,1] = 1.0
    R[2,0] = -np.sin(np.deg2rad(theta))
    R[2,2] = np.cos(np.deg2rad(theta))
    
    return R
	
def su_GGPmodel_r(GGPmodel,lat,degree,XX,YY,hem):
    """
	Returns a map of the density function su from Khokhlov et al 2013
	rotated to the center of the projection
	GGPmodel is a dictionary with the parameters of a zonal GGP
	degree - is the degree in which the covariance is calculated
	dx and dy are the space in X and Y axes of the equal area projection
	hem is the hemisphere side (1 means positive, -1 means negative) 
    """
    m = m_TAF(GGPmodel, lat)
    Cov = Cov_modelo(GGPmodel,lat,degree)
    pol=-GGPmodel['g10']/abs(GGPmodel['g10'])
    #print(pol)
    #print(m[0]/abs(m[0]))
    B = np.sqrt(m[0]**2+m[1]**2+m[2]**2)
    m_r = np.zeros(3)
    m_r[2] = B #The total field in the vertical axis - because we want the rotated distribution
    Binc = np.rad2deg(np.arctan(m[2]/np.sqrt(m[0]**2+m[1]**2)))

    Cov_r = np.matmul(np.matmul(Rt(-pol*(90-Binc)),Cov),np.transpose(Rt(-pol*(90-Binc))))
    Lamb = np.linalg.inv(Cov_r)

    m_norm = np.sqrt(np.dot(np.matmul(Lamb,m_r),m_r))
    
    s = np.zeros(np.shape(XX))
    
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(XX)[1]):
            if (XX[i,j]**2 + YY[i,j]**2)<=1 :
                s[i,j] = su(XX[i,j],YY[i,j],hem,Lamb,m_norm,m_r)
    
    return m,s	

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

def s_lmGGP(terms,GGPmodel):
    
    """
    Modified from pmagpy (mktk03(terms,seed,G2,G3))
    generates a list of sigma l m  drawn from the a given GGP model
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

    all_s = []
    degrees = []
    
    s = s_lm(1,0,alpha,beta,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)
    deg = np.array([1,0])
    
    all_s.append(s)
    degrees.append(deg)
    
    s = s_lm(1,1,alpha,beta,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)
    deg = np.array([1,1])
    
    all_s.append(s)
    degrees.append(deg)
    
    deg = np.array([1,1])
    all_s.append(s)
    degrees.append(deg)
    
    for l in range(2,terms+1):
        for m in range(l+1):
            s = s_lm(l,m,alpha,beta,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)
            deg = np.array([l,m])
            all_s.append(s)
            degrees.append(deg)
            if m!=0:
                deg = np.array([l,m])
                all_s.append(s)
                degrees.append(deg)
    return all_s,degrees

def intxy(M,Integ):
    """
    Calculates the integral in the plane X,Y 
    The function is given by the M matrix and the Integ is a Matrix with zeros and dx*dy 
    In the area concenerd to intergrate:
    The integrate is calculated only inside the circle of radios 1 (equal area projection) 
    """
    integ = 0.
    integ = np.sum(M*Integ)
    
    return integ

def Med_desvxy_mod(sp,sn,XX,YY,Integ):
    """
    It calculates the mean x and y and the stardard deviation of x and y using a GGP model
    Considering directional distribution funtion results from the positive and negative 
    equal area projections sp and sn
    """
    normp = intxy(sp,Integ)
    normn = intxy(sn,Integ)
	
    xmp = intxy(XX*sp,Integ)
    xmn = intxy(XX*sn,Integ)
    xm = (xmp+xmn)/(normp+normn)
    ymp = intxy(YY*sp,Integ)
    ymn = intxy(YY*sn,Integ)
    ym = (ymp+ymn)/(normp+normn)
    xstd2p = (intxy((XX-xm)*(XX-xm)*sp,Integ))
    xstd2n = (intxy((XX-xm)*(XX-xm)*sn,Integ))
    xstd = np.sqrt((xstd2p+xstd2n)/(normp+normn))
    ystd2p = (intxy((YY-ym)*(YY-ym)*sp,Integ))
    ystd2n = (intxy((YY-ym)*(YY-ym)*sn,Integ))
    ystd = np.sqrt((ystd2p+ystd2n)/(normp+normn))
	
    return normp,normn,xm,ym,xstd,ystd
	
def covxy_mod(sp,sn,XX,YY,Integ):
    """
    It calculates the covariance between x and y (equal area) using a GGP model
    Considering directional distribution funtion results from the positive and negative 
    equal area projections sp and sn
    """
    normp = intxy(sp,Integ)
    normn = intxy(sn,Integ)
	
    xmp = intxy(XX*sp,Integ)
    xmn = intxy(XX*sn,Integ)
    xm = (xmp+xmn)/(normp+normn)
    ymp = intxy(YY*sp,Integ)
    ymn = intxy(YY*sn,Integ)
    ym = (ymp+ymn)/(normp+normn)
    
    covxyp = (intxy((XX-xm)*(YY-ym)*sp,Integ))
    covxyn = (intxy((XX-xm)*(YY-ym)*sn,Integ))
    covxy = (covxyp+covxyn)/(normp+normn)
	
	
    
	
    return covxy

def Med_desvxy_covxy_mod(sp,sn,XX,YY,Integ):
    """
    It calculates the mean x and y and the stardard deviation of x and y and covariance of x and y using a GGP model
    Considering directional distribution funtion results from the positive and negative 
    equal area projections sp and sn
    """
    normp = intxy(sp,Integ)
    normn = intxy(sn,Integ)
	
    xmp = intxy(XX*sp,Integ)
    xmn = intxy(XX*sn,Integ)
    xm = (xmp+xmn)/(normp+normn)
    ymp = intxy(YY*sp,Integ)
    ymn = intxy(YY*sn,Integ)
    ym = (ymp+ymn)/(normp+normn)
    xstd2p = (intxy((XX-xm)*(XX-xm)*sp,Integ))
    xstd2n = (intxy((XX-xm)*(XX-xm)*sn,Integ))
    xstd = np.sqrt((xstd2p+xstd2n)/(normp+normn))
    ystd2p = (intxy((YY-ym)*(YY-ym)*sp,Integ))
    ystd2n = (intxy((YY-ym)*(YY-ym)*sn,Integ))
    ystd = np.sqrt((ystd2p+ystd2n)/(normp+normn))
	
    covxyp = (intxy((XX-xm)*(YY-ym)*sp,Integ))
    covxyn = (intxy((XX-xm)*(YY-ym)*sn,Integ))
    covxy = (covxyp+covxyn)/(normp+normn)
	
    return normp,normn,xm,ym,xstd,ystd,covxy

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

def integ(dx,dy):
    """ It returns the Matrix Integ as the numerical integral of one*dx*dy 
    in the circle of inside of equal area, outside is zero"""
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    Integ = np.zeros(np.shape(XX)) #matrix that will be used every integration
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(YY)[1]):
            if np.sqrt((XX[i,j]+0.5*dx)**2+(YY[i,j]+0.5*dy)**2)<=1 and np.sqrt((XX[i,j]-0.5*dx)**2+(YY[i,j]-0.5*dy)**2)<=1:
                if np.sqrt((XX[i,j]-0.5*dx)**2+(YY[i,j]+0.5*dy)**2)<=1 and np.sqrt((XX[i,j]+0.5*dx)**2+(YY[i,j]-0.5*dy)**2)<=1:
                    #it will consider only squares inside the equal area circle
                    Integ[i,j] = dx*dy
    return Integ/np.sum(Integ), XX, YY, X, Y

def integ_x(dx,dy):
    """ It returns the Matrix Integ as the numerical integral of one*dx
    in the circle of inside of equal area, outside is zero"""
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    Integ = np.zeros(np.shape(XX)) #matrix that will be used every integration
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(YY)[1]):
            if np.sqrt((XX[i,j]+0.5*dx)**2+(YY[i,j]+0.5*dy)**2)<=1 and np.sqrt((XX[i,j]-0.5*dx)**2+(YY[i,j]-0.5*dy)**2)<=1:
                if np.sqrt((XX[i,j]-0.5*dx)**2+(YY[i,j]+0.5*dy)**2)<=1 and np.sqrt((XX[i,j]+0.5*dx)**2+(YY[i,j]-0.5*dy)**2)<=1:
                    #it will consider only squares inside the equal area circle
                    Integ[i,j] = dx
    return Integ/np.sum(Integ), XX, YY, X, Y

def integ_y(dx,dy):
    """ It returns the Matrix Integ as the numerical integral of one*dy
    in the circle of inside of equal area, outside is zero"""
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    Integ = np.zeros(np.shape(XX)) #matrix that will be used every integration
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(YY)[1]):
            if np.sqrt((XX[i,j]+0.5*dx)**2+(YY[i,j]+0.5*dy)**2)<=1 and np.sqrt((XX[i,j]-0.5*dx)**2+(YY[i,j]-0.5*dy)**2)<=1:
                if np.sqrt((XX[i,j]-0.5*dx)**2+(YY[i,j]+0.5*dy)**2)<=1 and np.sqrt((XX[i,j]+0.5*dx)**2+(YY[i,j]-0.5*dy)**2)<=1:
                    #it will consider only squares inside the equal area circle
                    Integ[i,j] = dy
    return Integ/np.sum(Integ), XX, YY, X, Y
	
	
def zonal(lat,g10=-18,G2=0.0,G3=0.0,G4=0.0,a_r=1.0):
    """
    Calculates the zonal field for a given latitude and returns the horizontal and vertical components Btetha, Br
    where a is the Earth radius and r is the distance you want to calculate from the center of earth
    a_r = a/r
    for example: a_r = 1.0 for the surface of Earth
    
    """
    
    
    Theta = 90-lat
    g20 = G2*g10
    g30 = G3*g10
    g40 = G4*g10

    costheta = np.cos(np.deg2rad(Theta))
    sintheta = np.sin(np.deg2rad(Theta))

    Br1 = 2*(a_r**3)*g10*costheta
    Br2 = (3*g20/2)*(a_r**4)*(3*(costheta**2) - 1)
    Br3 = (a_r**5)*(2*g30)*(5*(costheta**3) - 3*costheta)
    Br4 = (a_r**6)*(5*g40/8)*(35*(costheta**4) - 30*(costheta**2)+3)


    Bt1 = (a_r**3)*g10*sintheta
    Bt2 =  (a_r**4)*(3*g20)*sintheta*costheta
    Bt3 = (a_r**5)*(g30/2)*(15*sintheta*(costheta**2) - 3*sintheta)
    Bt4 = (a_r**6)*(g40/2)*(35*sintheta*(costheta**3) - 15*sintheta*costheta)


    Brtot = Br1+Br2+Br3+Br4 
    Bttot = Bt1+Bt2+Bt3+Bt4 
    
    Bx = -Bttot #bx eh bteta negativo
    By = 0
    Bz = -Brtot #bz eh br negativo
    H = np.sqrt(Bx**2 + By**2 )

    Inc = np.rad2deg(np.arctan(Bz/H))
    
    return Bttot, Brtot

def m_TAF(GGPmodel, lat):
    """
    Calculates the Time average field for a given GGP dictionay
    it returns one array m = (Bx,By,Bz)
    """
    m = np.zeros(3)
    Bteta,Br = zonal(lat,GGPmodel['g10'],GGPmodel['g20']/GGPmodel['g10'],GGPmodel['g30']/GGPmodel['g10'])
    m[0] = -Bteta
    m[1] = 0.
    m[2] = -Br
    return m	


def prediction_map_GGP_su_r(lat,GGPmodel,degree=8,dx=0.01,dy=0.01):
    """
    Predicts the map of su rotated to the vertical in a equalarea projection x , y or also called x_E and x_N from a GGP model and a latitude
    - GGP: a dictionary with the informations about the model
    - lat: latitude
    - degree: Maximum degree for calculating the covariance matrix of the field B (default is 8)
    - dx: spacement you want to map (default is 0.01)
    - dy: spacement you want to map (default is 0.01)
    it returns sp and sn, which are positive inclinations hemisphere and negative inclination hemisphere su maps of equal area projection 
    which are three columns arryas with x, y and su
    """
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    
   
    hem=1
    m,sp = su_GGPmodel_r(GGPmodel, lat, degree, XX, YY, hem)
    m,sn = su_GGPmodel_r(GGPmodel, lat, degree, XX, YY, -hem)
       
    return sp, sn
    

def prediction_map_GGP_su(lat,GGPmodel,degree=8,dx=0.01,dy=0.01):
    """
    Predicts the map of su in a equalarea projection x , y or also called x_E and x_N from a GGP model and a latitude
    - GGP: a dictionary with the informations about the model
    - lat: latitude
    - degree: Maximum degree for calculating the covariance matrix of the field B (default is 8)
    - dx: spacement you want to map (default is 0.01)
    - dy: spacement you want to map (default is 0.01)
    it returns sp and sn, which are positive inclinations hemisphere and negative inclination hemisphere su maps of equal area projection 
    which are three columns arryas with x, y and su
    """
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    
   
    hem=1
    m,X,Y,XX,YY,sp = su_GGPmodel(GGPmodel, lat, degree, dx, dy, hem)
    m,X,Y,XX,YY,sn = su_GGPmodel(GGPmodel, lat, degree, dx, dy, -hem)
       
    return sp, sn
    
def prediction_x_y_std_E_A_GGP(lats,GGPmodel,imprima=True, degree=8,dx=0.01,dy=0.01,hem=1):
    """
    Predicts the x mean, y mean, std, cov from a GGP model and a set of latitudes
    or a single latitude as an array with len=1 -> [lat]
    It returns an array with 7 columns lat,x,y,sigmax,sigmay,E,Adir
    if imprima=True it will print the latitude at each calculation"""
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    
    Mod = np.zeros([len(lats),7])
    Mod[:,0] = lats 
    for i in range(len(lats)):
        if imprima==True:
            print('Calculating for the latitude:')
            print(lats[i])
        m,sp = su_GGPmodel_r(GGPmodel, lats[i], degree, XX, YY, hem)
        m,sn = su_GGPmodel_r(GGPmodel, lats[i], degree, XX, YY, -hem)
        
        Integ, XX, YY, X, Y = integ(dx,dy)
                
        normp,normn, xmm, ymm, xstdm, ystdm, covxym = Med_desvxy_covxy_mod(sp,sn,XX, YY, Integ)
        
        Mod[i,1] = xmm
        Mod[i,2] = ymm
        Mod[i,3] = xstdm
        Mod[i,4] = ystdm
        Mod[i,5] = (ystdm**2)/(xstdm**2)
        Mod[i,6] = np.sqrt((ystdm*xstdm)**2 - covxym**2)
        
    return Mod

def plotmap(sp, sn,GGP,lat,dx,dy):
    name = GGP['name']
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    minp = np.min(sp)
    minn = np.min(sn)
    if minp<minn:
        minimo = minp
    else:
        minimo = minn
    maxp = np.max(sp)
    maxn = np.max(sn)
    if maxp>maxn:
        maximo = maxp
    else:
        maximo = maxn
    fig = plt.figure(figsize=[13,6], dpi=80)
            
    plt.subplot(121)
    xb=np.arange(-1,1.01,0.01)
    ybn=-np.sqrt(abs(1-xb**2))
    ybp=np.sqrt(abs(1-xb**2))
    plt.plot(xb,ybn, '--',c='0.5')
    plt.plot(xb,ybp,'--',c='0.5')
    plt.plot(0,0,'+',c='k')
    plt.axis('off', aspect='equal')
    plt.contour(X,Y,sp,levels=np.linspace(minimo,maximo,8), zorder = 5020)
    plt.text(0.0,0.92,'Lat=%i$^\circ$'%lat,horizontalalignment='center')
    plt.text(0.0,0.85,name,horizontalalignment='center')
    plt.text(0.0,0.78,'Positive inclination', horizontalalignment='center')
                
         
    plt.subplot(122)
    plt.plot(xb,ybn, '--',c='0.5')
    plt.plot(xb,ybp,'--',c='0.5')
    plt.plot(0,0,'+',c='k')
    plt.axis('off', aspect='equal')
    plt.contour(X,Y,sn,levels=np.linspace(minimo,maximo,8),zorder = 5020)
    plt.text(0.0,0.92,'Lat=%i$^\circ$'%lat,horizontalalignment='center')
    plt.text(0.0,0.85,name,horizontalalignment='center')
    plt.text(0.0,0.78,'Negative inclination', horizontalalignment='center')         
     
    plt.show()
