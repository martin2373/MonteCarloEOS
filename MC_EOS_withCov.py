# -*- coding: utf-8 -*-
"""
Functions used for Monte Carlo fitting of various types of Equations of State
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import lmfit
import scipy.optimize as opt
import corner

#Birch-Murnaghan Third Order EOS
def BM3(x,a,b,c):
    P = ((3/2)*a*((c/x)**(7/3)-(c/x)**(5/3))*(1+(3/4)*(b-4)*((c/x)**(2/3)-1)))
    return P

#Vinet EOS
def Vinet(V,K0,Kp,V0):
    f = (V/V0)**(1/3)
    eta = (3/2)*(Kp-1)
    P = 3*K0*((1-f)/(f**2))*np.exp(eta*(1-f))
    return P

#Function to return a weighted residual for a BM3 function
def BM3Res_weight(params, V, P, sig_P, w):
    V0 = params['V0']
    K0 = params['K0']
    Kp = params['Kp']
    V = np.array(V)
    P = np.array(P)
    sig_P = np.array(sig_P)
    Pmodel = ((3/2)*K0*((V0/V)**(7/3)-(V0/V)**(5/3))*(1+(3/4)*(Kp-4)*((V0/V)**(2/3)-1)))
    return ((Pmodel-P)/(sig_P*w))

#Function to return a weighted residual for a Vinet function
def VinetRes_weight(params, V, P, sig_P, w):
    V0 = params['V0']
    K0 = params['K0']
    Kp = params['Kp']
    V = np.array(V)
    P = np.array(P)
    sig_P = np.array(sig_P)
    Pmodel = Vinet(V,K0,Kp,V0)
    return ((Pmodel-P)/(sig_P*w))

#Thermal Pressure 
def Pth(V,T,V0,theta0,gamma0,q,n):
    R = 8.31 #J/mol/K
    gamma = gamma0*((V/V0)**q)
    theta = theta0*np.exp((gamma0-gamma)/q) #K
    x = theta/T
    x300 = theta/300
    U300 = 9*n*R*((theta/8)+300*((300/theta)**3)*((x300**3)*((1/3)-(x300/8)+(((1/6)*(x300**2))/10)-(((1/30)*x300**4)/(7*4*3*2))))) #J
    UT = 9*n*R*((theta/8)+T*((T/theta)**3)*((x**3)*((1/3)-(x/8)+(((1/6)*(x**2))/10)-(((1/30)*x**4)/(7*4*3*2))))) #J
    Pth = ((gamma*(UT-U300))/(V*1E-30))*1E-9 #GPa
    return Pth

#Alternate Thermal Pressure
def Pth_q0q1(V,T,V0,theta0,gamma0,q0,q1,n):
    R = 8.31 #J/mol/K
    q = q0*(V/V0)**q1
    gamma = gamma0*((V/V0)**q)
    theta = theta0*np.exp((gamma0-gamma)/q) #K
    x = theta/T
    x300 = theta/300
    U300 = 9*n*R*((theta/8)+300*((300/theta)**3)*((x300**3)*((1/3)-(x300/8)+(((1/6)*(x300**2))/10)-(((1/30)*x300**4)/(7*4*3*2))))) #J
    UT = 9*n*R*((theta/8)+T*((T/theta)**3)*((x**3)*((1/3)-(x/8)+(((1/6)*(x**2))/10)-(((1/30)*x**4)/(7*4*3*2))))) #J
    Pth = ((gamma*(UT-U300))/(V*1E-30))*1E-9 #GPa
    return Pth

#Thermal Pressure in the style of Tange
def Pth_TangeStyle(V,T,V0,theta0,gamma0,a,b,n):
    R = 8.31 #J/mol/K
    gamma = gamma0*(1 + a*(((V/V0)**b) - 1))
    theta = theta0*((V/V0)**(-(1 - a)*gamma0))*np.exp(-(gamma-gamma0)/b)
    x = theta/T
    x300 = theta/300
    U300 = 9*n*R*((theta/8)+300*((300/theta)**3)*((x300**3)*((1/3)-(x300/8)+(((1/6)*(x300**2))/10)-(((1/30)*x300**4)/(7*4*3*2))))) #J
    UT = 9*n*R*((theta/8)+T*((T/theta)**3)*((x**3)*((1/3)-(x/8)+(((1/6)*(x**2))/10)-(((1/30)*x**4)/(7*4*3*2))))) #J
    Pth = ((gamma*(UT-U300))/(V*1E-30))*1E-9 #GPa
    return Pth

#A Mie-Gruneisen-Debye EOS Using a BM3 300 K EOS
def MGD(V,T,K0,Kp,V0,theta0,gamma0,q,n):
    Ptherm = Pth(V,T,V0,theta0,gamma0,q,n) #GPa
    P300 = BM3(V,K0,Kp,V0) #GPa
    return Ptherm+P300

#A Mie-Gruneisen-Debye EOS Using a BM3 300 K EOS in the style of Tange
def MGD_Tange(V,T,K0,Kp,V0,theta0,gamma0,a,b,n):
    Ptherm = Pth_TangeStyle(V,T,V0,theta0,gamma0,a,b,n)
    P300 = BM3(V,K0,Kp,V0)
    return Ptherm+P300

#A Mie-Gruneisen-Debye EOS Using a Vinet 300 K EOS
def MGD_Vinet(V,T,K0,Kp,V0,theta0,gamma0,q,n):
    Ptherm = Pth(V,T,V0,theta0,gamma0,q,n) #GPa
    P300 = Vinet(V,K0,Kp,V0) #GPa
    return Ptherm+P300

#Function to return a weighted residual for an MGD function using a BM3 300 K EOS
def MGDres_weight_full(params,V,T,P,K0,Kp,V0,theta0,sig_P,n):
    gamma0 = params['gamma0']
    q = params['q']
    theta0 = params['theta0'] #added 2/10/22, comment out later if needed
    Pmodel = Pth(V,T,V0,theta0,gamma0,q,n) + BM3(V,K0,Kp,V0) #+ Pref #Pref added 3/31/22, comment out later if needed
    res = ((Pmodel - P)/sig_P)
    return res

#Function to return a weighted residual for an MGD function using a Vinet 300 K EOS
def MGDres_weight_Vinet(params,V,T,P,K0,Kp,V0,theta0,sig_P,n):
    gamma0 = params['gamma0']
    q = params['q']
    theta0 = params['theta0'] #added 2/10/22, comment out later if needed
    Pmodel = Pth(V,T,V0,theta0,gamma0,q,n) + Vinet(V,K0,Kp,V0)
    res = ((Pmodel - P)/sig_P)
    return res
###############################################################################
#Vibrational Component of Thermal Pressure
def Pth_v(V,T,V0,theta0,gamma0,q,n):
    R = 8.31 #J/mol/K
    gamma = gamma0*((V/V0)**q)
    theta = theta0*np.exp((gamma0-gamma)/q) #K
    x = theta/T
    x300 = theta/300
    U300 = 9*n*R*((theta/8)+300*((300/theta)**3)*((x300**3)*((1/3)-(x300/8)+(((1/6)*(x300**2))/10)-(((1/30)*x300**4)/(7*4*3*2))))) #J
    UT = 9*n*R*((theta/8)+T*((T/theta)**3)*((x**3)*((1/3)-(x/8)+(((1/6)*(x**2))/10)-(((1/30)*x**4)/(7*4*3*2))))) #J
    Pth_v = ((gamma*(UT-U300))/(V*1E-30))*1E-9 #GPa
    return Pth_v

#Electronic Component of Thermal Pressure
def Pth_e(V,T,V0,Pth_v,ge,B0,k,n):
    
    Vm = V*1E-30 #cubic angstroms to cubic m
    V0m = V0*1E-30 #cubic angstroms to cubic m
    
    intCve = (n*55.845E-3)*(1/2)*B0*((Vm/V0m)**k)*((T**2) - (300**2)) #integral of heat capacity with a molar mass term (kg/mol)
    
    Pth_e = (ge/Vm)*intCve*1E-9 #Electronic component of thermal pressure (GPa). 1E-9 term is to convert to GPa
    
    return Pth_e

#Thermal Pressure for Fe as used in Fei et al., 2016
def Pth_FeiElec(V,T,V0,theta0,gamma0,q,n):
    
    ge = 2
    B0 = 0.07
    k = 1.34
    
    Pth_vib = Pth_v(V,T,V0,theta0,gamma0,q,n)
    Pth_el = Pth_e(V,T,V0,Pth_v,ge,B0,k,n)
    Pth = Pth_vib + Pth_el #GPa
    return Pth

#Function to return a weighted residual for an MGD function in the style of Fei et al., 2016
def MGDres_weight_Vinet_elec(params,V,T,P,K0,Kp,V0,theta0,sig_P,n):
    gamma0 = params['gamma0']
    q = params['q']
    Pmodel = Pth_FeiElec(V,T,V0,theta0,gamma0,q,n) + Vinet(V,K0,Kp,V0)
    res = ((Pmodel - P)/sig_P)    
    return res

#Function to return an unweighted residual for an MGD function in the style of Fei et al., 2016
def MGD_Vinet_elec(V,T,K0,Kp,V0,theta0,gamma0,q,n):
    Ptherm = Pth_FeiElec(V,T,V0,theta0,gamma0,q,n) #GPa
    P300 = Vinet(V,K0,Kp,V0) #GPa
    return Ptherm+P300
###############################################################################
#Other potential EOS functions for use
def MGDres_weight_Vinet_q0q1(params,V,T,P,K0,Kp,V0,theta0,sig_P,n):
    gamma0 = params['gamma0']
    q0 = params['q0']
    q1 = params['q1']
    Pmodel = Pth_q0q1(V,T,V0,theta0,gamma0,q0,q1,n) + Vinet(V,K0,Kp,V0)
    res = ((Pmodel - P)/sig_P)**2
    return res

def HTBM(V,T,K0,Kp,V0,a0,a1,dK):
    KT = K0+(T-300)*dK
    VT = V0*np.exp((T-300)*a0 + 0.5*(T**2-300**2)*a1)
    Pth = ((3/2)*KT*((VT/V)**(7/3)-(VT/V)**(5/3))*(1+(3/4)*(Kp-4)*((VT/V)**(2/3)-1)))
    return Pth

def HTBMres(params,V,T,P,sig_P):
    K0 = params['K0']
    Kp = params['Kp']
    V0 = params['V0']
    a0 = params['a0']
    a1 = params['a1']
    dK = params['dK']
    KT = K0+(T-300)*dK
    VT = V0*np.exp((T-300)*a0 + 0.5*(T**2-300**2)*a1)
    Pth = ((3/2)*KT*((VT/V)**(7/3)-(VT/V)**(5/3))*(1+(3/4)*(Kp-4)*((VT/V)**(2/3)-1)))
    return ((Pth-P)/sig_P)**2
######################################################################################

#300 K EOS Fitting Routine
"""
This is the section where the bulk of the work gets done to fit a 300 K EOS (both BM3 and Vinet)
and propagate the uncertainty.

DataFile is a string that represents an excel spreadsheet with Pressure and Volume Data with uncertainties
EOSFile is a string that represents a text file with initial guesses for V0, K0, and K'
Varys is an array of True and/or False values that determine whether or not to Vary each of the 3 parameters. Ex) [True, True, False] would fit for V0 and K0, but would keep K' fixed.
steps is an integer value that defines the number of MC samples to take.
"""
def EOS_MC_300lmfit_weight(DataFile,EOSFile,Varys,steps,**kwargs):

    df = pd.read_excel(DataFile)
    P = df['Pressure'].to_numpy()
    V = df['Volume'].to_numpy()
    sig_P = df['sigmaP'].to_numpy()
    sig_V = df['sigmaV'].to_numpy()
    
    if 'sample' in kwargs:
        if kwargs['sample'][0] == False:
            sig_V = np.zeros_like(V)
            print('Not sampling V\n')
        if kwargs['sample'][1] == False:
            sig_P = np.ones_like(P)*1E-5
            print('Not sampling P\n')
        print(sig_V,sig_P)
    
    if 'unweighted' in kwargs:
        if kwargs['unweighted'] == True:
            sig_P = np.ones_like(P)
    
    if 'EOS' in kwargs:
        EOS = kwargs['EOS']
    else:
        EOSfile = open(EOSFile)
        line = EOSfile.readline()
        line = line.strip()
        EOS = line.split()
    V0_seed = float(EOS[0])
    K0_seed = float(EOS[1])
    Kp_seed = float(EOS[2])
    
    #If using a different reference:  Vref = [V0,P0]
    if 'Vref' in kwargs:
        V0_seed = kwargs['Vref'][0]
        P = P - kwargs['Vref'][1]
    
    params = lmfit.Parameters()
    params.add('V0',value=V0_seed,vary=Varys[0])
    params.add('K0',value=K0_seed,vary=Varys[1])
    params.add('Kp',value=Kp_seed,vary=Varys[2])
    
    K0 = []
    Kp = []
    V0 = []
    chisqr = []
    
    #Vinet Fits
    K0v = []
    Kpv = []
    V0v = []
    chisqrv = []
    
    #Weight by Number of points, Think about generalizing the category?
#    w = []
#    for ind in range(len(V)):
#        w.append(df.Paper.value_counts(normalize=True)[df['Paper'][ind]])
#    w = np.array(w)
    w = np.ones(len(V))
    
    for j in range(steps):
        V_mc = []
        P_mc = []
        for i in range (len(V)):
            V_mc.append(V[i] + sig_V[i]*np.random.randn())
            P_mc.append(P[i] + sig_P[i]*np.random.randn())
    
        fit = lmfit.minimize(BM3Res_weight, params, args=(V_mc,P_mc,sig_P,w))
        values = fit.params.valuesdict()
        K0.append(values['K0'])
        Kp.append(values['Kp'])
        V0.append(values['V0'])
        chisqr.append(fit.chisqr)
        
        ##### Vinet ###########################
        fit_vinet = lmfit.minimize(VinetRes_weight, params, args=(V_mc,P_mc,sig_P,w)) #added for Vinet
        values_vinet = fit_vinet.params.valuesdict() #Added for Vinet
        K0v.append(values_vinet['K0'])
        Kpv.append(values_vinet['Kp'])
        V0v.append(values_vinet['V0'])
        chisqrv.append(fit.chisqr)
        #######################################

    chisqr_min = np.amin(chisqr)
    min_index = np.argwhere(chisqr==chisqr_min)
    min_index = int(min_index[0])
    K0_min = (K0[min_index])
    Kp_min = (Kp[min_index])
    V0_min = (V0[min_index])
    
    P_best = BM3(V,K0_min,Kp_min,V0_min)
    
    K0_avg = np.mean(K0)
    Kp_avg = np.mean(Kp)
    V0_avg = np.mean(V0)
    
    K0_sd = np.std(K0)
    Kp_sd = np.std(Kp)
    V0_sd = np.std(V0)
    
    ###### Vinet ################
    K0_vinet_avg = np.mean(K0v)
    Kp_vinet_avg = np.mean(Kpv)
    V0_vinet_avg = np.mean(V0v)
    
    K0_vinet_sd = np.std(K0v)
    Kp_vinet_sd = np.std(Kpv)
    V0_vinet_sd = np.std(V0v)
    ############################
    
    V_calc = np.linspace(np.min(V),np.max(V),801)
    P_calc = BM3(V_calc, K0_avg,Kp_avg,V0_avg)
    
    fits = np.array([[K0_avg,Kp_avg,V0_avg],[K0_sd,Kp_sd,V0_sd]])
    mins = np.array([K0_min,Kp_min,V0_min])
    pairs = np.transpose(np.array([V0,K0,Kp]))
    cov = np.cov(np.array([V0,K0,Kp]))
    corr = np.corrcoef([V0,K0,Kp])
    
    fits_vinet = np.array([[K0_vinet_avg,Kp_vinet_avg,V0_vinet_avg],[K0_vinet_sd,Kp_vinet_sd,V0_vinet_sd]])
    pairs_vinet = np.transpose(np.array([V0v,K0v,Kpv]))
    cov_vinet = np.cov(np.array([V0v,K0v,Kpv]))
    corr_vinet = np.corrcoef([V0v,K0v,Kpv])

    
    plt.figure()
    plt.plot(P,V,'o')
    plt.plot(P_calc,V_calc,'-')
    plt.ylabel('Volume ($\AA^3$)')
    plt.xlabel('Pressure (GPa)')
    plt.tick_params(direction='in',right=True,top=True)
    
    plt.figure()
    plt.scatter(pairs[:,1],pairs[:,2],c=chisqr)
    plt.tick_params(direction='in',top=True,right=True)
    plt.xlabel('$K_0$ (GPa)')
    plt.ylabel('$K\'$')
    plt.colorbar()
    
    plt.figure()
    plt.plot(pairs[:,1],pairs[:,0],'ko')
    plt.tick_params(direction='in',top=True,right=True)
    plt.xlabel('$K_0$ (GPa)')
    plt.ylabel('$V_0 (\AA^3)$')
    
    plt.figure()
    plt.plot(pairs[:,2],pairs[:,0],'ko')
    plt.tick_params(direction='in',top=True,right=True)
    plt.xlabel('$K\'$ (GPa)')
    plt.ylabel('$V_0 (\AA^3)$')

    plt.figure()
    plt.plot(P,BM3(V,K0_avg,Kp_avg,V0_avg)-P,'o')
    plt.xlabel('Pressure (GPa)')
    plt.ylabel('Pressure Residual (GPa)')
    plt.tick_params(direction='in',top=True,right=True)
    
    if (Varys[0]==True):
        figure = corner.corner(pairs,labels=['$V_0 (\AA^3)$','$K_0 (GPa)$','K\''],truths=[V0_avg,K0_avg,Kp_avg],quantiles=(0.16,0.84),levels=(1-np.exp(-0.5),))
        ndim = 3
        axes = np.array(figure.axes).reshape((ndim,ndim))
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi,xi]
                ax.scatter(pairs[:,xi],pairs[:,yi],c=chisqr,s=10,alpha=0.5)
        plt.title('BM3')
        
        figure2 = corner.corner(pairs_vinet,labels=['$V_0 (\AA^3)$','$K_0 (GPa)$','K\''],truths=[V0_vinet_avg,K0_vinet_avg,Kp_vinet_avg],quantiles=(0.16,0.84),levels=(1-np.exp(-0.5),))
        ndim = 3
        axes2 = np.array(figure2.axes).reshape((ndim,ndim))
        for yi in range(ndim):
            for xi in range(yi):
                ax2 = axes2[yi,xi]
                ax2.scatter(pairs_vinet[:,xi],pairs_vinet[:,yi],c=chisqrv,s=10,alpha=0.5)
        plt.title('Vinet')


    if 'filename' in kwargs:
        pairs = pd.DataFrame(pairs,columns=['V0','K0','Kp'])
        pairs_vinet = pd.DataFrame(pairs_vinet,columns=['V0','K0','Kp'])
        output = kwargs['filename'] + '_MCFits.txt'
        output_vinet = kwargs['filename'] + '_MCFits_Vinet.txt'
        pairs.to_csv(output,float_format='%.2f',index=False)
        pairs_vinet.to_csv(output_vinet,float_format='%.2f',index=False)
        
        BM3Res = {'Pressure':P,
                  'Residual':BM3(df['Volume'],K0_avg,Kp_avg,V0_avg)-P,
                  'Paper': P}
        VinetRes = {'Pressure':P,
                    'Residual':Vinet(df['Volume'],K0_vinet_avg,Kp_vinet_avg,V0_vinet_avg)-P,
                    'Paper': df['Paper']}
        
        BM3Res = pd.DataFrame(BM3Res)
        VinetRes = pd.DataFrame(VinetRes)
        output_Res = kwargs['filename'] + '_BM3Resdiduals.txt'
        output_Res_vinet = kwargs['filename'] + '_VinetResiduals.txt'
        BM3Res.to_csv(output_Res,float_format='%.2f',index=False)
        VinetRes.to_csv(output_Res_vinet,float_format='%.2f',index=False)
        
        BM3file = open(kwargs['filename']+'_BM3Fits.txt','w')
        BM3file.write('%f\t%f\t%f\t%f\t%f\t%f\t422\t0.0\t1.0\t0.5\t2' % (V0_avg,V0_sd,K0_avg,K0_sd,Kp_avg,Kp_sd))
        BM3file.write('\n%f\t%f\t%f' % (corr[0,0],corr[0,1],corr[0,2]))
        BM3file.write('\n%f\t%f\t%f' % (corr[1,0],corr[1,1],corr[1,2]))
        BM3file.write('\n%f\t%f\t%f' % (corr[2,0],corr[2,1],corr[2,2]))
        BM3file.close()
        
        Vinetfile = open(kwargs['filename']+'_VinetFits.txt','w')
        Vinetfile.write('%f\t%f\t%f\t%f\t%f\t%f\t422\t0.0\t1.0\t0.5\t2' % (V0_vinet_avg,V0_vinet_sd,K0_vinet_avg,K0_vinet_sd,Kp_vinet_avg,Kp_vinet_sd))
        Vinetfile.write('\n%f\t%f\t%f' % (corr_vinet[0,0],corr_vinet[0,1],corr_vinet[0,2]))
        Vinetfile.write('\n%f\t%f\t%f' % (corr_vinet[1,0],corr_vinet[1,1],corr_vinet[1,2]))
        Vinetfile.write('\n%f\t%f\t%f' % (corr_vinet[2,0],corr_vinet[2,1],corr_vinet[2,2]))
        Vinetfile.close()
    
    return fits,mins,cov,corr,pairs,fits_vinet,cov_vinet,corr_vinet

#High T EOS Fitting Routine
"""
This is the section where the bulk of the work gets done to fit an MGD EOS (Using a BM3 Reference EOS!!!)
and propagate the uncertainty.

DataFile is a string that represents an excel spreadsheet with Pressure, Volume, and Temperature Data with uncertainties
EOSFile is a string that represents a text file with initial values for V0, sigma V0, K0, sigma K0, K', sigmaK', theta_0, sigmaTheta_0, gamma_0, q, and the atoms/unit cell.
Varys is an array of True and/or False values that determine whether or not to Vary each of the parameters q and gamma_0. Ex) [True, True] would fit for gamma_0 and q
steps is an integer value that defines the number of MC samples to take.
"""
def EOS_MC_MGDlmfit_fixed300_weight(DataFile,EOSFile,varys,steps,**kwargs):
    
    df = pd.read_excel(DataFile)
    V = df['Volume'].to_numpy() #angstroms3
    P = df['Pressure'].to_numpy() #GPa
    T = df['Temperature'].to_numpy() #K
    sig_V = df['sigmaV'].to_numpy()
    sig_T = df['sigmaT'].to_numpy()
    sig_P = df['sigmaP'].to_numpy() #GPa
#    sig_P = np.ones_like(V)
    
    if 'Pref' in kwargs:
        P = P-kwargs['Pref']
        Pref = 120
    else:
        Pref = 0
    
    EOSfile = open(EOSFile)
    line = EOSfile.readline()
    line = line.strip()
    EOS = line.split()
    V0_seed = float(EOS[0]) #angstroms3
    sigmaV0 = float(EOS[1])
    K0_seed = float(EOS[2]) #GPa
    sigmaK0 = float(EOS[3])
    Kp_seed = float(EOS[4])
    sigmaKp = float(EOS[5])
    theta0_seed = float(EOS[6]) #K
    sigmatheta0 = float(EOS[7])
    gamma0_seed = float(EOS[8])
    q_seed = float(EOS[9])
    at = float(EOS[10]) #atoms/unit cell
    n = at/(6.022E23) #mol per unit cell, change as necessary
    
    line = EOSfile.readline()
    line = line.strip()
    corr1 = line.split()
    line = EOSfile.readline()
    line = line.strip()
    corr2 = line.split()
    line = EOSfile.readline()
    line = line.strip()
    corr3 = line.split()
    corr300 = np.array([corr1,corr2,corr3])
    
    if 'cov' in kwargs:
        cov = kwargs['cov']
    else:
        cov = np.array([[sigmaV0**2,float(corr300[0,1])*sigmaV0*sigmaK0,float(corr300[0,2])*sigmaV0*sigmaKp],[float(corr300[1,0])*sigmaK0*sigmaV0,sigmaK0**2,float(corr300[1,2])*sigmaK0*sigmaKp],[float(corr300[2,0])*sigmaKp*sigmaV0,float(corr300[2,1])*sigmaKp*sigmaK0,sigmaKp**2]])   
    
    if 'sample' in kwargs:
        if kwargs['sample'][0] == False:
            sig_V = np.zeros_like(V)
            print('Not sampling V\n')
        if kwargs['sample'][1] == False:
            sig_P = np.ones_like(P)*1E-5
            print('Not sampling P\n')
        if kwargs['sample'][2] == False:
            sig_T = np.zeros_like(T)
            print('Not sampling T\n')
        if kwargs['sample'][3] == False:
            cov = np.array([[0,0,0],[0,0,0],[0,0,0]])
            print('Not sampling EOS\n')
    
    params = lmfit.Parameters()
    params.add('gamma0',value=gamma0_seed,vary=varys[0])
    params.add('q',value=q_seed,vary=varys[1])
    params.add('theta0',value=theta0_seed,vary=varys[2]) #added 2/10/22 for ppv testing, comment out if needed later
    
    gamma0 = []
    q = []
    theta0 = [] #added 2/10/22, comment out later if needed
    V0mc = []
    K0mc = []
    Kpmc = []
    theta0mc = []
    chisqr = []
    
    for j in range(steps):
        Vmc = []
        Pmc = []
        Tmc = []
        for i in range(len(V)):
            Vmc.append(V[i] + sig_V[i]*np.random.randn()) #randomly sample V
            Pmc.append(P[i] + sig_P[i]*np.random.randn()) #randomly sample P
            Tmc.append(T[i] + sig_T[i]*np.random.randn()) #randomly sample T
        V0temp, K0temp, Kptemp = np.random.multivariate_normal([V0_seed,K0_seed,Kp_seed],cov) #randomly sample K0 and K' from within covariance
#        K0mc.append(K0_seed + sigmaK0*np.random.randn())
#        Kpmc.append(Kp_seed + sigmaKp*np.random.randn())
#        V0mc.append(V0_seed + sigmaV0*np.random.randn()) #randomly sample and store V0
        K0mc.append(K0temp) #store 300K parameter
        Kpmc.append(Kptemp) #store 300K parameter
        V0mc.append(V0temp)
        theta0mc.append(theta0_seed + sigmatheta0*np.random.randn()) #randomly sample and store theta0
        
        #Weighted Non-linear least-squares fit (see lmfit doc)
        DoIt = lmfit.Minimizer(MGDres_weight_full,params,fcn_args=(np.array(Vmc),np.array(Tmc),np.array(Pmc),K0mc[j],Kpmc[j],V0mc[j],theta0mc[j],sig_P,n)) #Pref added 3/31/22, comment out if needed
        result = DoIt.minimize()
        values = result.params.valuesdict() #get fit parameter values 
        gamma0.append(values['gamma0']) #store fit parameter
        q.append(values['q']) #store fit parameter
        theta0.append(values['theta0']) #added 2/10/22 for ppv testing, comment out if needed later
        chisqr.append(result.chisqr)
        
    gamma0_avg = np.mean(gamma0)
    q_avg = np.mean(q)
    gamma0_std = np.std(gamma0)
    q_std = np.std(q)
    theta0_avg = np.mean(theta0)
    theta0_std = np.std(theta0)
    
    fits = np.array([[gamma0_avg,q_avg,theta0_avg],[gamma0_std,q_std,theta0_std]])
    pairs = np.transpose(np.array([gamma0,q,theta0,V0mc,K0mc,Kpmc]))
    Corr = np.corrcoef([gamma0,q,theta0])
    
    
    plt.figure()
    plt.scatter(gamma0,q,c=chisqr)
    plt.xlabel('$\gamma_0$')
    plt.ylabel('$q$')
    plt.tick_params(direction='in',top=True, right=True)
    plt.colorbar()
    
    plt.figure()
    plt.plot(K0mc,Kpmc,'ko')
    plt.xlabel('$K_0$')
    plt.ylabel('$K\'$')
    plt.tick_params(direction='in',top=True, right=True)

    plt.figure()
    plt.plot(V0mc,K0mc,'ko')
    plt.xlabel('$V_0$')
    plt.ylabel('$K_0$')
    plt.tick_params(direction='in',top=True, right=True)
    
    Vcalc = np.linspace(np.min(V),np.max(V),1000)
    Pcalc300 = BM3(Vcalc,K0_seed,Kp_seed,V0_seed)
    Pcalc1000 = MGD(Vcalc,1000,K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)
    Pcalc1500 = MGD(Vcalc,1500,K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)
    Pcalc2000 = MGD(Vcalc,2000,K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)
    
    norm = matplotlib.colors.Normalize(vmin=np.min(T),vmax=np.max(T))
    cmap = matplotlib.cm.get_cmap('jet')
    
    plt.figure()
    plt.scatter(P,V,c=T,cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
    plt.plot(Pcalc300,Vcalc,'-',color=cmap(norm(300)))
    plt.plot(Pcalc1000,Vcalc,'-',color=cmap(norm(1000)))
    plt.plot(Pcalc1500,Vcalc,'-',color=cmap(norm(1500)))
    plt.plot(Pcalc2000,Vcalc,'-',color=cmap(norm(2000)))
    plt.ylabel('Volume ($\AA^3$)')
    plt.xlabel('Pressure (GPa)')
    plt.tick_params(direction = 'in',top=True,right=True)
    
#    Fei = df.loc[df['Paper']=='Fei']
#    Fun = df.loc[df['Paper']=='Funamori']
#    Fisch = df.loc[df['Paper']=='Fischer']
#    Dub = df.loc[df['Paper']=='Dubrovinsky']
#    Yam = df.loc[df['Paper']=='Yamazaki']
#    Uch = df.loc[df['Paper']=='Uchida']
#    Murph = df.loc[df['Paper']=='Murphy']
#    Brown = df.loc[df['Paper']=='Brown']
#    
#    plt.figure()
#    plt.scatter(Fei['Pressure'],(MGD(Fei['Volume'],Fei['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Fei['Pressure']),c=Fei['Temperature'],cmap='jet',marker='o',label='Fei')
#    plt.scatter(Fisch['Pressure'],(MGD(Fisch['Volume'],Fisch['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Fisch['Pressure']),c=Fisch['Temperature'],cmap='jet',marker='^',label='Fischer')
#    plt.scatter(Dub['Pressure'],(MGD(Dub['Volume'],Dub['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Dub['Pressure']),c=Dub['Temperature'],cmap='jet',marker='v',label='Dubrovinsky')
#    plt.scatter(Uch['Pressure'],(MGD(Uch['Volume'],Uch['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Uch['Pressure']),c=Uch['Temperature'],cmap='jet',marker='d',label='Uchida')
#    plt.scatter(Yam['Pressure'],(MGD(Yam['Volume'],Yam['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Yam['Pressure']),c=Yam['Temperature'],cmap='jet',marker='s',label='Yamazaki')
#    plt.scatter(Fun['Pressure'],(MGD(Fun['Volume'],Fun['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Fun['Pressure']),c=Fun['Temperature'],cmap='jet',marker='*',label='Funamori')
##    plt.scatter(Murph['Pressure'],(MGD(Murph['Volume'],Murph['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Murph['Pressure']),c=Murph['Temperature'],cmap='jet',marker='+',label='Murphy')
##    plt.scatter(Brown['Pressure'],(MGD(Brown['Volume'],Brown['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Brown['Pressure']),c=Brown['Temperature'],cmap='jet',marker='x',label='Brown')
#    
#    cbar = plt.colorbar()
#    cbar.set_label('Temperature (K)')
#    plt.clim(np.min(df['Temperature']),np.max(df['Temperature']))
#    plt.xlabel('Pressure (GPa)')
#    plt.ylabel('Residual (GPa)')
#    plt.tick_params(direction = 'in',top=True,right=True)
#    plt.xlim([0,np.max(P)*1.75])
#    plt.legend()
    
    plt.figure()
    plt.scatter(P,MGD(V,T,K0_seed,Kp_seed,V0_seed,theta0_avg,gamma0_avg,q_avg,n) - P,c=T,cmap='jet',marker='o')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
    plt.clim(np.min(df['Temperature']),np.max(df['Temperature']))
    plt.xlabel('Pressure (GPa)')
    plt.ylabel('Pressure Residual (GPa)')
    plt.tick_params(direction = 'in',top=True,right=True)
    plt.legend()
    
    plt.figure()
    plt.scatter(V,Pth(V,T,V0_seed,theta0_avg,gamma0_avg,q_avg,n),c=T,cmap='jet',marker='o')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
    plt.clim(np.min(df['Temperature']),np.max(df['Temperature']))
    plt.xlabel('Volume ($\AA^3$)')
    plt.ylabel('Thermal Pressure (GPa)')
    plt.tick_params(direction = 'in',top=True,right=True)
    plt.legend()
    
    figure = corner.corner(pairs[:,0:2],labels=['$\gamma_0$','q'],truths=[gamma0_avg,q_avg],quantiles=(0.16,0.84),levels=(1-np.exp(-0.5),))
    ndim = 2
    axes = np.array(figure.axes).reshape((ndim,ndim))
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi,xi]
            ax.scatter(pairs[:,xi],pairs[:,yi],c=chisqr,s=10,alpha=0.5)
    
    if 'filename' in kwargs:
        pairs = pd.DataFrame(pairs,columns=['gamma0','q','theta0','V0','K0','Kp'])
        output = kwargs['filename'] + '_MGDBM3_MCFits.txt'
        pairs.to_csv(output,float_format='%.2f',index=False)

        MGDRes = {'Residual':MGD(df['Volume'],df['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_avg,gamma0_avg,q_avg,n)-df['Pressure'],
                  'Scale': df['scale']}
        MGDRes = pd.DataFrame(MGDRes)
        output_Res = kwargs['filename'] + '_MGDBM3_Resdiduals.txt'
        MGDRes.to_csv(output_Res,float_format='%.2f',index=False)

        
        BM3file = open(kwargs['filename']+'_MGDBM3_Fits.txt','w')
        BM3file.write('%f\t%f\t%f\t%f\t%f\t%f\t422\t0.0\t%f\t%f\t%f\t%f\t2' % (V0_seed,sigmaV0,K0_seed,sigmaK0,Kp_seed,sigmaKp,gamma0_avg,gamma0_std,q_avg,q_std))
        BM3file.write('\n%f\t%f' % (Corr[0,0],Corr[0,1]))
        BM3file.write('\n%f\t%f' % (Corr[1,0],Corr[1,1]))
        BM3file.close()
    
    
    return fits, pairs, Corr

#Routine for doing a MC fit of a High-Temperature Birch-Murnaghan EOS
def MC_EOS_HTBMlmfit(DataFile, EOSFile,varys,steps):
    df = pd.read_excel(DataFile)
    P = df['Pressure'].to_numpy()
    V = df['Volume'].to_numpy()
    T = df['Temperature'].to_numpy()
    sig_V = df['sigmaV'].to_numpy()
    sig_P = df['sigmaP'].to_numpy()
#    sig_P = np.ones_like(V)
    sig_T = df['sigmaT'].to_numpy()
    
    EOSfile = open(EOSFile)
    line = EOSfile.readline()
    line = line.strip()
    EOS = line.split()
    
    V0_seed = float(EOS[0])
    sigmaV0 = float(EOS[1])
    K0_seed = float(EOS[2])
    sigmaK0 = float(EOS[3])
    Kp_seed = float(EOS[4])
    sigmaKp = float(EOS[5])
    a0_seed = float(EOS[6])
    a1_seed = float(EOS[7])
    dK_seed = float(EOS[8])
    
    params = lmfit.Parameters()
    params.add('K0',value=K0_seed,vary= varys[0])
    params.add('Kp',value=Kp_seed,vary= varys[1])
    params.add('V0',value=V0_seed,vary= varys[2])
    params.add('a0',value=a0_seed,vary= varys[3])
    params.add('a1',value=a1_seed,vary= varys[4])
    params.add('dK',value=dK_seed,vary= varys[5])
    
    K0 = []
    Kp = []
    V0 = []
    a0 = []
    a1 = []
    dK = []
    chisqr = []

    for j in range (steps):
        Vmc = []
        Pmc = []
        Tmc = []
        for i in range (len(V)):
            Vmc.append(V[i] + sig_V[i]*np.random.randn())
            Pmc.append(P[i] + sig_P[i]*np.random.randn())
            Tmc.append(T[i] + sig_T[i]*np.random.randn())
        Vmc = np.array(Vmc)
        Pmc = np.array(Pmc)
        Tmc = np.array(Tmc)
            
        DoIt = lmfit.Minimizer(HTBMres,params=params,fcn_args=(Vmc,Tmc,Pmc,sig_P))
        result = DoIt.minimize()
        values = result.params.valuesdict()
        chisqr.append(result.chisqr)
        K0.append(values['K0'])
        Kp.append(values['Kp'])
        V0.append(values['V0'])
        a0.append(values['a0'])
        a1.append(values['a1'])
        dK.append(values['dK'])
        if chisqr[j] == np.min(chisqr):
            result_min = result
    
    K0_avg = np.mean(K0)
    K0_sd = np.std(K0)
    Kp_avg = np.mean(Kp)
    Kp_sd = np.std(Kp)
    V0_avg = np.mean(V0)
    V0_sd = np.std(V0)
    a0_avg = np.mean(a0)
    a0_sd = np.std(a0)
    a1_avg = np.mean(a1)
    a1_sd = np.std(a1)
    dK_avg = np.mean(dK)
    dK_sd = np.std(dK)
    
    chisqr_min = np.amin(chisqr)
    min_index = np.argwhere(chisqr==chisqr_min)
    min_index = int(min_index[0])
    K0_min = K0[min_index]
    Kp_min = Kp[min_index]
    V0_min = V0[min_index]
    a0_min = a0[min_index]
    a1_min = a1[min_index]
    dK_min = dK[min_index]
    
    stats = np.array([[K0_avg,Kp_avg,V0_avg,a0_avg,a1_avg,dK_avg],[K0_sd,Kp_sd,V0_sd,a0_sd,a1_sd,dK_sd]])
    mins = np.array([K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min])
    alpha_avg = a0_avg + a1_avg*T
    cov = np.cov([V0,K0,Kp,a0,a1,dK])
    corr = np.corrcoef([V0,K0,Kp,a0,a1,dK])
    pairs = np.transpose(np.array([V0,K0,Kp,a0,a1,dK]))
    
    Vcalc = np.linspace(np.min(V),np.max(V),500)
    
    norm = matplotlib.colors.Normalize(vmin=np.min(T),vmax=np.max(T))
    cmap = matplotlib.cm.get_cmap('jet')
    plt.figure()
    plt.scatter(P,V,c=T,cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
#    plt.plot(HTBM(Vcalc,3000,K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min),Vcalc,'-',color=cmap(norm(3000)))
#    plt.plot(HTBM(Vcalc,2500,K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min),Vcalc,'-',color=cmap(norm(2500)))
    plt.plot(HTBM(Vcalc,2000,K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min),Vcalc,'-',color=cmap(norm(2000)))
    plt.plot(HTBM(Vcalc,1500,K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min),Vcalc,'-',color=cmap(norm(1500)))
    plt.plot(HTBM(Vcalc,1000,K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min),Vcalc,'-',color=cmap(norm(1000)))
    plt.plot(HTBM(Vcalc,300,K0_min,Kp_min,V0_min,a0_min,a1_min,dK_min),Vcalc,'-',color=cmap(norm(300)))
#    plt.plot(HTBM(Vcalc,568,K0_avg,Kp_avg,V0_avg,a0_avg,a1_avg,dK_avg),Vcalc,'-',color=cmap(norm(568)))
    plt.xlabel('Pressure (GPa)')
    plt.ylabel('Volume ($\AA^3$)')
    plt.tick_params(direction = 'in', right = 'on', top = 'on')
#    fig = plt.gcf()
#    fig.set_size_inches(18,12)
    #plt.title('Nickel HTBM')
    
    plt.figure()
    plt.scatter(T,alpha_avg,c=P,cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Pressure (GPa)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('$\\alpha$')
    plt.ylim([-2.5E-5, 5E-4])
    plt.tick_params(direction = 'in', right = True, top = True)

    return stats, cov, corr, pairs


#High T EOS Fitting Routine
"""
This is the section where the bulk of the work gets done to fit an MGD EOS (Using a Vinet Reference EOS!!!)
and propagate the uncertainty.

DataFile is a string that represents an excel spreadsheet with Pressure, Volume, and Temperature Data with uncertainties
EOSFile is a string that represents a text file with initial values for V0, sigma V0, K0, sigma K0, K', sigmaK', theta_0, sigmaTheta_0, gamma_0, q, and the atoms/unit cell.
Varys is an array of True and/or False values that determine whether or not to Vary each of the parameters q and gamma_0. Ex) [True, True] would fit for gamma_0 and q
steps is an integer value that defines the number of MC samples to take.
"""
def EOS_MC_MGDlmfit_fixed300_weight_Vinet(DataFile,EOSFile,varys,steps,**kwargs):
    
    df = pd.read_excel(DataFile)
    V = df['Volume'].to_numpy() #angstroms3
    P = df['Pressure'].to_numpy() #GPa
    T = df['Temperature'].to_numpy() #K
    sig_V = df['sigmaV'].to_numpy()
    sig_T = df['sigmaT'].to_numpy()
    sig_P = df['sigmaP'].to_numpy() #GPa
#    sig_P = np.ones_like(V)
    
    EOSfile = open(EOSFile)
    line = EOSfile.readline()
    line = line.strip()
    EOS = line.split()
    V0_seed = float(EOS[0]) #angstroms3
    sigmaV0 = float(EOS[1])
    K0_seed = float(EOS[2]) #GPa
    sigmaK0 = float(EOS[3])
    Kp_seed = float(EOS[4])
    sigmaKp = float(EOS[5])
    theta0_seed = float(EOS[6]) #K
    sigmatheta0 = float(EOS[7])
    gamma0_seed = float(EOS[8])
    q_seed = float(EOS[9])
    at = float(EOS[10]) #atoms/unit cell
    n = at/(6.022E23) #mol per unit cell, change as necessary
    
#    cov = np.array([[sigmaV0**2,sigmaV0*sigmaK0,sigmaV0*sigmaKp],[sigmaK0*sigmaV0,sigmaK0**2,sigmaK0*sigmaKp],[sigmaKp*sigmaV0,sigmaKp*sigmaK0,sigmaKp**2]])

    line = EOSfile.readline()
    line = line.strip()
    corr1 = line.split()
    line = EOSfile.readline()
    line = line.strip()
    corr2 = line.split()
    line = EOSfile.readline()
    line = line.strip()
    corr3 = line.split()
    corr300 = np.array([corr1,corr2,corr3])
    
    if 'cov' in kwargs:
        cov = kwargs['cov']
    else:
        cov = np.array([[sigmaV0**2,float(corr300[0,1])*sigmaV0*sigmaK0,float(corr300[0,2])*sigmaV0*sigmaKp],[float(corr300[1,0])*sigmaK0*sigmaV0,sigmaK0**2,float(corr300[1,2])*sigmaK0*sigmaKp],[float(corr300[2,0])*sigmaKp*sigmaV0,float(corr300[2,1])*sigmaKp*sigmaK0,sigmaKp**2]])   

    if 'sample' in kwargs:
        if kwargs['sample'][0] == False:
            sig_V = np.zeros_like(V)
            print('Not sampling V\n')
        if kwargs['sample'][1] == False:
            sig_P = np.ones_like(P)*1E-5
            print('Not sampling P\n')
        if kwargs['sample'][2] == False:
            sig_T = np.zeros_like(T)
            print('Not sampling T\n')
        if kwargs['sample'][3] == False:
            cov = np.array([[0,0,0],[0,0,0],[0,0,0]])
            print('Not sampling EOS\n')

    params = lmfit.Parameters()
    params.add('gamma0',value=gamma0_seed,vary=varys[0])
    params.add('q',value=q_seed,vary=varys[1])
    params.add('theta0',value=theta0_seed,vary=varys[2]) #added 2/10/22 for ppv testing, comment out if needed later
    
    gamma0 = []
    q = []
    theta0 = [] #added 2/10/22 for ppv testing, comment out if needed later
    V0mc = []
    K0mc = []
    Kpmc = []
    theta0mc = []
    chisqr = []
    
    for j in range(steps):
        Vmc = []
        Pmc = []
        Tmc = []
        for i in range(len(V)):
            Vmc.append(V[i] + sig_V[i]*np.random.randn()) #randomly sample V
            Pmc.append(P[i] + sig_P[i]*np.random.randn()) #randomly sample P
            Tmc.append(T[i] + sig_T[i]*np.random.randn()) #randomly sample T
        V0temp, K0temp, Kptemp = np.random.multivariate_normal([V0_seed,K0_seed,Kp_seed],cov) #randomly sample K0 and K' from within covariance
#        K0mc.append(K0_seed + sigmaK0*np.random.randn())
#        Kpmc.append(Kp_seed + sigmaKp*np.random.randn())
#        V0mc.append(V0_seed + sigmaV0*np.random.randn()) #randomly sample and store V0
        K0mc.append(K0temp) #store 300K parameter
        Kpmc.append(Kptemp) #store 300K parameter
        V0mc.append(V0temp)
        theta0mc.append(theta0_seed + sigmatheta0*np.random.randn()) #randomly sample and store theta0
        
        #Weighted Non-linear least-squares fit (see lmfit doc)
        DoIt = lmfit.Minimizer(MGDres_weight_Vinet,params,fcn_args=(np.array(Vmc),np.array(Tmc),np.array(Pmc),K0mc[j],Kpmc[j],V0mc[j],theta0mc[j],sig_P,n))
        result = DoIt.minimize()
        values = result.params.valuesdict() #get fit parameter values 
        gamma0.append(values['gamma0']) #store fit parameter
        q.append(values['q']) #store fit parameter
        theta0.append(values['theta0']) #added 2/10/22 for ppv testing, comment out if needed later
        chisqr.append(result.chisqr)

    gamma0_avg = np.mean(gamma0)
    q_avg = np.mean(q)
    gamma0_std = np.std(gamma0)
    q_std = np.std(q)
    theta0_avg = np.mean(theta0)
    theta0_std = np.std(theta0)
    
    fits = np.array([[gamma0_avg,q_avg,theta0_avg],[gamma0_std,q_std,theta0_std]])
    pairs = np.transpose(np.array([gamma0,q,theta0,V0mc,K0mc,Kpmc]))
    Corr = np.corrcoef([gamma0,q,theta0])
    
    
    plt.figure()
    plt.plot(gamma0,q,'ko')
    plt.xlabel('$\gamma_0$')
    plt.ylabel('$q$')
    plt.tick_params(direction='in',top=True, right=True)
    
    plt.figure()
    plt.plot(K0mc,Kpmc,'ko')
    plt.xlabel('$K_0$')
    plt.ylabel('$K\'$')
    plt.tick_params(direction='in',top=True, right=True)

    plt.figure()
    plt.plot(V0mc,K0mc,'ko')
    plt.xlabel('$V_0$')
    plt.ylabel('$K_0$')
    plt.tick_params(direction='in',top=True, right=True)
    
    Vcalc = np.linspace(np.min(V),np.max(V),1000)
    Pcalc300 = Vinet(Vcalc,K0_seed,Kp_seed,V0_seed)
    Pcalc1000 = MGD_Vinet(Vcalc,1000,K0_seed,Kp_seed,V0_seed,theta0_avg,gamma0_avg,q_avg,n)
    Pcalc1500 = MGD_Vinet(Vcalc,1500,K0_seed,Kp_seed,V0_seed,theta0_avg,gamma0_avg,q_avg,n)
    Pcalc2000 = MGD_Vinet(Vcalc,2000,K0_seed,Kp_seed,V0_seed,theta0_avg,gamma0_avg,q_avg,n)
    
    norm = matplotlib.colors.Normalize(vmin=np.min(T),vmax=np.max(T))
    cmap = matplotlib.cm.get_cmap('jet')
    
    plt.figure()
    plt.scatter(P,V,c=T,cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
    plt.plot(Pcalc300,Vcalc,'-',color=cmap(norm(300)))
    plt.plot(Pcalc1000,Vcalc,'-',color=cmap(norm(1000)))
    plt.plot(Pcalc1500,Vcalc,'-',color=cmap(norm(1500)))
    plt.plot(Pcalc2000,Vcalc,'-',color=cmap(norm(2000)))
    plt.ylabel('Volume ($\AA^3$)')
    plt.xlabel('Pressure (GPa)')
    plt.tick_params(direction = 'in',top=True,right=True)
    plt.title('Vinet MGD')
    
    figure = corner.corner(pairs[:,0:2],labels=['$\gamma_0$','q'],truths=[gamma0_avg,q_avg],quantiles=(0.16,0.84),levels=(1-np.exp(-0.5),))
    ndim = 2
    axes = np.array(figure.axes).reshape((ndim,ndim))
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi,xi]
            ax.scatter(pairs[:,xi],pairs[:,yi],c=chisqr,s=10,alpha=0.5)

    if 'filename' in kwargs:
        pairs = pd.DataFrame(pairs,columns=['gamma0','q','V0','K0','Kp'])
        output = kwargs['filename'] + '_MGDVinet_MCFits.txt'
        pairs.to_csv(output,float_format='%.2f',index=False)

        MGDRes = {'Residual':MGD(df['Volume'],df['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_avg,gamma0_avg,q_avg,n)-df['Pressure'],
                  'Paper': df['Paper']}
        MGDRes = pd.DataFrame(MGDRes)
        output_Res = kwargs['filename'] + '_MGDVinet_Resdiduals.txt'
        MGDRes.to_csv(output_Res,float_format='%.2f',index=False)

        
        Vinetfile = open(kwargs['filename']+'_MGDVinet_Fits.txt','w')
        Vinetfile.write('%f\t%f\t%f\t%f\t%f\t%f\t422\t0.0\t%f\t%f\t%f\t%f\t2' % (V0_seed,sigmaV0,K0_seed,sigmaK0,Kp_seed,sigmaKp,gamma0_avg,gamma0_std,q_avg,q_std))
        Vinetfile.write('\n%f\t%f' % (Corr[0,0],Corr[0,1]))
        Vinetfile.write('\n%f\t%f' % (Corr[1,0],Corr[1,1]))
        Vinetfile.close()

    
    return fits, pairs, Corr

def EOS_MC_MGDlmfit_fixed300_weight_Vinet_q0q1(DataFile,EOSFile,cov,varys,steps):
    
    df = pd.read_excel(DataFile)
    V = df['Volume'].to_numpy() #angstroms3
    P = df['Pressure'].to_numpy() #GPa
    T = df['Temperature'].to_numpy() #K
    sig_V = df['sigmaV'].to_numpy()
    sig_T = df['sigmaT'].to_numpy()
    sig_P = df['sigmaP'].to_numpy() #GPa
#    sig_P = np.ones_like(V)
    
    EOSfile = open(EOSFile)
    line = EOSfile.readline()
    line = line.strip()
    EOS = line.split()
    V0_seed = float(EOS[0]) #angstroms3
    sigmaV0 = float(EOS[1])
    K0_seed = float(EOS[2]) #GPa
    sigmaK0 = float(EOS[3])
    Kp_seed = float(EOS[4])
    sigmaKp = float(EOS[5])
    theta0_seed = float(EOS[6]) #K
    sigmatheta0 = float(EOS[7])
    gamma0_seed = float(EOS[8])
    q_seed = float(EOS[9])
    at = float(EOS[10]) #atoms/unit cell
    n = at/(6.022E23) #mol per unit cell, change as necessary
    q1_seed = 1
    
    cov = np.array([[sigmaV0**2,sigmaV0*sigmaK0,sigmaV0*sigmaKp],[sigmaK0*sigmaV0,sigmaK0**2,sigmaK0*sigmaKp],[sigmaKp*sigmaV0,sigmaKp*sigmaK0,sigmaKp**2]])
    
    params = lmfit.Parameters()
    params.add('gamma0',value=gamma0_seed,vary=varys[0])
    params.add('q0',value=q_seed,vary=varys[1])
    params.add('q1',value=q1_seed,vary = True)
    
    gamma0 = []
    q0 = []
    q1 = []
    V0mc = []
    K0mc = []
    Kpmc = []
    theta0mc = []
    
    for j in range(steps):
        Vmc = []
        Pmc = []
        Tmc = []
        for i in range(len(V)):
            Vmc.append(V[i] + sig_V[i]*np.random.randn()) #randomly sample V
            Pmc.append(P[i] + sig_P[i]*np.random.randn()) #randomly sample P
            Tmc.append(T[i] + sig_T[i]*np.random.randn()) #randomly sample T
        V0temp, K0temp, Kptemp = np.random.multivariate_normal([V0_seed,K0_seed,Kp_seed],cov) #randomly sample K0 and K' from within covariance
#        K0mc.append(K0_seed + sigmaK0*np.random.randn())
#        Kpmc.append(Kp_seed + sigmaKp*np.random.randn())
#        V0mc.append(V0_seed + sigmaV0*np.random.randn()) #randomly sample and store V0
        K0mc.append(K0temp) #store 300K parameter
        Kpmc.append(Kptemp) #store 300K parameter
        V0mc.append(V0temp)
        theta0mc.append(theta0_seed + sigmatheta0*np.random.randn()) #randomly sample and store theta0
        
        #Weighted Non-linear least-squares fit (see lmfit doc)
        DoIt = lmfit.Minimizer(MGDres_weight_Vinet_q0q1,params,fcn_args=(np.array(Vmc),np.array(Tmc),np.array(Pmc),K0mc[j],Kpmc[j],V0mc[j],theta0mc[j],sig_P,n))
        result = DoIt.minimize()
        values = result.params.valuesdict() #get fit parameter values 
        gamma0.append(values['gamma0']) #store fit parameter
        q0.append(values['q0']) #store fit parameter
        q1.append(values['q1'])

    gamma0_avg = np.mean(gamma0)
    q0_avg = np.mean(q0)
    q1_avg = np.mean(q1)
    gamma0_std = np.std(gamma0)
    q0_std = np.std(q0)
    q1_std = np.std(q1)
    
    fits = np.array([[gamma0_avg,q0_avg,q1_avg],[gamma0_std,q0_std,q1_std]])
    pairs = np.transpose(np.array([gamma0,q0,q1,V0mc,K0mc,Kpmc]))
    Corr = np.corrcoef([gamma0,q0,q1])
    
    
    plt.figure()
    plt.plot(gamma0,q0,'ko')
    plt.xlabel('$\gamma_0$')
    plt.ylabel('$q_0$')
    plt.tick_params(direction='in',top=True, right=True)
    
    plt.figure()
    plt.plot(q0,q1,'ko')
    plt.xlabel('$q_0$')
    plt.ylabel('$q_1$')
    plt.tick_params(direction='in',top=True, right=True)
    
    plt.figure()
    plt.plot(K0mc,Kpmc,'ko')
    plt.xlabel('$K_0$')
    plt.ylabel('$K\'$')
    plt.tick_params(direction='in',top=True, right=True)

    plt.figure()
    plt.plot(V0mc,K0mc,'ko')
    plt.xlabel('$V_0$')
    plt.ylabel('$K_0$')
    plt.tick_params(direction='in',top=True, right=True)
    
    
    return fits, pairs, Corr

def EOS_MC_MGDlmfit_fixed300_weight_Vinet_elec(DataFile,EOSFile,varys,steps,**kwargs):
    
    df = pd.read_excel(DataFile)
    V = df['Volume'].to_numpy() #angstroms3
    P = df['Pressure'].to_numpy() #GPa
    T = df['Temperature'].to_numpy() #K
    sig_V = df['sigmaV'].to_numpy()
    sig_T = df['sigmaT'].to_numpy()
    sig_P = df['sigmaP'].to_numpy() #GPa
#    sig_P = np.ones_like(V)
    
    EOSfile = open(EOSFile)
    line = EOSfile.readline()
    line = line.strip()
    EOS = line.split()
    V0_seed = float(EOS[0]) #angstroms3
    sigmaV0 = float(EOS[1])
    K0_seed = float(EOS[2]) #GPa
    sigmaK0 = float(EOS[3])
    Kp_seed = float(EOS[4])
    sigmaKp = float(EOS[5])
    theta0_seed = float(EOS[6]) #K
    sigmatheta0 = float(EOS[7])
    gamma0_seed = float(EOS[8])
    q_seed = float(EOS[9])
    at = float(EOS[10]) #atoms/unit cell
    n = at/(6.022E23) #mol per unit cell, change as necessary
    
#    cov = np.array([[sigmaV0**2,sigmaV0*sigmaK0,sigmaV0*sigmaKp],[sigmaK0*sigmaV0,sigmaK0**2,sigmaK0*sigmaKp],[sigmaKp*sigmaV0,sigmaKp*sigmaK0,sigmaKp**2]])

    line = EOSfile.readline()
    line = line.strip()
    corr1 = line.split()
    line = EOSfile.readline()
    line = line.strip()
    corr2 = line.split()
    line = EOSfile.readline()
    line = line.strip()
    corr3 = line.split()
    corr300 = np.array([corr1,corr2,corr3])
    
    if 'cov' in kwargs:
        cov = kwargs['cov']
    else:
        cov = np.array([[sigmaV0**2,float(corr300[0,1])*sigmaV0*sigmaK0,float(corr300[0,2])*sigmaV0*sigmaKp],[float(corr300[1,0])*sigmaK0*sigmaV0,sigmaK0**2,float(corr300[1,2])*sigmaK0*sigmaKp],[float(corr300[2,0])*sigmaKp*sigmaV0,float(corr300[2,1])*sigmaKp*sigmaK0,sigmaKp**2]])   

    if 'sample' in kwargs:
        if kwargs['sample'][0] == False:
            sig_V = np.zeros_like(V)
            print('Not sampling V\n')
        if kwargs['sample'][1] == False:
            sig_P = np.ones_like(P)*1E-5
            print('Not sampling P\n')
        if kwargs['sample'][2] == False:
            sig_T = np.zeros_like(T)
            print('Not sampling T\n')
        if kwargs['sample'][3] == False:
            cov = np.array([[0,0,0],[0,0,0],[0,0,0]])
            print('Not sampling EOS\n')

    params = lmfit.Parameters()
    params.add('gamma0',value=gamma0_seed,vary=varys[0])
    params.add('q',value=q_seed,vary=varys[1])
    
    gamma0 = []
    q = []
    V0mc = []
    K0mc = []
    Kpmc = []
    theta0mc = []
    chisqr = []
    
    for j in range(steps):
        Vmc = []
        Pmc = []
        Tmc = []
        for i in range(len(V)):
            Vmc.append(V[i] + sig_V[i]*np.random.randn()) #randomly sample V
            Pmc.append(P[i] + sig_P[i]*np.random.randn()) #randomly sample P
            Tmc.append(T[i] + sig_T[i]*np.random.randn()) #randomly sample T
        V0temp, K0temp, Kptemp = np.random.multivariate_normal([V0_seed,K0_seed,Kp_seed],cov) #randomly sample K0 and K' from within covariance
#        K0mc.append(K0_seed + sigmaK0*np.random.randn())
#        Kpmc.append(Kp_seed + sigmaKp*np.random.randn())
#        V0mc.append(V0_seed + sigmaV0*np.random.randn()) #randomly sample and store V0
        K0mc.append(K0temp) #store 300K parameter
        Kpmc.append(Kptemp) #store 300K parameter
        V0mc.append(V0temp)
        theta0mc.append(theta0_seed + sigmatheta0*np.random.randn()) #randomly sample and store theta0
        
        #Weighted Non-linear least-squares fit (see lmfit doc)
        DoIt = lmfit.Minimizer(MGDres_weight_Vinet_elec,params,fcn_args=(np.array(Vmc),np.array(Tmc),np.array(Pmc),K0mc[j],Kpmc[j],V0mc[j],theta0mc[j],sig_P,n))
        result = DoIt.minimize()
        values = result.params.valuesdict() #get fit parameter values 
        gamma0.append(values['gamma0']) #store fit parameter
        q.append(values['q']) #store fit parameter
        chisqr.append(result.chisqr)

    gamma0_avg = np.mean(gamma0)
    q_avg = np.mean(q)
    gamma0_std = np.std(gamma0)
    q_std = np.std(q)
    
    fits = np.array([[gamma0_avg,q_avg],[gamma0_std,q_std]])
    pairs = np.transpose(np.array([gamma0,q,V0mc,K0mc,Kpmc]))
    Corr = np.corrcoef([gamma0,q])
    
    
    plt.figure()
    plt.plot(gamma0,q,'ko')
    plt.xlabel('$\gamma_0$')
    plt.ylabel('$q$')
    plt.tick_params(direction='in',top=True, right=True)
    
    plt.figure()
    plt.plot(K0mc,Kpmc,'ko')
    plt.xlabel('$K_0$')
    plt.ylabel('$K\'$')
    plt.tick_params(direction='in',top=True, right=True)

    plt.figure()
    plt.plot(V0mc,K0mc,'ko')
    plt.xlabel('$V_0$')
    plt.ylabel('$K_0$')
    plt.tick_params(direction='in',top=True, right=True)
    
    Vcalc = np.linspace(np.min(V),np.max(V),1000)
    Pcalc300 = Vinet(Vcalc,K0_seed,Kp_seed,V0_seed)
    Pcalc1000 = MGD_Vinet_elec(Vcalc,1000,K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)
    Pcalc1500 = MGD_Vinet_elec(Vcalc,1500,K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)
    Pcalc2000 = MGD_Vinet_elec(Vcalc,2000,K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)
    
    norm = matplotlib.colors.Normalize(vmin=np.min(T),vmax=np.max(T))
    cmap = matplotlib.cm.get_cmap('jet')
    
    plt.figure()
    plt.scatter(P,V,c=T,cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
    plt.plot(Pcalc300,Vcalc,'-',color=cmap(norm(300)))
    plt.plot(Pcalc1000,Vcalc,'-',color=cmap(norm(1000)))
    plt.plot(Pcalc1500,Vcalc,'-',color=cmap(norm(1500)))
    plt.plot(Pcalc2000,Vcalc,'-',color=cmap(norm(2000)))
    plt.ylabel('Volume ($\AA^3$)')
    plt.xlabel('Pressure (GPa)')
    plt.tick_params(direction = 'in',top=True,right=True)
    plt.title('Vinet MGD')
    
    Fei = df.loc[df['Paper']=='Fei']
    Fun = df.loc[df['Paper']=='Funamori']
    Fisch = df.loc[df['Paper']=='Fischer']
    Dub = df.loc[df['Paper']=='Dubrovinsky']
    Yam = df.loc[df['Paper']=='Yamazaki']
    Uch = df.loc[df['Paper']=='Uchida']
    Murph = df.loc[df['Paper']=='Murphy']
    Brown = df.loc[df['Paper']=='Brown']
    
    plt.figure()
    plt.scatter(Fei['Pressure'],(MGD_Vinet_elec(Fei['Volume'],Fei['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Fei['Pressure']),c=Fei['Temperature'],cmap='jet',marker='o',label='Fei')
    plt.scatter(Fisch['Pressure'],(MGD_Vinet_elec(Fisch['Volume'],Fisch['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Fisch['Pressure']),c=Fisch['Temperature'],cmap='jet',marker='^',label='Fischer')
    plt.scatter(Dub['Pressure'],(MGD_Vinet_elec(Dub['Volume'],Dub['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Dub['Pressure']),c=Dub['Temperature'],cmap='jet',marker='v',label='Dubrovinsky')
    plt.scatter(Uch['Pressure'],(MGD_Vinet_elec(Uch['Volume'],Uch['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Uch['Pressure']),c=Uch['Temperature'],cmap='jet',marker='d',label='Uchida')
    plt.scatter(Yam['Pressure'],(MGD_Vinet_elec(Yam['Volume'],Yam['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Yam['Pressure']),c=Yam['Temperature'],cmap='jet',marker='s',label='Yamazaki')
    plt.scatter(Fun['Pressure'],(MGD_Vinet_elec(Fun['Volume'],Fun['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Fun['Pressure']),c=Fun['Temperature'],cmap='jet',marker='*',label='Funamori')
#    plt.scatter(Murph['Pressure'],(MGD_Vinet_elec(Murph['Volume'],Murph['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Murph['Pressure']),c=Murph['Temperature'],cmap='jet',marker='+',label='Murphy')
#    plt.scatter(Brown['Pressure'],(MGD_Vinet_elec(Brown['Volume'],Brown['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-Brown['Pressure']),c=Brown['Temperature'],cmap='jet',marker='x',label='Brown')
    cbar = plt.colorbar()
    cbar.set_label('Temperature (K)')
    plt.clim(np.min(df['Temperature']),np.max(df['Temperature']))
    plt.xlabel('Pressure (GPa)')
    plt.ylabel('Residual (GPa)')
    plt.tick_params(direction = 'in',top=True,right=True)
    plt.xlim([0,np.max(P)*1.75])
    plt.legend()
    plt.title('Vinet MGD')
    
    figure = corner.corner(pairs[:,0:2],labels=['$\gamma_0$','q'],truths=[gamma0_avg,q_avg],quantiles=(0.16,0.84),levels=(1-np.exp(-0.5),))
    ndim = 2
    axes = np.array(figure.axes).reshape((ndim,ndim))
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi,xi]
            ax.scatter(pairs[:,xi],pairs[:,yi],c=chisqr,s=10,alpha=0.5)

    if 'filename' in kwargs:
        pairs = pd.DataFrame(pairs,columns=['gamma0','q','V0','K0','Kp'])
        output = kwargs['filename'] + '_MGDVinet_MCFits.txt'
        pairs.to_csv(output,float_format='%.2f',index=False)

        MGDRes = {'Residual':MGD(df['Volume'],df['Temperature'],K0_seed,Kp_seed,V0_seed,theta0_seed,gamma0_avg,q_avg,n)-df['Pressure'],
                  'Paper': df['Paper']}
        MGDRes = pd.DataFrame(MGDRes)
        output_Res = kwargs['filename'] + '_MGDVinet_Resdiduals.txt'
        MGDRes.to_csv(output_Res,float_format='%.2f',index=False)

        
        Vinetfile = open(kwargs['filename']+'_MGDVinet_Fits.txt','w')
        Vinetfile.write('%f\t%f\t%f\t%f\t%f\t%f\t422\t0.0\t%f\t%f\t%f\t%f\t2' % (V0_seed,sigmaV0,K0_seed,sigmaK0,Kp_seed,sigmaKp,gamma0_avg,gamma0_std,q_avg,q_std))
        Vinetfile.write('\n%f\t%f' % (Corr[0,0],Corr[0,1]))
        Vinetfile.write('\n%f\t%f' % (Corr[1,0],Corr[1,1]))
        Vinetfile.close()

    
    return fits, pairs, Corr