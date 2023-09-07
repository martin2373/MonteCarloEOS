"""
This file contains the functions needed to calculate pressure and pressure uncertainty
using a Monte Carlo style sampling routine that samples volumes, temperatures, and pressure
scale parameters to propagate the uncertainty and covariance through to the resulting
pressure value.
"""

from sys import argv
import numpy as np
import math
import matplotlib.pyplot as plt

#script, EOSfile, inputVFile = argv

#EOSfile format: 
#EOS type: 0 BM, 1 Vinet
# V0, K0, Kp
# dV0, dK0, dKp 
# corrV0K0, corrV0Kp, corrK0Kp
# Theta0, Gamma0, q -- assuming for the moment MGD model as in Fei et al., 2007
# dTh0, dG0, dq

def BM3(x,a,b,c):
    P = ((3/2)*a*((c/x)**(7/3)-(c/x)**(5/3))*(1+(3/4)*(b-4)*((c/x)**(2/3)-1)))
    return P

def Vinet(V,K0,Kp,V0):
    f = (V/V0)**(1/3)
    eta = (3/2)*(Kp-1)
    P = 3*K0*((1-f)/(f**2))*np.exp(eta*(1-f))
    
#    print(V,K0,Kp,V0,f,eta,P)
    return P

def MGDinf(V,T,EOS,z,Type):
    
    K0 = EOS[0]
    Kp = EOS[1]
    V0 = EOS[2]
    theta0 = EOS[3]
    gamma0 = EOS[4]
    gammainf = EOS[5]
    B = EOS[6]
    a = EOS[7]*(10**-6)
    m = EOS[8]
    e = EOS[9]*(10**-6)
    g = EOS[10]
    n = EOS[11]
    
    R = 8.31 #J/molK
    
    if Type=='BM3':
        P300 = BM3(V,K0,Kp,V0)
    if Type == 'Vinet':
        P300 = Vinet(V,K0,Kp,V0)
    
    gamma = gammainf + (gamma0 - gammainf)*((V/V0)**B)
    
    theta = theta0*((V/V0)**(-gammainf))*np.exp(((gamma0-gammainf)/B)*(1-((V/V0)**B)))
    
    F_anhT = (-1.5)*n*R*a*((V/V0)**m)*(T**2)
    F_elT = (-1.5)*n*R*e*((V/V0)**g)*(T**2)
    F_anh300 = (-1.5)*n*R*a*((V/V0)**m)*(300**2)
    F_el300 = (-1.5)*n*R*e*((V/V0)**g)*(300**2)

    x = theta/T
    x300 = theta/300
    F_qh300 = 9*n*R*((theta/8)+300*((300/theta)**3)*((x300**3)*((1/3)-(x300/8)+(((1/6)*(x300**2))/10)-(((1/30)*x300**4)/(7*4*3*2))))) #J
    F_qhT = 9*n*R*((theta/8)+T*((T/theta)**3)*((x**3)*((1/3)-(x/8)+(((1/6)*(x**2))/10)-(((1/30)*x**4)/(7*4*3*2))))) #J

    U300 = F_qh300 + F_anh300 + F_el300
    UT = F_qhT + F_anhT + F_elT
    
    Pth = z*((gamma*(UT-U300))/(V*1E-30))*(1E-9)
    
    P = P300 + Pth
    
    
    return P


"""
This function does the recalculation.

EOSfile is a text file formatted as described above.
inputVFile is a text file that contains V, sigmaV, T, and sigmaT in columns
"""
def CalcPAndPErr(EOSfile, inputVFile):

    inputs = open(EOSfile)
    #read in Type of EOS
    line=inputs.readline()
    line=line.strip()
    EOSTYPE=float(line)
    
    #read in EOS (V0, K0, Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    EOS=line.split() #chop entries into an arrage
    
    #read in dEOS (dV0, dK0, dKp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    dEOS=line.split() #chop entries into an arrage
    
    #read in correlations (covV0K0, covV0Kp, covK0Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    corr=line.split() #chop entries into an arrage
    
    #read in thermal EOS (Theta0,Gamma0,q)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    MGD=line.split() #chop entries into an array
    
    #read in thermal dEOS
    line=inputs.readline()
    line=line.strip() #deal with the blanks
    dMGD=line.split() #chop entries into an array
    
    #inputVFile format
    #volume volume uncertainty
    
    VTDataList =[]
    #read in V data 
    with open(inputVFile) as f:
        for line in f:
            line=line.strip()
            VTDataList.append(line.split())
    NumPnts = len(VTDataList)
    
    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
    KVar = math.pow(float(dEOS[1]),2)
    KpVar = math.pow(float(dEOS[2]),2) 
    
    mean=[float(EOS[0]),float(EOS[1]),float(EOS[2])]
    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
    
    Th0=float(MGD[0])
    G0=float(MGD[1])
    q=float(MGD[2])
    V0=float(EOS[0])
    #a=float(MGD[2])
    #b=float(MGD[3])
    R=8.31
    n=4 #atoms per unit cell
    
    Pcalc = []
    sigmaP = []
    for j in range (0,NumPnts):
        Plist=[]
        for i in range (0,2000):
            V0temp, Ktemp,Kptemp = np.random.multivariate_normal(mean, cov)
            Vtemp = np.random.normal(float(VTDataList[j][0]),float(VTDataList[j][1]))
            Ttemp = np.random.normal(float(VTDataList[j][2]),float(VTDataList[j][3]))
    
            if (EOSTYPE==0):   #BM 3rd order
                f = 0.5*(math.pow(float(V0temp)/Vtemp,(2./3.))-1.)
                firstpart=3*f*math.pow((2*f+1.0),2.5)
                TempP300= firstpart*Ktemp*(1.0+1.5*(Kptemp-4.0)*f)
            else:    #Vinet EOS
                f = math.pow((float(Vtemp)/float(V0temp)),(1./3.))
                TempP300 = 3*float(Ktemp)*(1-f)/(math.pow(f,2.))*math.exp(3./2.*(float(Kptemp)-1)*(1-f))
            gamma=G0*(math.pow((Vtemp/V0),q))
            theta = Th0*(math.pow((Vtemp/V0),-gamma))
            x300=theta/300.0
            xT=theta/Ttemp
            E300=3*(1./3.-x300/8.+(x300**2)/60.-(x300**4)/5040.)*3*n*R*300
            ETh=3*(1./3.-xT/8.+(xT**2)/60.-(xT**4)/5040.)*3*n*R*Ttemp
            TempP = TempP300+gamma/Vtemp*(ETh-E300)/602.2
    
            Plist.append(TempP)
        print("%.2f" % np.mean(Plist), ("%.2f" % np.std(Plist)), VTDataList[j][0], VTDataList[j][1], VTDataList[j][2], VTDataList[j][3])
        Pcalc.append(np.mean(Plist))
        sigmaP.append(np.std(Plist))
        
    return Pcalc, sigmaP

def PRecalc_300(EOSfile,inputVFile, corrzero):
    
    inputs = open(EOSfile)
    #read in Type of EOS
    line=inputs.readline()
    line=line.strip()
    EOSTYPE=float(line)
    
    #read in EOS (V0, K0, Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    EOS=line.split() #chop entries into an arrage
    
    #read in dEOS (dV0, dK0, dKp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    dEOS=line.split() #chop entries into an arrage
    
    #read in correlations (covV0K0, covV0Kp, covK0Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    corr=line.split() #chop entries into an arrage
    
    #read in thermal EOS (Theta0,Gamma0,q)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    MGD=line.split() #chop entries into an array
    
    #read in thermal dEOS
    line=inputs.readline()
    line=line.strip() #deal with the blanks
    dMGD=line.split() #chop entries into an array
    
    #inputVFile format
    #volume volume uncertainty
    
    VTDataList =[]
    #read in V data 
    with open(inputVFile) as f:
        for line in f:
            line=line.strip()
            VTDataList.append(line.split())
    NumPnts = len(VTDataList)
    
    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
    KVar = math.pow(float(dEOS[1]),2)
    KpVar = math.pow(float(dEOS[2]),2) 
    
    mean=[float(EOS[0]),float(EOS[1]),float(EOS[2])]
    
    if corrzero == 0:
        cov = [[VVar,0, 0], [0,KVar,0], [0,0,KpVar]]
    else:
        cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
    
    print(cov)
    
    Th0=float(MGD[0])
    G0=float(MGD[1])
    q=float(MGD[2])
    V0=float(EOS[0])
    #a=float(MGD[2])
    #b=float(MGD[3])
    R=8.31
    n=4 #atoms per unit cell
    
    Pcalc = []
    sigmaP = []
    V = []
    sigmaV = []
    for j in range (0,NumPnts):
        Plist=[]
        V0 = []
        K0 = []
        Kp = []
        for i in range (0,2000):
            V0temp, Ktemp,Kptemp = np.random.multivariate_normal(mean, cov)
            Vtemp = np.random.normal(float(VTDataList[j][0]),float(VTDataList[j][1]))
            Ttemp = np.random.normal(float(VTDataList[j][2]),float(VTDataList[j][3]))
            
            K0.append(Ktemp)
            Kp.append(Kptemp)
            V0.append(V0temp)
            
            if (EOSTYPE==0):   #BM 3rd order
                f = 0.5*(math.pow(float(V0temp)/Vtemp,(2./3.))-1.)
                firstpart=3*f*math.pow((2*f+1.0),2.5)
                TempP300= firstpart*Ktemp*(1.0+1.5*(Kptemp-4.0)*f)
            else:    #Vinet EOS
                f = math.pow((float(Vtemp)/float(V0temp)),(1./3.))
                TempP300 = 3*float(Ktemp)*(1-f)/(math.pow(f,2.))*math.exp(3./2.*(float(Kptemp)-1)*(1-f))
        
            Plist.append(TempP300)
            
        if j == 0:
            plt.plot(K0,Kp,'ko')
            plt.xlabel('$K_0$')
            plt.ylabel('$K\'$')
            plt.tick_params(direction='in',top=True,right=True)
            
        print("%.2f" % np.mean(Plist), ("%.2f" % np.std(Plist)), VTDataList[j][0], VTDataList[j][1], VTDataList[j][2], VTDataList[j][3])
        
        Pcalc.append(np.mean(Plist))
        sigmaP.append(np.std(Plist))
        V.append(VTDataList[j][0])
        sigmaV.append(VTDataList[j][1])
        
    return Pcalc, sigmaP, V, sigmaV

def PRecalc_MGD(EOSfile, inputVFile, corrzero):

    inputs = open(EOSfile)
    #read in Type of EOS
    line=inputs.readline()
    line=line.strip()
    EOSTYPE=float(line)
    
    #read in EOS (V0, K0, Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    EOS=line.split() #chop entries into an arrage
    
    #read in dEOS (dV0, dK0, dKp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    dEOS=line.split() #chop entries into an arrage
    
    #read in correlations (covV0K0, covV0Kp, covK0Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    corr=line.split() #chop entries into an arrage
    
    #read in thermal EOS (Theta0,Gamma0,q)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    MGD=line.split() #chop entries into an array
    
    #read in thermal dEOS
    line=inputs.readline()
    line=line.strip() #deal with the blanks
    dMGD=line.split() #chop entries into an array
    
    #read in thermal corr (G0Th0, qTh0, G0q)
    line=inputs.readline()
    line=line.strip()
    corrMGD=line.split()
    
    #inputVFile format
    #volume volume uncertainty
    
    VTDataList =[]
    #read in V data 
    with open(inputVFile) as f:
        for line in f:
            line=line.strip()
            VTDataList.append(line.split())
    NumPnts = len(VTDataList)
    
    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
    KVar = math.pow(float(dEOS[1]),2)
    KpVar = math.pow(float(dEOS[2]),2) 
    
    mean=[float(EOS[0]),float(EOS[1]),float(EOS[2])]
    
    if corrzero == 0:
        cov = [[VVar,0, 0], [0,KVar,0], [0,0,KpVar]]
    else:
        cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
    
    print(cov)
    
    Th0=float(MGD[0])
    G0=float(MGD[1])
    q=float(MGD[2])
    V0=float(EOS[0])
    #a=float(MGD[2])
    #b=float(MGD[3])
    R=8.31
    n=4 #atoms per unit cell
    
    G0Var = math.pow(float(dMGD[1]),2)
    qVar = math.pow(float(dMGD[2]),2)
    G0qcov = float(corrMGD[2])*(float(dMGD[1])*float(dMGD[2]))
    meanMGD = [G0,q]
    
    if corrzero == 0:
        covMGD = [[G0Var,0],[0,qVar]]
    else:
        covMGD = [[G0Var,G0qcov],[G0qcov,qVar]]
    
    Pcalc = []
    sigmaP = []
    for j in range (0,NumPnts):
        Plist=[]
        for i in range (0,2000):
            V0temp, Ktemp,Kptemp = np.random.multivariate_normal(mean, cov)
            
            G0temp, qtemp = np.random.multivariate_normal(meanMGD,covMGD)
            
            Vtemp = np.random.normal(float(VTDataList[j][0]),float(VTDataList[j][1]))
            Ttemp = np.random.normal(float(VTDataList[j][2]),float(VTDataList[j][3]))
    
            if (EOSTYPE==0):   #BM 3rd order
                f = 0.5*(math.pow(float(V0temp)/Vtemp,(2./3.))-1.)
                firstpart=3*f*math.pow((2*f+1.0),2.5)
                TempP300= firstpart*Ktemp*(1.0+1.5*(Kptemp-4.0)*f)
            else:    #Vinet EOS
                f = math.pow((float(Vtemp)/float(V0temp)),(1./3.))
                TempP300 = 3*float(Ktemp)*(1-f)/(math.pow(f,2.))*math.exp(3./2.*(float(Kptemp)-1)*(1-f))
            
            gamma=G0temp*(math.pow((Vtemp/V0),qtemp))
            theta = Th0*(math.pow((Vtemp/V0),-gamma))
            x300=theta/300.0
            xT=theta/Ttemp
            E300=3*(1./3.-x300/8.+(x300**2)/60.-(x300**4)/5040.)*3*n*R*300
            ETh=3*(1./3.-xT/8.+(xT**2)/60.-(xT**4)/5040.)*3*n*R*Ttemp
            TempP = TempP300+gamma/Vtemp*(ETh-E300)/602.2
    
            Plist.append(TempP)
        print("%.2f" % np.mean(Plist), ("%.2f" % np.std(Plist)), VTDataList[j][0], VTDataList[j][1], VTDataList[j][2], VTDataList[j][3])
        Pcalc.append(np.mean(Plist))
        sigmaP.append(np.std(Plist))
        
    return Pcalc, sigmaP, VTDataList[:][0], VTDataList[:][1], VTDataList[:][2], VTDataList[:][3]

def PRecalc_SpezialeMGD(EOSfile, inputVFile, corrzero):

    inputs = open(EOSfile)
    #read in Type of EOS
    line=inputs.readline()
    line=line.strip()
    EOSTYPE=float(line)
    
    #read in EOS (V0, K0, Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    EOS=line.split() #chop entries into an arrage
    
    #read in dEOS (dV0, dK0, dKp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    dEOS=line.split() #chop entries into an arrage
    
    #read in correlations (covV0K0, covV0Kp, covK0Kp)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    corr=line.split() #chop entries into an arrage
    
    #read in thermal EOS (Theta0,Gamma0,q)
    line=inputs.readline()
    line=line.strip() #deals with the blanks
    MGD=line.split() #chop entries into an array
    
    #read in thermal dEOS
    line=inputs.readline()
    line=line.strip() #deal with the blanks
    dMGD=line.split() #chop entries into an array
    
    #read in thermal corr (G0Th0, qTh0, G0q)
    line=inputs.readline()
    line=line.strip()
    corrMGD=line.split()

    
    #inputVFile format
    #volume volume uncertainty
    
    VTDataList =[]
    #read in V data 
    with open(inputVFile) as f:
        for line in f:
            line=line.strip()
            VTDataList.append(line.split())
    NumPnts = len(VTDataList)
    
    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
    KVar = math.pow(float(dEOS[1]),2)
    KpVar = math.pow(float(dEOS[2]),2) 
    
    mean=[float(EOS[0]),float(EOS[1]),float(EOS[2])]
    
    if corrzero == 0:
        cov = [[VVar,0, 0], [0,KVar,0], [0,0,KpVar]]
    else:
        cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
    
    print(cov)
    
    Th0=float(MGD[0])
    G0=float(MGD[1])
    q0=float(MGD[2])
    q1=float(MGD[3])
    V0=float(EOS[0])
    #a=float(MGD[2])
    #b=float(MGD[3])
    R=8.31
    n=4 #atoms per unit cell
    
#    G0Var = math.pow(float(dMGD[1]),2)
#    qVar = math.pow(float(dMGD[2]),2)
#    G0qcov = float(corrMGD[2])*(float(dMGD[1])*float(dMGD[2]))
#    meanMGD = [G0,q0]
    
#    if corrzero == 0:
#        covMGD = [[G0Var,0],[0,qVar]]
#    else:
#        covMGD = [[G0Var,G0qcov],[G0qcov,qVar]]
    
    Pcalc = []
    sigmaP = []
    for j in range (0,NumPnts):
        Plist=[]
        for i in range (0,2000):
            V0temp, Ktemp,Kptemp = np.random.multivariate_normal(mean, cov)
            
            #G0temp, qtemp = np.random.multivariate_normal(meanMGD,covMGD)
            G0temp = np.random.normal(G0,float(dMGD[1]))
            q0temp = np.random.normal(q0,float(dMGD[2]))
            q1temp = np.random.normal(q1,float(dMGD[3]))
            
            Vtemp = np.random.normal(float(VTDataList[j][0]),float(VTDataList[j][1]))
            Ttemp = np.random.normal(float(VTDataList[j][2]),float(VTDataList[j][3]))
    
            if (EOSTYPE==0):   #BM 3rd order
                f = 0.5*(math.pow(float(V0temp)/Vtemp,(2./3.))-1.)
                firstpart=3*f*math.pow((2*f+1.0),2.5)
                TempP300= firstpart*Ktemp*(1.0+1.5*(Kptemp-4.0)*f)
            else:    #Vinet EOS
                f = math.pow((float(Vtemp)/float(V0temp)),(1./3.))
                TempP300 = 3*float(Ktemp)*(1-f)/(math.pow(f,2.))*math.exp(3./2.*(float(Kptemp)-1)*(1-f))
            
            gamma=G0temp*np.exp((q0temp/q1temp)*(((Vtemp/V0temp)**q1temp)-1))
            theta = Th0*(math.pow((Vtemp/V0temp),-gamma))
            x300=theta/300.0
            xT=theta/Ttemp
            E300=3*(1./3.-x300/8.+(x300**2)/60.-(x300**4)/5040.)*3*n*R*300
            ETh=3*(1./3.-xT/8.+(xT**2)/60.-(xT**4)/5040.)*3*n*R*Ttemp
            TempP = TempP300+gamma/Vtemp*(ETh-E300)/602.2
    
            Plist.append(TempP)
        print("%.2f" % np.mean(Plist), ("%.2f" % np.std(Plist)), VTDataList[j][0], VTDataList[j][1], VTDataList[j][2], VTDataList[j][3])
        Pcalc.append(np.mean(Plist))
        sigmaP.append(np.std(Plist))
        
    return Pcalc, sigmaP, VTDataList[:][0], VTDataList[:][1], VTDataList[:][2], VTDataList[:][3]

def Spez(V,T,n):
    
    R = 8.31 #J/mol/K
    
    P300 = BM3(V,160.2,3.99,74.71)
    
    gamma = 1.524*(np.exp((1.65/11.8)*(((V/74.71)**11.8)-1)))
    theta = 773*(V/74.71)**gamma
    x300=theta/300.0
    x=theta/T
    U300 = 9*n*R*((theta/8)+300*((300/theta)**3)*((x300**3)*((1/3)-(x300/8)+(((1/6)*(x300**2))/10)-(((1/30)*x300**4)/(7*4*3*2))))) #J
    UT = 9*n*R*((theta/8)+T*((T/theta)**3)*((x**3)*((1/3)-(x/8)+(((1/6)*(x**2))/10)-(((1/30)*x**4)/(7*4*3*2))))) #J
    Pth = ((gamma*(UT-U300))/(V*1E-30))*1E-9 #GPa
    
    P = P300 + Pth
    
    return P

def SakaiKeaneMgO(V):
    
    V0 = 74.698
    K0 = 160.622
    K0p = 4.3992
    gamma0 = 1.445
    gammainf = 1.18
    b = 4.1
    Kinfp = 2.69
    
    P300 = K0* ((K0p/(Kinfp**2))*(((V0/V)**Kinfp)-1) - ((K0p/Kinfp)-1)*np.log(V0/V))
    Pth = 0
    
    P = P300 + Pth
    
    return P

def D07(Volume,scales,**kwargs):
    
    P = []
    
    if 'Temperature' in kwargs:
        for V,scale,T in zip(Volume,scales,kwargs['Temperature']):
            if scale == 'MgO':
                z=2
                EOS = [160.3,4.18,74.713,760,1.50,0.75,2.96,-14.9,5.12,0,0,4/(6.022E23)]
                P.append(MGDinf(V,T,EOS,z,'Vinet'))
            if scale == 'Au':
                z=1
                EOS = [167,5.90,67.851,170,2.89,1.54,4.36,0,0,0,0,4/(6.022E23)]
                P.append(MGDinf(V,T,EOS,z,'Vinet'))
            if scale == 'Pt':
                z=1
                EOS = [277.3,5.12,60.385,220,2.82,1.83,8.11,-166.9,4.32,260,2.4,4/(6.022E23)]
                P.append(MGDinf(V,T,EOS,'Vinet'))
            if scale == 'NaCl':
                z=2
                EOS = [29.72,5.14,40.734,270,1.64,1.23,6.83,-24.0,7.02,0,0,1/(6.022E23)]
                P.append(MGDinf(V,T,EOS,'Vinet'))
            
    else:
        for V,scale in zip(Volume,scales):
            if scale == 'MgO':
                EOS = [160.3,4.18,74.713]
                P.append(Vinet(V,EOS[0],EOS[1],EOS[2]))
            if scale == 'Au':
                EOS = [167,5.90,67.851]
                P.append(Vinet(V,EOS[0],EOS[1],EOS[2]))
            if scale == 'Pt':
                EOS = [277.3,5.12,60.385]
                P.append(Vinet(V,EOS[0],EOS[1],EOS[2]))
            if scale == 'NaCl':
                EOS = [29.72,5.14,40.734]
                P.append(Vinet(V,EOS[0],EOS[1],EOS[2]))
    
    return P

def D07_MC(Volume,sigmaV,scales,**kwargs):
    
    Pcalc = []
    sigmaP = []
    
    if 'sigmaEOS' in kwargs:
        
        if 'Temperature' in kwargs:
            
            for V,sigmaV,T,sigmaT,scale in zip(Volume,sigmaV,kwargs['Temperature'],kwargs['sigmaT'],scales):
                #scalesmc = [scale] * 1000 #generate 1000 "samples" of the scale term.
                Vmc = np.random.normal(V,sigmaV,1000) #generate 1000 samples of volume, normally distributed
                Tmc = np.random.normal(T,sigmaT,1000) #generate 1000 samples of temperature, normally distributed
                
                if scale == 'MgO':
                    z=2
                    EOSmean = [74.713,160.3,4.18]
                    dEOS=[0.01,0.2	,0.01]
                    corr = [0.0,0.0,-0.95]
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2) 
                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    theta0 = 760*np.ones(1000)
                    gamma0 = 1.50*np.ones(1000)
                    gammainf = 0.75*np.ones(1000)
                    B = 2.96*np.ones(1000)
                    a = -14.9*np.ones(1000)
                    m = 5.12*np.ones(1000)
                    e = np.zeros(1000)
                    g = np.zeros(1000)
                    n = (4/(6.022E23))*np.ones(1000)

                    
                    EOS = np.array([K0,Kp,V0,theta0,gamma0,gammainf,B,a,m,e,g,n])
                    EOS = EOS.transpose()
                    
                if scale == 'Au':
                    z=1
                    EOSmean = [67.851,167,5.90]
                    dEOS=[0.004,1.67,0.02]
                    corr = [-0.967	,0.884,-0.969]
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2) 

                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    theta0 = 170*np.ones(1000)
                    gamma0 = 2.89*np.ones(1000)
                    gammainf = 1.54*np.ones(1000)
                    B = 4.36*np.ones(1000)
                    a = np.zeros(1000)
                    m = np.zeros(1000)
                    e = np.zeros(1000)
                    g = np.zeros(1000)
                    n = (4/(6.022E23))*np.ones(1000)

                    EOS = np.array([K0,Kp,V0,theta0,gamma0,gammainf,B,a,m,e,g,n])
                    EOS = EOS.transpose()
                if scale == 'Pt':
                    z=1
                    EOSmean = [60.385,277.3,5.12]
                    dEOS=[0.01,2.77,0.02]
                    corr = [-0.967,0.884,-0.969]
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2)                     
                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    theta0 = 220*np.ones(1000)
                    gamma0 = 2.82*np.ones(1000)
                    gammainf = 1.83*np.ones(1000)
                    B = 8.11*np.ones(1000)
                    a = -166.9*np.ones(1000)
                    m = 4.32*np.ones(1000)
                    e = 260*np.ones(1000)
                    g = 2.4*np.ones(1000)
                    n = (4/(6.022E23))*np.ones(1000)

                    EOS = np.array([K0,Kp,V0,theta0,gamma0,gammainf,B,a,m,e,g,n])
                    EOS = EOS.transpose()

                if scale == 'NaCl':
                    z=2
                    EOSmean = [40.734,29.72,5.14]
                    dEOS=[0.00,2.90,0.26]
                    corr = [0.0,0.0,-0.969]   
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2) 
                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    theta0 = 270*np.ones(1000) 
                    gamma0 = 1.64*np.ones(1000)
                    gammainf = 1.23*np.ones(1000)
                    B = 6.83*np.ones(1000)
                    a = -24.0*np.ones(1000)
                    m = 7.02*np.ones(1000)
                    e = np.zeros(1000)
                    g = np.zeros(1000)
                    n = (1/(6.022E23))*np.ones(1000)

                    EOS = np.array([K0,Kp,V0,theta0,gamma0,gammainf,B,a,m,e,g,n])
                    EOS = EOS.transpose()
                
                Pmc = []
                for i in range (0,1000):
                    Pmc.append(MGDinf(Vmc[i],Tmc[i],EOS[i],z,'Vinet'))
                
                Pcalc.append(np.mean(Pmc))
                sigmaP.append(np.std(Pmc))
        
        else:
            for V,sigmaV,scale in zip(Volume,sigmaV,scales):
                #scalesmc = [scale] * 1000 #generate 1000 "samples" of the scale term.
                Vmc = np.random.normal(V,sigmaV,1000) #generate 1000 samples of volume, normally distributed
                
                if scale == 'MgO':
                    EOSmean = [74.713,160.3,4.18]
                    dEOS=[0.01,0.2	,0.01]
                    corr = [0.0,0.0,-0.95]
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2) 
                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    EOS = np.array([K0,Kp,V0])
                    EOS = EOS.transpose()
                if scale == 'Au':
                    EOSmean = [67.851,167,5.90]
                    dEOS=[0.004,1.67,0.02]
                    corr = [-0.967	,0.884,-0.969]
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2) 

                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    EOS = np.array([K0,Kp,V0])
                    EOS = EOS.transpose()
                if scale == 'Pt':
                    EOSmean = [60.385,277.3,5.12]
                    dEOS=[0.01,2.77,0.02]
                    corr = [-0.967,0.884,-0.969]
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2)                     
                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    EOS = np.array([K0,Kp,V0])
                    EOS = EOS.transpose()

                if scale == 'NaCl':
                    EOSmean = [40.734,29.72,5.14]
                    dEOS=[0.00,2.90,0.26]
                    corr = [0.0,0.0,-0.969]   
                    
                    VKcov = float(corr[0])*(float(dEOS[0])*float(dEOS[1]))
                    VKpcov = float(corr[1])*(float(dEOS[0])*float(dEOS[2]))
                    KKpcov = float(corr[2])*(float(dEOS[1])*float(dEOS[2]))
                    VVar = math.pow(float(dEOS[0]),2) #The variance is the square of the standard deviation
                    KVar = math.pow(float(dEOS[1]),2)
                    KpVar = math.pow(float(dEOS[2]),2) 
                    
                    cov = [[VVar,VKcov, VKpcov], [VKcov,KVar,KKpcov], [VKpcov,KKpcov,KpVar]]
                    EOS = np.random.multivariate_normal(EOSmean,cov,size=1000)
                    
                    V0 = EOS[:,0]
                    K0 = EOS[:,1]
                    Kp = EOS[:,2]
                    
                    EOS = np.array([K0,Kp,V0])
                    EOS = EOS.transpose()
                
                Pmc = []
                for i in range (0,1000):
                    Pmc.append(Vinet(Vmc[i],EOS[i,0],EOS[i,1],EOS[i,2]))
                
                Pcalc.append(np.mean(Pmc))
                sigmaP.append(np.std(Pmc))

    else:
    
        if 'Temperature' in kwargs:
            for V,sigmaV,T,sigmaT,scale in zip(Volume,sigmaV,kwargs['Temperature'],kwargs['sigmaT'],scales):
                scalesmc = [scale] * 1000 #generate 1000 "samples" of the scale term.
                Vmc = np.random.normal(V,sigmaV,1000) #generate 1000 samples of volume, normally distributed
                Tmc = np.random.normal(T,sigmaT,1000) #generate 1000 samples of temperature, normally distributed
                
                Pmc = D07(np.array(Vmc),np.array(scalesmc),Temperature = Tmc) #calculate 1000 sample pressures
                Pcalc.append(np.mean(Pmc))
                sigmaP.append(np.std(Pmc))
                #print(Pcalc)
    
            
        else:
            for V,sigmaV,scale in zip(Volume,sigmaV,scales):
                scalesmc = [scale] * 1000 #generate 1000 "samples" of the scale term.
                Vmc = np.random.normal(V,sigmaV,1000) #generate 1000 samples of volume, normally distributed
                
                Pmc = D07(np.array(Vmc),np.array(scalesmc)) #calculate 1000 sample pressures
                Pcalc.append(np.mean(Pmc))
                sigmaP.append(np.std(Pmc))
    
    return Pcalc,sigmaP
        
        
#        Plist=[]
#        for i in range (0,2000):
#            Vtemp = np.random.normal(float(Volume[j]),float(sigmaV[j]))
#            Plist.append(np.array(Vtemp),)
        

