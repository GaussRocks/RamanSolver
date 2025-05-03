import numpy as np
from scipy import constants

c = constants.speed_of_light*100 #cm/s
pi = constants.pi
gRmax = 5.014860716573898e-12
 
hRpar = np.array([
[56.25, 1.00, 52.10, 17.37],
[100.00, 11.40, 110.42, 38.81],
[231.25, 36.67, 175.00, 58.33],
[362.50, 67.67, 162.50, 54.17],
[463.00, 74.00, 135.33, 45.11],
[497.00, 4.50, 24.50, 8.17],
[611.50, 6.80, 41.50, 13.83],
[691.67, 4.60, 155.00, 51.67],
[793.67, 4.20, 59.50, 19.83],
[835.50, 4.50, 64.30, 21.43],
[930.00, 2.70, 150.00, 50.00],
[1080.00, 3.10, 91.00, 30.33],
[1215.00, 3.00, 160.00, 53.33]])

def gRintegral(dfi, gRpeak):
    w = 2*pi*(dfi)
    Nt=2**16
    ti = 0
    tf = 1e-11
    dt = (tf-ti)/(Nt-1)
    t = np.linspace(ti, tf, Nt) #s
    Sn = 0
    for jj in range(0,len(hRpar)): #Integral sweep
        A = hRpar[jj][1]
        gamma = (pi*c*hRpar[jj][3])
        Gamma = (pi*c*hRpar[jj][2])
        Wvi = (2*pi*c*hRpar[jj][0])
        integrand = (A/2)*(np.cos((Wvi-w)*t) - np.cos((Wvi+w)*t))*np.exp(-gamma*t)*np.exp(-(Gamma**2*t**2)/4)  
        integral = np.trapezoid(y=integrand, dx=dt)
        Sn = Sn + integral
    return (Sn/gRmax)*gRpeak

def gRspectrum(lbd_i, lbd_f, N):
    lbd = np.linspace(lbd_i, lbd_f, N)
    f = (c/100)/lbd[::-1]
    df = (f-f[0])
    gR=[]
    for dfi in df: #frequency sweep
        gR.append(gRintegral(dfi, 1))   
    return df, np.array(gR)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    wl, gr = gRspectrum(1420e-9, 1850e-9, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wl/c, gr)
    ax.set(xlim=[0,1500], ylim=[0,1.25], 
           xlabel=r'$\Delta$ Wave number [$cm^{-1}$]',
           ylabel='Normalized Raman gain')
    ax.set_yticks(np.arange(0,1.5,0.25))
    ax.set_xticks(np.arange(0,1750, 250))
