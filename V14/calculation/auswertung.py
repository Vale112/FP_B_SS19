# coding=utf-8
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import sympy
from uncertainties import correlated_values, correlation_matrix
from scipy.integrate import quad
from uncertainties.unumpy import (nominal_values as noms, std_devs as sdevs)
import scipy.constants as con
from scipy.constants import physical_constants as pcon
from scipy.signal import find_peaks
from astropy.io import ascii
from tab2tex import make_table

if not os.path.isdir('build'):
    os.mkdir('build')
if not os.path.isdir('build/tables'):
    os.mkdir('build/tables')

#------------------------Nullmessung 
data = np.genfromtxt('data/nullmessung', unpack=True)  #liest die Messdaten ein
x=np.linspace(1,len(data),len(data))
plt.plot(x,data)
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.savefig("build/Nullmessung.pdf")
plt.clf()

def lin(x,m,b):
    return m*x+b
params, covariance= curve_fit(lin,[],)





#------------------------Würfel 1
print('--------------Würfel 1-------------')

I_1=np.array([93.2, 36.5, 108.8, 63.2, 45.6, 66.2, 57.8, 34.8, 64, 65.8, 61.1, 49.1])
I_1_err=np.array([2.4, 1.7, 2.5, 2.1, 2, 2.3, 2.1, 1.8, 2.2, 2.3, 3.1, 2.0])
def ac(I,I_err):
    A =[[1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0,1, 0, 0, 1,0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, np.sqrt(2), 0, np.sqrt(2), 0, 0, 0, 0, 0],
    [0, 0, np.sqrt(2), 0, np.sqrt(2), 0, np.sqrt(2), 0, 0],
    [0, 0, 0, 0, 0, np.sqrt(2), 0, np.sqrt(2), 0],
    [0, 0, 0, np.sqrt(2), 0, 0, 0, np.sqrt(2), 0],
    [np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0, np.sqrt(2)],
    [0, np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0]]
    A_T=np.transpose(A)    

    I_0=110#Eingangsintensität
    I_0_err=1.7

    N=np.log(I_0/I)
    temp=np.linalg.inv(np.dot(A_T,A))
    ac=np.dot(temp,np.dot(A_T,N))
    print("ac=",ac)
    sigma_I=np.sqrt((I_0_err/I_0)**2+(I_err/I)**2)
    sigma_I_sqrd=np.dot(sigma_I,sigma_I)
    C=sigma_I_sqrd*temp
    ac_err=np.sqrt(np.diag(C))
    print("ac_err=",ac_err)
    print("Mittelwert: ",np.sum(ac)/np.size(ac),"+-",np.sum(ac_err)/np.size(ac_err))
    return 0
ac(I_1,I_1_err)
#N_1=np.log(I/I_1)
#temp=np.linalg.inv(np.dot(A_T,A))
#ac=np.dot(temp,np.dot(A_T,N_1))
#print("ac=",ac)
#sigma_I_1=np.sqrt((I_1_err/I_1)**2+(I_err/I)**2)#Fehler sigma_i
#print(sigma_I_1)
#sigma_I_1_sqrd=np.dot(sigma_I_1,sigma_I_1)
#C=sigma_I_1_sqrd*temp
#ac_err=np.sqrt(np.diag(C))
#print("ac_err=",ac_err)


#------------------------Würfel 2
print('--------------Würfel 2-------------')
I_2=np.array([9.4, 9.5, 9.4, 9.4, 9.5, 9.4, 11.3, 7.2, 11.3, 11.3, 7.2, 11.3])
I_2_err=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 2.8, 0.2, 2.8, 2.8, 0.2, 2.8])
ac(I_2,I_2_err)

#------------------------Würfel 3
print('--------------Würfel 3-------------')
I_3=np.array([38.6, 36.5, 38.6, 38.6, 36.5, 38.6, 48, 37.5, 48, 48, 37.5, 48])
I_3_err=np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.6, 0.7, 0.7, 0.6, 0.7])
ac(I_3,I_3_err)

#------------------------Würfel 5
print('--------------Würfel 5-------------')
I_5=np.array([9.4, 27, 25.9, 14.6, 24.7, 15, 20.9, 14.9, 16.2, 46, 12.8, 13.2])
I_5_err=np.array([0.3, 0.5, 0.4, 0.4, 0.4, 0.3, 0.4, 0.4, 0.4, 0.6, 0.3, 0.3])
ac(I_5,I_5_err)
