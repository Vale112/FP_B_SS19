# coding=utf-8
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
from matplotlib import rc

#if not os.path.isdir('build'):
#    os.mkdir('build')
#if not os.path.isdir('build/tables'):
#    os.mkdir('build/tables')

sw1, h1, sw2, h2 = np.genfromtxt('calculation/data.txt', unpack=True)
def B_sw(x):
    sw_off=0.012
    return sc.mu_0*(8*11/(0.1639*np.sqrt(125)))*(x+sw_off)

def B_h(x):
    h_off=0.024
    return sc.mu_0*(8*154/(0.1579*np.sqrt(125)))*(x+h_off)
#Erstelle Array mit B-Werte
B1=B_sw(sw1*0.1)+B_h(h1*0.3)
B2=B_sw(sw2*0.1)+B_h(h2*0.3)
#Runde Werte ab
B1=np.around(B1,8)
B2=np.around(B2,8)
#Korrigiere Werte mit ausgeschalteter Spule
B1[0]=33.98e-6
B2[0]=41.28e-6

print("Resonz1:",B1)
print("Resonz2:",B2)


x=np.linspace(0.1,1.0,num=10,endpoint="true")

#Regression
def lin_reg(x,a,b):
    return a*x+b
popt1, pcov1 = curve_fit(lin_reg,x,B1)
perr1 = np.sqrt(np.diag(pcov1))
print("b1_fit: ",popt1,perr1)
popt2, pcov2 = curve_fit(lin_reg,x,B2)
perr2 = np.sqrt(np.diag(pcov2))
print("b2_fit: ",popt2,perr2)
#Plots
plt.plot(x,B1*1e6,"rx",label="1. Peak")
plt.plot(x,B2*1e6,"bx",label="2.Peak")
plt.plot(x,lin_reg(x,popt1[0],popt1[1])*1e6,label="Fit 1.Peak")
plt.plot(x,lin_reg(x,popt2[0],popt2[1])*1e6,label="Fit 2.Peak")
plt.xlabel(r'$\nu$ (MHz)')
plt.ylabel(r'$B$ (µT)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("build/B.pdf")

#Lande-Faktoren
a1=ufloat(popt1[0],perr1[0])
a2=ufloat(popt2[0],perr2[0])
g1=sc.h/(9.274009994e-24*a1*1e-6) 
print("g1=",g1)
g2=sc.h/(9.274009994e-24*a2*1e-6) 
print("g2=",g2)
print("g1/g2:",g1/g2)
#Magnetfeld
b1=ufloat(popt1[1],perr1[1])
b2=ufloat(popt2[1],perr2[1])
print("b_hor:",(b1+b2)/2)
#Spins
def I(g):
    a=2.0023#gj
    k=unp.sqrt((a/(4*g)-1)**2 +3*a/(4*g)-3/4)
    return a/(4*g)-1 +k

print("I1:",I(g1))
print("I2:",I(g2))

#QZeeman
def U(g,b,e):
    mag=9.274009994e-24
    return g*mag*b+g**2 *mag**2 *b**2*1/e
e87=4.53e-24 #1.resost
e85=2.01e-24 #2. resost
print("u1:", U(g1,B1[9],e87))
print("u2:", U(g2,B2[9],e85))