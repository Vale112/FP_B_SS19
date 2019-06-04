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
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const
from scipy.constants import physical_constants as pcon
from astropy.io import ascii
from tab2tex import make_table
from scipy.stats import sem
import pytemperature

if not os.path.isdir('build'):
    os.mkdir('build')
if not os.path.isdir('build/tables'):
    os.mkdir('build/tables')

#------------------------Untergrundrechnung 
print('--------------Untergrund-------------')
times1, timem1, current1, temp1 = np.genfromtxt('data/Messdaten_1K.txt', unpack=True)  #liest die Messdaten ein
times2, timem2, current2, temp2 = np.genfromtxt('data/Messdaten_2K.txt', unpack=True)  #liest die Messdaten ein
current1 *= 10**(-3)    #Umrechnen in 10^-11 A
temp1 *= 10**(-1)       # Umrechnen in °C
# E = unp.uarray(E, dE) # Erzeugt aus den Daten Messwerte inkl Fehler für die vereinfachte Fehlerrechnung mit uncertainties

#Berechne Heizrate
hr1 = list(map(lambda x,y: x-y, temp1[1:], temp1[:-1]))
hr2 = list(map(lambda x,y: x-y, temp2[1:], temp2[:-1]))
hr1m = np.mean(hr1)*2
hr2m = np.mean(hr2)*2
print("Heizrate1 per min:",hr1m, '±', sem(hr1))
print("Heizrate2 per min:",hr2m, '±', sem(hr2))

#Auswahl der Untergrunddaten
unterc1 = np.array([])
unterc1 = np.append(unterc1, current1[0:39])
unterc1 = np.append(unterc1, current1[41:45])
unterc1 = np.append(unterc1, current1[110:130])
untert1 = np.array([])
untert1 = np.append(untert1, temp1[0:39])
untert1 = np.append(untert1, temp1[41:45])
untert1 = np.append(untert1, temp1[110:130])

unterc2 = np.array([])
unterc2 = np.append(unterc2, current2[0:25])
unterc2 = np.append(unterc2, current2[72:80])
untert2 = np.array([])
untert2 = np.append(untert2, temp2[0:25])
untert2 = np.append(untert2, temp2[72:80])

#Fit des Untergrunds
def expo(T, a, b, c):           #Fitfunktion für den Untergrund
    return a * np.exp(b*T) + c 

params1, covariance1 = curve_fit(expo, untert1, unterc1)
errors1 = np.sqrt(np.diag(covariance1))
print('Parameter des Untergrundfits 1')
print('Amplitude a1 =', params1[0], '±', errors1[0])
print('Exponentenfaktor b1 =', params1[1], '±', errors1[1])
print('Achsenabschnitt c1 =', params1[2], '±', errors1[2])

params2, covariance2 = curve_fit(expo, untert2, unterc2)
errors2 = np.sqrt(np.diag(covariance2))
print('Parameter des Untergrundfits 2')
print('Amplitude a2 =', params2[0], '±', errors2[0])
print('Exponentenfaktor b2 =', params2[1], '±', errors2[1])
print('Achsenabschnitt c2 =', params2[2], '±', errors2[2])

#Erzeugt einen Plot der Messdaten
tplot1 = np.linspace(-65,58)
plt.plot(pytemperature.c2k(temp1), current1, 'rx', label='Messdaten') #Messpunkte
plt.plot(pytemperature.c2k(temp1[39:41]), current1[39:41], 'bx', label='ausgelassene Messdaten') #Messpunkte ohne Berücksichtigung, da ausreißer
plt.plot(pytemperature.c2k(untert1), unterc1, 'gx', label='Messdaten zur Untergrundsrechnung') #Messdaten für die Untergrundsrechnung
plt.plot(pytemperature.c2k(tplot1), expo(tplot1,*params1), 'y-', label='Untergrundfit') #Fitkurve für den Untergrund
plt.bar(pytemperature.c2k(temp1[45]), 0.7, width=0.3, color='m')
plt.bar(pytemperature.c2k(temp1[109]), 0.7, width=0.3, color='m')
plt.bar(255, 0.7, width=0.3, color='orange')
plt.xlabel(r'Temperatur $T\:/\: \mathrm{K}$')
plt.ylabel(r'Strom $I\:/\: 10^{-11}\mathrm{A}$')
plt.ylim(-0.05, 1)
plt.legend(loc='best')
plt.savefig('build/Messdaten1.pdf')
plt.clf()

tplot2 = np.linspace(-60,62)
plt.plot(pytemperature.c2k(temp2), current2, 'rx', label='Messdaten') #Messpunkte
plt.plot(pytemperature.c2k(untert2), unterc2, 'gx', label='Messdaten zur Untergrundsrechnung') #Messdaten für die Untergrundsrechnung
plt.plot(pytemperature.c2k(tplot2), expo(tplot2,*params2), 'y-', label='Untergrundfit') #Fitkurve für den Untergrund
plt.bar(pytemperature.c2k(temp2[25]), 1.5, width=0.3, color='m')
plt.bar(pytemperature.c2k(temp2[72]), 1.5, width=0.3, color='m')
plt.bar(pytemperature.c2k(temp2[61]), 1.5, width=0.3, color='orange')
plt.xlabel(r'Temperatur $T\:/\: \mathrm{°C}$')
plt.ylabel(r'Strom $I\:/\: 10^{-11}\mathrm{A}$')
plt.ylim(-0.05, 3)
plt.legend(loc='best')
plt.savefig('build/Messdaten2.pdf')
plt.clf()

#Untergrund wird von Messdaten subtrahiert und Bereich einschränken
current1 -= expo(temp1,*params1)
# current1 = current1[45:110]
# temp1 = temp1[45:110]
current2 -= expo(temp2,*params2)
# current2 = current2[25:72]
# temp2 = temp2[25:72]

#------------------------kleine Tempearuren 
print('--------------kl Temperaturen-------------')
lncurrent1 = np.log(current1[45:82])
lncurrent2 = np.log(current2[25:62])
kB = const.Boltzmann #Boltzmann Konstante

#Fit für Daten
def lin (x, a, b):
    return a*x+b

params1_2, covariance1_2 = curve_fit(lin, 1/pytemperature.c2k(temp1[45:82]), lncurrent1)
errors1_2 = np.sqrt(np.diag(covariance1_2))
print('Parameter der kl T 1')
print('Steigung a1 =', params1_2[0], '±', errors1_2[0])
print('Achsenabschnitt b1 =', params1_2[1], '±', errors1_2[1])
a1=unp.uarray(params1_2[0], errors1_2[0])
print('Aktivierungsenergie W1 =', -a1*kB)

params2_2, covariance2_2 = curve_fit(lin, 1/pytemperature.c2k(temp2[25:62]), lncurrent2)
errors2_2 = np.sqrt(np.diag(covariance2_2))
print('Parameter der kl T 2')
print('Steigung a2 =', params2_2[0], '±', errors2_2[0])
print('Achsenabschnitt b2 =', params2_2[1], '±', errors2_2[1])
a2=unp.uarray(params2_2[0], errors2_2[0])
print('Aktivierungsenergie W2 =', -a2*kB)

#Plott für kleine T 
plt.plot(1/pytemperature.c2k(temp1[45:82]), lncurrent1, 'rx', label='Messdaten Heizrate 1') #Messpunkte
plt.plot(1/pytemperature.c2k(np.linspace(-36.5, -18.2)), lin(1/pytemperature.c2k(np.linspace(-36.5, -18.2)),*params1_2), 'y-', label='Fit Heizrate 1') #Fitkurve für kleine T
plt.plot(1/pytemperature.c2k(temp2[25:62]), lncurrent2, 'mx', label='Messdaten Heizrate 2') #Messpunkte
plt.plot(1/pytemperature.c2k(np.linspace(-37, -11)), lin(1/pytemperature.c2k(np.linspace(-37, -11)),*params2_2), 'b-', label='Fit Heizrate 2') #Fitkurve für kleine T
plt.xlabel(r'Temperatur $1\:/\:T\:/\:1\:/\: \mathrm{K}$')
plt.ylabel(r'Strom $ln(I\:/\: 10^{-11}\mathrm{A})$')
plt.legend(loc='best')
plt.grid()
plt.savefig('build/kleineT.pdf')
plt.clf()

#------------------------große Tempearuren 
print('--------------gr Temperaturen-------------')
#Zuche T*, wo i(T*) ca 0
def nearest_toZero(array):
    # array2 = np.array([])
    # array2 = np.append(array2, array[array>0])
    idx = (np.abs(array - 0)).argmin()
    return array[idx]
print('1.i(T*) ungefähr 0 bei  T*=:', temp1[current1 == nearest_toZero(current1[temp1 > -18])])
print('2.i(T*) ungefähr 0 bei  T*=:', temp2[current2 == nearest_toZero(current2[temp2 > -10])])

#Bestimme y-Werte für Fit
def lnint(T, I, hr):
    array = np.array([])
    for t in T:
        if t == T[-1]:
            break
        array = np.append(array, np.log(np.abs(np.trapz(I[T>=t], T[T>=t]))/(np.abs(I[T == t])*hr)))
    return array

lnint1 = lnint(temp1[:125], current1[:125], hr1m)
lnint2 = lnint(temp2[:80], current2[:80], hr2m)

#Fit
params1_3, covariance1_3 = curve_fit(lin, 1/pytemperature.c2k(temp1[41:105]), lnint1[41:-19])
errors1_3 = np.sqrt(np.diag(covariance1_3))
print('Parameter der gr T 1 für einen Fit zwischen', 1/pytemperature.c2k(temp1[41]), 'und', 1/pytemperature.c2k(temp1[104]))
print('Steigung a1 =', params1_3[0], '±', errors1_3[0])
print('Achsenabschnitt b1 =', params1_3[1], '±', errors1_3[1])
a1=unp.uarray(params1_3[0], errors1_3[0])
b1=unp.uarray(params1_3[1], errors1_3[1])
print('Aktivierungsenergie W1 =', a1*kB)
print('Relaxationszeit T_0 =', unp.exp(b1))

params2_3, covariance2_3 = curve_fit(lin, 1/pytemperature.c2k(temp2[25:71]), lnint2[25:71])
errors2_3 = np.sqrt(np.diag(covariance2_3))
print('Parameter der gr T 2, für einen Fit zwischen', 1/pytemperature.c2k(temp2[25]), 'und', 1/pytemperature.c2k(temp2[70]))
print('Steigung a2 =', params2_3[0], '±', errors2_3[0])
print('Achsenabschnitt b2 =', params2_3[1], '±', errors2_3[1])
a2=unp.uarray(params2_3[0], errors2_3[0])
b2=unp.uarray(params2_3[1], errors2_3[1])
print('Aktivierungsenergie W2 =', a2*kB)
print('Relaxationszeit T_0 =', unp.exp(b2))


#Plott für große T 
plt.plot(1/pytemperature.c2k(temp1[:124]), lnint1, 'rx', label='nicht gefittete Messdaten Heizrate 1') #Messpunkte
plt.plot(1/pytemperature.c2k(temp1[41:105]), lnint1[41:105], 'bx', label='gefittetet Messdaten Heizrate 1') #Messpunkte
plt.plot(1/pytemperature.c2k(np.linspace(-62, 5)), lin(1/pytemperature.c2k(np.linspace(-62, 5)),*params1_3), 'y-', label='Fit Heizrate 1') #Fitkurve für große T
# plt.plot(1/pytemperature.c2k(temp2[:79]), lnint2, 'kx', label='nicht gefittete Messdaten Heizrate 2') #Messpunkte
# plt.plot(1/pytemperature.c2k(temp2[25:71]), lnint2[25:71], 'mx', label='gefittetet Messdaten Heizrate 2') #Messpunkte
# plt.plot(1/pytemperature.c2k(np.linspace(-62, 5)), lin(1/pytemperature.c2k(np.linspace(-62, 5)),*params2_3), 'c-', label='Fit Heizrate 2') #Fitkurve für kleine T
# plt.plot(1/pytemperature.c2k(temp2[25:62]), lncurrent2, 'mx', label='Messdaten Heizrate 2') #Messpunkte
# plt.plot(1/pytemperature.c2k(np.linspace(-37, -11)), lin(1/pytemperature.c2k(np.linspace(-37, -11)),*params2_2), 'b-', label='Fit Heizrate 2') #Fitkurve für kleine T
plt.xlabel(r'Temperatur $1\:/\:T\:/\:1\:/\: \mathrm{K}$')
plt.ylabel(r'$\ln\left(\int_{T}^{T*} i(T) \mathrm{d}\,T \:/\: i(T) \cdot b \right)$')
plt.legend(loc='best')
plt.grid()
plt.savefig('build/großeT1.pdf')
plt.clf()


#Plott für große T 
plt.plot(1/pytemperature.c2k(temp2[:79]), lnint2, 'rx', label='nicht gefittete Messdaten Heizrate 2') #Messpunkte
plt.plot(1/pytemperature.c2k(temp2[25:71]), lnint2[25:71], 'bx', label='gefittetet Messdaten Heizrate 2') #Messpunkte
plt.plot(1/pytemperature.c2k(np.linspace(-62, 5)), lin(1/pytemperature.c2k(np.linspace(-62, 5)),*params2_3), 'y-', label='Fit Heizrate 2') #Fitkurve für kleine T
# plt.plot(1/pytemperature.c2k(temp2[25:62]), lncurrent2, 'mx', label='Messdaten Heizrate 2') #Messpunkte
# plt.plot(1/pytemperature.c2k(np.linspace(-37, -11)), lin(1/pytemperature.c2k(np.linspace(-37, -11)),*params2_2), 'b-', label='Fit Heizrate 2') #Fitkurve für kleine T
plt.xlabel(r'Temperatur $1\:/\:T\:/\:1\:/\: \mathrm{K}$')
plt.ylabel(r'$\ln\left(\int_{T}^{T*} i(T) \mathrm{d}\,T \:/\: i(T) \cdot b \right)$')
plt.legend(loc='best')
plt.grid()
plt.savefig('build/großeT2.pdf')
plt.clf()




#  # Erzeugt eine Tabelle aus den gegebenen Daten, beachte E ist Fehlerbehaftet
#  make_table(
#          header = [' $E_\\text{i}$ / \kilo\electronvolt', '$i$'],
#          data = [E, data],
#          places = [(4.4, 1.4), 4.0],
#          caption = 'Gegebene Werte zur Kalibrierung des Germanium-Detektors \cite{referenz1}.',
#          label = 'tab:zuordnung_eu',
#          filename = 'build/tables/zuordnung_Eu.tex'
#          )
#  
#  #Lineare Funktion für Ausgleichsgeraden
#  def lin(x,m,b):
#      return m*x+b
#  
#  # Linerare Fit mit gefitteten Kanalnummern zu Energie
#  params, covariance= curve_fit(lin,noms(index_f),noms(E))
#  errors = np.sqrt(np.diag(covariance))
#  print('Kalibrationswerte mit gefitteten Kanälen:')
#  print('Steigung m =', params[0], '±', errors[0])
#  print('Achsenabschnitt b =', params[1], '±', errors[1])
#  #Zusammenfassen der Werte und Ungenauigkeiten der Fit-Parameter
#  m=ufloat(params[0],errors[0])
#  b=ufloat(params[1],errors[1])
#  
#  #Erzeugt einen Plot der Messdaten
#  x=np.linspace(1,8192,8192)
#  plt.bar(x, data, label='Balken')     #Histogrambalken
#  plt.plot(x, data, 'rx', label='Messdaten') #Messpunkte
#  plt.plot(x, lin(x, *params), 'y-', label='Fit') #Ausgleichsrechnung
#  plt.xlim(0, 4000)
#  plt.xlabel(r'Kanalnummer $i$')
#  plt.ylabel(r'Zählrate $N$')
#  plt.legend(loc='best')
#  plt.yscale('log')
#  plt.savefig('build/Eu_log_Kanal.pdf')
#  plt.clf()
#  
#  
#  
#  
#  #--------------------Aufgabenteil b)
#  
#  #Erzeugt ein leeres Array und berechnet dann die Werte, welche nach und nach eingetragen werden
#  Q=[]
#  Z=[]
#  for i in range(len(data)):
#   Z.append(np.sqrt(2*np.pi)*E[i]*data[i])
#   Q.append(Z[i]/(E[i]/100*4676))
#  
#  # Ruft den letzten wert eines Arrays ab, und gibt diesen aus
#  e_photo=E[-1]
#  print('Photo: ', e_photo)