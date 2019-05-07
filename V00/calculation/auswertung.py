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

#------------------------Aufgabenteil a) 
print('--------------Aufgabenteil a)-------------')
data, E, dE = np.genfromtxt('data/Eu152.txt', unpack=True)  #liest die Messdaten ein
E = unp.uarray(E, dE) # Erzeugt aus den Daten Messwerte inkl Fehler für die vereinfachte Fehlerrechnung mit uncertainties

# Erzeugt eine Tabelle aus den gegebenen Daten, beachte E ist Fehlerbehaftet
make_table(
        header = [' $E_\\text{i}$ / \kilo\electronvolt', '$i$'],
        data = [E, data],
        places = [(4.4, 1.4), 4.0],
        caption = 'Gegebene Werte zur Kalibrierung des Germanium-Detektors \cite{referenz1}.',
        label = 'tab:zuordnung_eu',
        filename = 'build/tables/zuordnung_Eu.tex'
        )

def lin(x,m,b):
    return m*x+b

#Erzeugt einen Plot der Messdaten
x=np.linspace(1,8192,8192)
plt.bar(x, data, label='Balken')     #Histogrambalken
plt.plot(x, data, 'rx', label='Messdaten') #Messpunkte
plt.plot(x, lin(x), 'y-', label='Fit') #Ausgleichsrechnung
plt.xlim(0, 4000)
plt.xlabel(r'Kanalnummer $i$')
plt.ylabel(r'Zählrate $N$')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('build/Eu_log_Kanal.pdf')
plt.clf()

#Energieeichung: Wird bei jeder Betrachtung eines Spektrums benötigt
#Lineare Funktion für Ausgleichsgeraden
def lin(x,m,b):
    return m*x+b

# Linerare Fit mit gefitteten Kanalnummern zu Energie
params, covariance= curve_fit(lin,noms(index_f),noms(E))
errors = np.sqrt(np.diag(covariance))
print('Kalibrationswerte mit gefitteten Kanälen:')
print('Steigung m =', params[0], '±', errors[0])
print('Achsenabschnitt b =', params[1], '±', errors[1])
#Zusammenfassen der Werte und Ungenauigkeiten der Fit-Parameter
m=ufloat(params[0],errors[0])
b=ufloat(params[1],errors[1])


#--------------------Aufgabenteil b)

#Erzeugt ein leeres Array und berechnet dann die Werte, welche nach und nach eingetragen werden
Q=[]
Z=[]
for i in range(len(data)):
 Z.append(np.sqrt(2*np.pi)*E[i]*data[i])
 Q.append(Z[i]/(E[i]/100*4676))

# Ruft den letzten wert eines Arrays ab, und gibt diesen aus
e_photo=E[-1]
print('Photo: ', e_photo)