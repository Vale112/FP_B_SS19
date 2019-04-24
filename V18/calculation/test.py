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

#------------------------Aufgabenteil a) {Untersuchung des Eu-Spektrums}
data = np.genfromtxt('data/Eu152.txt', unpack=True)
# Ener, Wahr = np.genfromtxt('data/2_0/Test.txt', unpack=True)
E, W, peaks_ind = np.genfromtxt('data/2_0/Eu.txt', unpack=True)

#verwendete und nicht verwendete peaks
peaks_v = [309, 614, 740, 861, 1027, 1108, 1939, 2159, 2399, 2702, 2765, 3500] #entsprechen auch peaks_ind
peaks_n = [106, 218, 919]

#Energieeichung: Wird bei jeder Betrachtung eines Spektrums benötigt
#Lineare Funktion für Ausgleichsgeraden
def lin(x,m,b):
    return m*x+b

#Linearer Fit mit Augabe der Parameter
params, covariance= curve_fit(lin,peaks_ind,E)
errors = np.sqrt(np.diag(covariance))

#Zusammenfassen der Werte und Ungenauigkeiten der Fit-Parameter
m=ufloat(params[0],errors[0])
b=ufloat(params[1],errors[1])

#------------Berechnen der Detektoreffizenz
#Berechnung der Aktivität am Messtag
A=ufloat(4130,60) #Aktivität Europium-Quelle am 01.10.2000
t_halb = ufloat(4943,5) #Halbwertszeit Europium in Tagen
dt = 18*365.25 + 194 #Zeitintervall in Tagen
A_jetzt=A*unp.exp(-unp.log(2)*dt/t_halb)#Aktivität Versuchstag
print('Aktivität zum Messzeitpunkt',A_jetzt)
#Gauß-Funktion für Peakhoehenbestimmung
def gauss(x,sigma,h,a,mu):
 return a+h*np.exp(-((x-mu)/sigma)**2)
#Verwende Gauß-Fit in jedem Bin des Spektrums um Peakhöhe zu erhalten
def gaussian_fit_peaks(test_ind):
 peak_inhalt = []
 index_fit = []
 hoehe = []
 unter = []
 sigma = []
 for i in test_ind:
     a=i-40
     b=i+40
     params_gauss,covariance_gauss=curve_fit(gauss,np.arange(a,b+1),data[a:b+1],p0=[1,data[i],0,i-1])
     errors_gauss = np.sqrt(np.diag(covariance_gauss))
     sigma_fit=ufloat(params_gauss[0],errors_gauss[0])
     h_fit=ufloat(params_gauss[1],errors_gauss[1])
     a_fit=ufloat(params_gauss[2],errors_gauss[2])
     mu_fit=ufloat(params_gauss[3],errors_gauss[3])
     #print(h_fit*sigma_fit*np.sqrt(2*np.pi)
     index_fit.append(mu_fit)
     hoehe.append(h_fit)
     unter.append(a_fit)
     sigma.append(sigma_fit)
     peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
 return index_fit, peak_inhalt, hoehe, unter, sigma
index_f, peakinhalt, hoehe, unter, sigma = gaussian_fit_peaks(peaks_ind.astype('int'))
E_det =[]
for i in range(len(index_f)):
    E_det.append(lin(index_f[i],*params))
 
#Berechnung des Raumwinkels
a=ufloat(0.073+0.015, 0.001) #in m
r=ufloat(0.0225, 0) #in m
omega_4pi = (1-a/(a**2+r**2)**(0.5))/2
print('Raumwinkel',omega_4pi)

#Berechnung Detektoreffizienz für jeden Energiepeak
Q=[]
Z=[]
for i in range(len(W)):
 Z.append(np.sqrt(2*np.pi)*hoehe[i]*sigma[i])
 Q.append(Z[i]/(omega_4pi*A_jetzt*W[i]/100*4676))


#Potenzfunktion für Fit
def potenz(x,a,b,c,e):
 return a*(x-b)**e+c

#Durchführung des Exponential-Fits und Ausgabe der Parameter
#print('Daten für den Exponentialfit:')
#print(noms(Q), noms(E_det))
params2, covariance2= curve_fit(potenz,noms(E_det),noms(Q),sigma=sdevs(Q), p0=[1, 0.1, 0, 0.5])
errors2 = np.sqrt(np.diag(covariance2))
#Zusammenfassen der Fit-Parameter
a=ufloat(params2[0],errors2[0])
b=ufloat(params2[1],errors2[1])
c=ufloat(params2[2],errors2[2])
e=ufloat(params2[3],errors2[3])

#-----------------------Teilaufgabe b) {Untersuchung des Cs-Spektrums}
data_b = np.genfromtxt('data/Cs137.txt', unpack=True)
x_plot = np.linspace(0, len(data_b), len(data_b))

#Finde Peaks in Spektrum und ordne sie der Energie zu
peaks_2 = find_peaks(data_b, height=60, distance=20)
indexes_2 = peaks_2[0]
peak_heights_2 = peaks_2[1]
energie_2 = lin(indexes_2, *params)

# print('Indices der peaks: ', indexes_2)
#Identifiziere die charakteristischen Energie-Peaks
e_photo=energie_2[-1]
i_photo=indexes_2[-1]
print('Photo: ', np.round(e_photo, 4), i_photo)
e_compton=energie_2[-2]
i_compton=indexes_2[-2]
print('Compton: ', np.round(e_compton, 4), i_compton)
e_rueck=energie_2[14]
i_rueck=indexes_2[14]
print('Rueckstreu: ', np.round(e_rueck, 4), i_rueck)

#print(len(energie_2), len(indexes_2))
#print(e_rueck, e_compton, e_photo)
#print(indexes_2[-4], indexes_2[-2], indexes_2[-1])
e_photo_t = 661.59 #theoretischer Photoenergiepeak
m_e = 511   #Elektronenmasse in keV
#Vergleiche zwischen gemessenen und theoretischen Werten der Peaks
e_compton_theo = 2*e_photo_t**2/(m_e*(1+2*e_photo_t/m_e))
vgl_compton = 1-e_compton/e_compton_theo
print(f'Ein Vergleich des theoretischen E_compton', np.round(e_compton_theo, 4), 'mit dem gemessenen E_compton', np.round(e_compton, 4), 'beträgt:', np.round(vgl_compton, 4))
e_rueck_theo = e_photo_t/(1+2*e_photo_t/m_e)
vgl_rueck = 1-e_rueck/e_rueck_theo
print(f'Ein Vergleich des theoretischen E_rueck', np.round(e_rueck_theo, 4), 'mit dem gemessenen E_compton', np.round(e_rueck, 4), 'beträgt:', np.round(vgl_rueck, 4))

#Führe wieder Gausß-Fit für den Vollenergiepeak durch, um Peakhöhe bestimmen zu können
a=indexes_2[-1].astype('int')-50
b=indexes_2[-1].astype('int')+50
params_gauss_b,covariance_gauss_b=curve_fit(gauss,lin(np.arange(a,b+1),*params),data_b[a:b+1],p0=[1,data_b[indexes_2[-1]],0,lin(indexes_2[-1]-0.1,*params)])
errors_gauss_b = np.sqrt(np.diag(covariance_gauss_b))
#Fasse Wert und Ungenauigkeit der Fit-Parameter wieder jeweils zusammen
sigma_fit=ufloat(params_gauss_b[0],errors_gauss_b[0])
h_fit=ufloat(params_gauss_b[1],errors_gauss_b[1])
a_fit=ufloat(params_gauss_b[2],errors_gauss_b[2])
mu_fit=ufloat(params_gauss_b[3],errors_gauss_b[3])
Z_photo=h_fit*sigma_fit*np.sqrt(2*np.pi)
print(f'Die Spitze des Vollenergiepeaks liegt bei {Z_photo} keV.')
#-------------------------------------------------------------------------------
print('\nVergleich Halb- zu Zehntelwertsbreite:')
#lin beschreibt noch die lineare Regression vom beginn der Auswertung
h_g = ufloat(2.2, 0.1)
print('Halbwertsbreite Gemessen: ', h_g)
h_t = np.sqrt(8*np.log(2))*sigma_fit
print('Halbwertsbreite Theorie: ', h_t)
print('Rel. Fehler der Halbwertsbreiten Werte: ', (h_g-h_t)/h_t)
z_g = ufloat(3.9, 0.1)
print('Zehntelbreite Gemessen: ', z_g)
z_t = np.sqrt(8*np.log(10))*sigma_fit
print('Zehntelbreite Theorie: ', z_t)
print('Rel. Fehler der Zehntelbreiten: ', (z_g-z_t)/z_t)
print('Verhältnis zwischen gemessener', z_g/h_g, 'und theoretischer', z_t/h_t,'Breiten \n')
#-------------------------------------------------------------------------------
inhalt_photo = ufloat(sum(data_b[i_photo-13:i_photo+9]*noms(sigma_fit)*np.sqrt(2*np.pi)), sum(np.sqrt(data_b[i_photo-13:i_photo+9]*noms(sigma_fit)*np.sqrt(2*np.pi))))
print('Inhalt des Photo-Peaks: ', inhalt_photo)
min_ind_comp = 53
inhalt_comp = ufloat(sum(data_b[min_ind_comp:i_compton]*noms(sigma_fit)*np.sqrt(2*np.pi)), sum(np.sqrt(data_b[min_ind_comp:i_compton]*noms(sigma_fit)*np.sqrt(2*np.pi))))
print(f'Der Inhalt des Compton-Kontinuums, liegt bei: {inhalt_comp}')
mu_ph = ufloat(0.007, 0.003) #in cm^-1, abgelesen aus Diagramm
mu_comp = ufloat(0.36, 0.07)
l=3.9   #Länge des Detektors
abs_wahrsch_ph = 1-unp.exp(-mu_ph*l)
abs_wahrsch_comp = 1-unp.exp(-mu_comp*l)
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks liegt bei: {abs_wahrsch_ph} Prozent')
print(f'Die absolute Wahrscheinlichkeit eine Comptonpeaks liegt bei: {abs_wahrsch_comp} Prozent\n')

#-------------Aufgabenteil e) {Was das? Gucken wir mal}
data_e = np.genfromtxt('data/salz.txt', unpack=True)
# peaks_4 = find_peaks(data_e, height=75, distance=30)
# indexes_4 = peaks_4[0]
# peak_heights_4 = peaks_4[1]
# energie_4 = lin(indexes_4,*params)
indexes_plot=np.array([164, 236, 319, 468, 551, 607, 739, 880, 1518, 1912, 2324, 2788, 3424, 4386], dtype='int')
indexes_plot.astype('int')
x=np.linspace(1,8192,8192)
plt.bar(lin(x,*params), data_e,label='Messdaten')
plt.plot(lin(indexes_plot, *params),data_e[indexes_plot],'rx',label='verwendete Peaks')
plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.xlim(0, 2500)
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig('build/Uran.pdf')
plt.clf()
# make_table(
#     header=['Index $i$', '$Z_\\text{i}$', '$E_\\text{i}$ / \kilo\electronvolt'],
#     data= [indexes_plot, data_e[indexes_plot], lin(indexes_plot, *params)],
#     places=[4.0, 3.1, 4.2],
#     caption ='Zugeordnete Indizes, Zählrate $Z_\\text{i}$ und Energie $E_\\text{i}$ der gefundenen Peaks.',
#     label='tab:last',
#     filename ='build/tables/last.tex'
# )
def gaussian_fit_peaks_e(test_ind):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    unter = []
    sigma = []
    for i in test_ind:
        a=i-10
        b=i+10
        params_gauss_e,covariance_gauss_e=curve_fit(gauss,np.arange(a,b+1),data_e[a:b+1],p0=[1,data_e[i],0,i-1])
        errors_gauss_e = np.sqrt(np.diag(covariance_gauss_e))
        sigma_fit=ufloat(params_gauss_e[0],errors_gauss_e[0])
        h_fit=ufloat(params_gauss_e[1],errors_gauss_e[1])
        a_fit=ufloat(params_gauss_e[2],errors_gauss_e[2])
        mu_fit=ufloat(params_gauss_e[3],errors_gauss_e[3])
        #print(h_fit*sigma_fit*np.sqrt(2*np.pi))
        if i == 3315:
            plt.plot(np.arange(a, b+1), data_e[a:b+1], label='Daten')
            #plt.plot(np.arange(a, b+1), gauss(np.arange(a, b+1), *params_gauss_d), label='Fit')
            plt.savefig('build/test.pdf')
            plt.clf()
        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit, peak_inhalt, hoehe, unter, sigma

E_e, W_e, peaks_ind_e = np.genfromtxt('data/2_0/salz_Zuordnung.txt', unpack=True)
index_e, peakinhalt_e, hoehe_e, unter_e, sigma_e = gaussian_fit_peaks_e(peaks_ind_e.astype('int'))
# print(f'Peakinhalt {peakinhalt_e} Hoehe {hoehe_e}, Sigma {sigma_e}')
E_e_det = []
for i in range(len(W_e)):
    E_e_det.append(lin(index_e[i], *params))
Z_e=[]
Q_e=[]
A_e=[]
for i in range(3, len(W_e)):
    Z_e.append(np.sqrt(2*np.pi)*hoehe_e[i]*sigma_e[i])
    A_e.append(Z_e[i-3]/(4510*omega_4pi*W_e[i]/100*potenz(noms(E_e_det[i]),*params2))) # für Index 2 wird Potenz negativ wieso? dadurch Aktivität etc negativ
    Q_e.append(Z_e[i-3]/(omega_4pi*A_e[i-3]*W_e[i]/100*4510))
# print(E_e_det)
# print(potenz(noms(E_e_det[11]),*params2))
print(f'\nDaten zur Berechnung der Akivität: {E_e}, {params2}, \n den Peakinhalt Z {Z_e},\n die Effizienz Q {Q_e} \n und der Aktivität {A_e}')
print('gemittelte Aktivität für Cobalt: ', np.mean(A_e[0:9]))

make_table(
    header= ['$W\/\%$', 'Q', '$Z_i$', '$E_i$ / \kilo\electronvolt', '$A_i$ / \\becquerel'],
    data=[W_e[3:], Q_e ,Z_e, E_e[3:], A_e],
    places=[2.2, (1.3, 1.3), (4.2 , 3.2), 4.1, (3.0, 2.0)],
    caption='Berechnete Aktivität der betrachteten Emissionslinien mit dazu korrespondierenden Detektor-Effizienzen.',
    label='tab:aktivitaet_e',
    filename='build/tables/aktivitaet_e.tex'
)