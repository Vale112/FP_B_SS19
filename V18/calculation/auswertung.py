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

#------------------------Aufgabenteil a) {Untersuchung des Eu-Spektrums}
data = np.genfromtxt('data/Eu152.txt', unpack=True)
# Ener, Wahr = np.genfromtxt('data/2_0/Test.txt', unpack=True)
E, W, peaks_ind = np.genfromtxt('data/2_0/Eu_Zuordnung.txt', unpack=True)
make_table(
        header = [' $E$ / \kilo\electronvolt', ' $W$ / \%', 'Kanalnummer $i$'],
        data = [E, W, peaks_ind],
        places = [4.0, 2.1, 4.0],
        caption = 'Gegebene Werte zur Kalibrierung des Germanium-Detektors \cite{referenz2}. Aufgelistet sind die jeweilige Energie, die Emissionswahrscheinlichkeit $W$ und die zugeordnete Kanalnummer $i$.',
        label = 'tab:zuordnung_eu',
        filename = 'build/tables/zuordnung_Eu.tex'
        )
# Findet Peaks, zuordnung dann manuelle
# peaks = find_peaks(data, height=80 , distance=2)
# indexes = peaks[0]
# peak_heights = peaks[1]
# print(indexes)

#verwendete und nicht verwendete peaks
peaks_v = [309, 614, 740, 861, 1027, 1108, 1939, 2159, 2399, 2702, 2765, 3500] #entsprechen auch peaks_ind
peaks_n = [106, 218, 919]

#plot von EU mit Kanälen
x=np.linspace(1,8192,8192)
plt.bar(x, data, label='Messdaten')
plt.plot(peaks_v, data[peaks_v], 'rx', label='Verwendete Peaks')
plt.plot(peaks_n, data[peaks_n], 'yx', label='Nicht verwendete Peaks')
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

#Linearer Fit mit Augabe der Parameter
params, covariance= curve_fit(lin,peaks_ind,E)
errors = np.sqrt(np.diag(covariance))
print('Kalibrationswerte:')
print('Steigung m =', params[0], '±', errors[0])
print('Achsenabschnitt b =', params[1], '±', errors[1])

#Zusammenfassen der Werte und Ungenauigkeiten der Fit-Parameter
m=ufloat(params[0],errors[0])
b=ufloat(params[1],errors[1])

#Plotten des vom Detektor aufgenommenen Spektrums + logarithmische y-Achse evtl mit Enegrie
x=np.linspace(1,8192,8192)
plt.bar(lin(x,*params), data, label='Messdaten')
plt.xlim(0, 1600)
plt.xlabel(r'Energie $E$')
plt.ylabel(r'Zählrate $N$')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('build/Eu_log_Energie.pdf')
plt.clf()

#Plotten der Eichung/Kalibrierung am Eu-Spektrum
x=np.linspace(250,3700,3450)
plt.plot(x, lin(x,*params),'r-',label='Fit')
plt.errorbar(peaks_ind, E, yerr=20, fillstyle= None, fmt=' x', label='Daten')
plt.ylim(0,1500)
plt.xlim(0, 4000)
plt.xlabel(r'Kanalnummer $i$')
plt.grid()
plt.ylabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.legend()
plt.savefig('build/kalibration.pdf')
plt.clf()

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

#Erstellen einer Tabelle der Fit-Parameter des Gauß-Fits
make_table(
 header= ['$i$', '$\mu_i$', '$a$', '$h_i$', '$\sigma_i$'],
 data=[peaks_v, index_f, unter, hoehe, sigma],
 caption='Parameter des durchgeführten Gauss-Fits pro Kanal.',
 label='tab:gauss_parameter',
 places=[3.0, (4.2, 1.2), (2.2, 1.2), (4.2, 2.2), (3.2, 1.2)],
 filename='build/tables/Gauss-Fit-Parameter.tex'
 )

#Erstellen einer Tabelle der Detektoreffizenz und den dazu verwendeten Werten
make_table(
 header=['$Z_i$', '$E_i$ / \kilo\electronvolt' ,'W/\%', '$Q_i$ / \\becquerel '],
 data=[Z, E_det, W, Q],
 caption = 'Peakhöhe, Energie und Detektoreffizenz als Ergebnis des Gaußfits.',
 label = 'tab:det_eff',
 places = [ (5.2, 3.2), (4.2, 1.2), 2.1, (1.2, 1.2)],
 filename = 'build/tables/det_eff.tex'
 )

#Betrachte Exponential-Fit für Beziehnung zwischen Effizienz und Energie
# Lasse erste Werte weg
# Q=Q[1:]
# E=E[1:]
# E_det=E_det[1:]

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

#Ausgabe der Fit-Parameter
print('\nKalibrationswerte Potenzfunktion:')
print(f'Steigung a = {a}')
print(f'Verschiebung_x b = {b}')
print(f'Verschiebung_y c = {c}')
print(f'Exponent e = {e}')

#Plotten der Effizenz gegen die Energie mit Exponential-Fit-Funktion
x=np.linspace(1,1600,10000)
plt.plot(x, potenz(x,*params2),'r-',label='Fit')
plt.errorbar(E,noms(Q), yerr=sdevs(Q),fmt=' x', ecolor='b',label='Daten')
plt.legend()
plt.xlabel(r'Energie $E \:/\: \mathrm{keV}$')
plt.grid()
plt.ylabel(r'Effizentz $Q(E)$')
plt.savefig('build/efficiency.pdf')
plt.clf()


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

#Plotten des vom Detektor aufgenommenen Cs-Spektrums + logarithmische y_Achse
plt.bar(lin(x_plot, *params), data_b, label='Messwerte')
plt.plot(lin(indexes_2[14], *params), energie_2[14], 'rx', label='Rückstreupeak')
plt.bar(lin(indexes_2[-2], *params), energie_2[-2], label='Comptonkante')
plt.plot(lin(indexes_2[-1], *params), energie_2[-1], 'yx', label='Vollenergiepeak')
plt.xlim(0, 800)
plt.xlabel(r'Energie $E \:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('build/Cs_log.pdf')
plt.clf()

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
Z_3 = data_b[a:b+1]*sigma_fit*np.sqrt(2*np.pi)
plt.plot(lin(np.arange(a,b+1),*params), gauss(lin(np.arange(a,b+1),*params), *params_gauss_b)*noms(sigma_fit)*np.sqrt(2*np.pi), 'k-', label='Fit')
plt.errorbar(lin(np.arange(a,b+1),*params), noms(Z_3), yerr=sdevs(Z_3), fillstyle= None, fmt=' x', label='Daten')
plt.xlim(657.5, 665)
plt.axhline(y=0.5*data_b[indexes_2[-1]]*noms(sigma_fit)*np.sqrt(2*np.pi), xmin = 0.392, xmax = 0.685, color='g',linestyle='dashed', label='Halbwertsbreite')
plt.axhline(y=0.1*data_b[indexes_2[-1]]*noms(sigma_fit)*np.sqrt(2*np.pi), xmin = 0.278, xmax = 0.8,color='r',linestyle='dashed', label='Zehntelbreite')
plt.ylabel('Zählrate $N$')
plt.xlabel('Energie $E \:/\: \mathrm{keV}$')
plt.legend(loc='best')
plt.grid()
plt.savefig('build/vollpeak.pdf')
plt.clf()
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
# #Plotte das zugeordnete Cs-Spektrum und setze Horizontale bei Zehntel- und Harlbwertsbreite
# x=np.linspace(1,8192,8192)
# plt.plot(lin(x, *params), data_b,'r-',label='Fit')
# plt.plot(lin(indexes_2, *params),data_b[indexes_2],'bx',label='Peaks')
# plt.axhline(y=0.5*data_b[indexes_2[-1]], color='g',linestyle='dashed', label='Halbwertshöhe')
# print('Halbwertshöhe', 0.5*data_b[indexes_2[-1]])
# print('Zehntelwertshöhe', 0.1*data_b[indexes_2[-1]])
# plt.axhline(y=0.1*data_b[indexes_2[-1]], color='r',linestyle='dashed', label='Zehntelhöhe')
# plt.xlim(0,700)
# plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
# plt.ylabel(r'Einträge')
# plt.grid()
# plt.legend(loc='upper left')
# plt.savefig('build/Cs.pdf')
# plt.yscale('log')
# plt.savefig('build/Cs_log.pdf')
# plt.clf()
#
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
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks auf Grund des Photoeffekts liegt bei: {abs_wahrsch_ph*100} Prozent')
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks auf Grund des Comptoneffekts liegt bei: {abs_wahrsch_comp*100} Prozent\n')


#------------------Aufgabenteil d) {Barium oder Antimon? Wir werden es erfahren.}
#Betrachte zuerst Orginalaufnahmen des Detektors
data_d = np.genfromtxt('data/mystery1.txt', unpack=True)
# x_plot = np.linspace(1, 8192, 8192)
# plt.bar(lin(x_plot, *params), data_d)
# plt.xlim(0, 8192)
# plt.xlabel(r'Energie $E \:/\: \mathrm{keV}$')
# plt.ylabel('Zählrate $N$')
# plt.yscale('log')
# plt.savefig('build/Ba_Sb_orginal_log.pdf')
# plt.clf()

#Finde höchste Peaks und ordne sie den passenden Energien des Spektrums zu
# peaks_3 = find_peaks(data_d, height=120, distance=1)
# indexes_3 = peaks_3[0]
# peak_heights_3 = peaks_3[1]
# energie_3 = lin(indexes_3,*params)
# print('Peaks des Barium-Spektrums bei : ',indexes_3, '\n mit Energie', energie_3 , 'keV \n und Höhe: ', data_d[indexes_3])
E_Ba, W_Ba, peaks_Ba_n = np.genfromtxt('data/2_0/Ba_Zuordnung.txt', dtype=float, unpack=True)
E_Ba_n, W_Ba_n, peaks_Ba = np.genfromtxt('data/2_0/Ba_Zuordnung.txt', dtype=int, unpack=True)
E_Ba_ist = lin(peaks_Ba, *params)

make_table(
        header = ['$E_\text{theo}$ / \kilo\electronvolt', '$W$ / \%', 'Kanalnummer $i$', '$E_\text{fit}$ / \kilo\electronvolt'],
        data = [E_Ba, W_Ba, peaks_Ba, E_Ba_ist],
        places = [4.0, 2.1, 4.0, 3.2],
        caption = 'Die Zuordnung zum Spektrum des ${}^{133}$Ba.',
        label = 'tab:zuordnung_Ba',
        filename = 'build/tables/zuordnung_Ba.tex'
        )

plt.plot(lin(x_plot, *params), data_d,'r-',label='Messwerte')
plt.plot(lin(peaks_Ba, *params),data_d[peaks_Ba],'bx',label='Peaks')
plt.xlabel(r'Energie $E \:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.xlim(0, 500)
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('build/mystery1_log.pdf')
plt.clf()

def gaussian_fit_peaks_d(test_ind):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    unter = []
    sigma = []
    for i in test_ind:
        a=i-40
        b=i+40
        params_gauss_d,covariance_gauss_d=curve_fit(gauss,np.arange(a,b+1),data_d[a:b+1],p0=[1,data_d[i],0,i-1])
        errors_gauss_d = np.sqrt(np.diag(covariance_gauss_d))
        sigma_fit=ufloat(params_gauss_d[0],errors_gauss_d[0])
        h_fit=ufloat(params_gauss_d[1],errors_gauss_d[1])
        a_fit=ufloat(params_gauss_d[2],errors_gauss_d[2])
        mu_fit=ufloat(params_gauss_d[3],errors_gauss_d[3])
        #print(h_fit*sigma_fit*np.sqrt(2*np.pi))
        #if i == 3316:
        #    plt.plot(np.arange(a, b+1), data_d[a:b+1], label='Daten')
        #    plt.plot(np.arange(a, b+1), gauss(np.arange(a, b+1), *params_gauss_d), label='Fit')
        #    plt.savefig('build/test.pdf')
        #    plt.clf()
        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit, peak_inhalt, hoehe, unter, sigma

#Führe wieder einen Gauß-Fit in den Bins durch um den Peakinhalt zu bestimmen
index_ba, peakinhalt_ba, hoehe_ba, unter_ba, sigma_ba = gaussian_fit_peaks_d(peaks_Ba.astype('int'))

E_ba_det = []
for i in range(len(index_ba)):
    E_ba_det.append(lin(index_ba[i],*params))


def potenz2(x, a, b, c, e):
    return e * a * (x-b)**(e-1)

#Berechne aktivität der Quelle am Messtag
#print(f'\nDaten zur Berechnung der Akivität: {E_ba_det}, {params2}')
A=peakinhalt_ba[4:]/(3205*omega_4pi*W_Ba[4:]/100*potenz(noms(E_ba_det[4:]),*params2)) #nur die mit E>150keV mitnehmen
# A_Fehler = peakinhalt_ba[4:]/(3205*omega_4pi*W_Ba[4:]/100*potenz2(noms(E_ba_det[4:]),*params2))*sdevs(E_ba_det[4:])
# print(A)
# print(A_Fehler)
# A = unp.uarray(A, A_Fehler)
A_det = []
#for i in range(0,2):
# A_det.append(ufloat(0, 0))
for i in A:
    A_det.append(i)
#print('A_det', A_det)
#print(unter_ba)
#print(peakinhalt_ba)
#Fasse Fit-Parameter in Tabelle zusammen
make_table(
    header= ['Kanalnummer $i$', '$E_\gamma$ / \kilo\electronvolt', '$h_\text{i}$', '$\sigma_\text{i}$ / \kilo\electronvolt'],
    data=[noms(index_ba), E_ba_det, hoehe_ba, sigma_ba],
    places=[3.0, (3.2, 1.2), (4.2, 2.2), (1.2, 1.2)],
    caption='Parameter des Gauß-Fits für das gegeben Spektrum',
    label='tab:Ba',
    filename='build/tables/Ba.tex'
)
#berechnung der neuen Effizienz
Z_d = []
Q_d = []
for i in range(4, len(W_Ba)):
    Z_d.append(np.sqrt(2*np.pi)*hoehe_ba[i]*sigma_ba[i])
    Q_d.append(Z[i]/(omega_4pi*A_det[i-4]*W_Ba[i]/100*3205))
#print('Z_d: ', Z_d)
#print('Q_d: ', noms(Q_d), sdevs(Q_d))
#Trage Ergebnisse der Aktivitätsbestimmung in Tabelle ein
#make_table(
#    header= ['$W$\/\%', '$Z_i$', '$E_i$ / \kilo\electronvolt ', '$A_i$ / \\becquerel '],
#    data=[W_ba,  unter_ba, peakinhalt_ba, A_det],
#    places=[2.1, (2.2, 2.2), (4.2, 3.1), (4.0, 2.2)],
#    caption='Berechnete Aktivitäten für jeden Bin mit dazu benötigten Werten.',
#    label ='plt:aktivitaet_ba',
#    filename ='build/tables/aktivitaet-ba.tex'
#)
make_table(
    header= ['$W\/\%$', 'Q', '$Z_i$', '$E_i$ / \kilo\electronvolt', '$A_i$ / \\becquerel'],
    data=[W_Ba[4:], Q_d ,Z_d, E_ba_det[4:], A_det],
    places=[2.1, (1.3, 1.3), (5.2 , 2.2), (3.2, 1.2), (4.0, 3.0)],
    caption='Berechnete Aktivität der betrachteten Emissionslinien mit dazu korrespondierenden Detektor-Effizienzen.',
    label='tab:aktivitaet_ba',
    filename='build/tables/aktivitaet_ba.tex'
)
A_gem = ufloat(np.mean(noms(A)),np.mean(sdevs(A)))
print('gemittelte Aktivität für Barium',A_gem)


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
#print(f'\nDaten zur Berechnung der Akivität: {E_e}, {params2}, \n den Peakinhalt Z {Z_e},\n die Effizienz Q {Q_e} \n und der Aktivität {A_e}')
print('Aktivitäten des Nuklids', A_e)
print('gemittelte Aktivität für das Salz: ', np.mean(A_e[0:9]))

make_table(
    header= ['$W_{\symup{i}}\/\percent$', 'Q', '$Z_\symup{i}$', '$E_\symup{i}$ / \kilo\electronvolt', '$A_\text{i}$ / \\becquerel'],
    data=[W_e[3:], Q_e ,Z_e, E_e[3:], A_e],
    places=[2.2, (1.3, 1.3), (4.2 , 3.2), 4.1, (3.0, 2.0)],
    caption='Berechnete Aktivität der betrachteten Emissionslinien mit dazu korrespondierenden Detektor-Effizienzen.',
    label='tab:aktivitaet_e',
    filename='build/tables/aktivitaet_e.tex'
)
make_table(
    header= ['$E_i$ / \kilo\electronvolt', '$W_\text{i}$ / \percent', 'Kanalnummer $i$'],
    data=[E_e, W_e, peaks_ind_e],
    places=[4.0, 2.1, 4.0],
    caption='Die ermittelten Peaks zur Nuklid Bestimmung.',
    label='tab:Salz',
    filename='build/tables/Salz_Peaks.tex'
)