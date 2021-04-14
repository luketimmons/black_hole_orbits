"""

Code for Question 4 of Project 14.1: Particle or Photon Orbits near a Black Hole

Name: Luke Timmons
Student Number: 304757457

"""

#imports libraries to be used
import PIL
from PIL import Image
import pandas as pd
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import pylab
from lmfit import Model
import os

#defines function for rate of change of 'r' as function of 'phi'
def func_f(z):
	return(z)

#defines function for rate of change of 'dr/dphi' as a function of 'phi'	
def func_g(tau,r,z,l):
	return((2*(z**2)/r) + r - (r**2)/(2*l**2) -1.5)

#defines function for rate of change of 'tau' as a function of 'phi'
def func_a(tau,r,z,l):
	return (r**2/l)


def func_k_0(tau_i,r_i,z_i,h):
	return (h*func_f(z_i))


def func_l_0(tau_i,r_i,z_i,h,l):
	return (h*func_g(tau_i,r_i,z_i,l))

def func_m_0(tau_i,r_i,z_i,h,l):
	return (h*func_a(tau_i,r_i,z_i,l))


#defines function to determine the angular momentum for a circular orbi
def func_circ_orbit(r_i):
	return (r_i**2/(2*(r_i-1.5)))



pi = np.pi
print(pi)

#define initial and final values for phi co-ordinate and the number of steps for the numerical integration
a=0
b=8*pi
N=1000
h=(b-a)/N


#intialises arrays for numerical integation
r_vals=[]
z_vals=[]
tau_vals=[]
phi_vals=[]




#initial values for co-ordinates for numerical integration
z_init = 0.0
tau_init =0.0
phi_init=0.0

#appends initial values to arrays for numerical integration
z_vals.append(z_init)
tau_vals.append(tau_init)
phi_vals.append(phi_init)


#creates an arrays for values of angular momentum
l_vals= np.arange(1.0001, 5.0001, 0.0001)

#creates an arrays for values of the initial radius of the orbit
r_init_vals = np.arange(1.6,6.1,0.1)

#initialises arrays for radial plunge angular momenta, perturbation of angular momentum, critical angular momenta and corresponding radii, circular orbit angular momenta and correspondind radii
var_l = []
l_crit = []
r_crit_vals = []
circ_l_vals=[]
circ_r_vals=[]


#for loop to iterate through initial radii of the particle orbits
for m in range(len(r_init_vals)):
	#sets
	r_init = r_init_vals[m]
	print(m)
	#initialises arrays for radial plunge angular momenta and proper time taken for the radial plunge to occur
	rad_plunge_l=[]
	plunge_t=[]
	print(r_init)

	#for loop to iterate through each value for the angular momentum
	for j in range(len(l_vals)):
		#initialies array for the r,z,tau, and phi values for the numerical integration
		r_vals=[]
		z_vals=[]
		tau_vals=[]
		phi_vals=[]

		#appends the initial values for the quantites to the appropriate arrays
		r_vals.append(r_init)
		z_vals.append(z_init)
		tau_vals.append(tau_init)
		phi_vals.append(phi_init)

		
		#sets the value for the angular momentum
		l = l_vals[j]
		
		#for loop to carry out the runge-kutta numerical integation
		for i in range(1,N):
			#sets the values for the proper time, r co-ordinate, phi co-ordinate, and the rate of change of r wrt phi
			tau=tau_vals[i-1]
			r=r_vals[i-1]
			z=z_vals[i-1]
			phi=phi_vals[i-1]
			plunge=False

			k_0 = func_k_0(tau,r,z,h)
			l_0 = func_l_0(tau,r,z,h,l)
			m_0 = func_m_0(tau,r,z,h,l)


			r1 = r+0.5*k_0
			z1 = z+0.5*l_0
			tau1 = tau+0.5*m_0

			k_1 = func_k_0(tau1,r1,z1,h)
			l_1 = func_l_0(tau1,r1,z1,h,l)
			m_1 = func_m_0(tau1,r1,z1,h,l)
			

			r2 = r+0.5*k_1
			z2 = z+0.5*l_1
			tau2 = tau+0.5*m_1

			k_2 = func_k_0(tau2,r2,z2,h)
			l_2 = func_l_0(tau2,r2,z2,h,l)
			m_2 = func_m_0(tau2,r2,z2,h,l)

			r3 = r+k_2
			z3 = z+l_2
			tau3 = tau+m_2

			k_3 = func_k_0(tau3,r3,z3,h)
			l_3 = func_l_0(tau3,r3,z3,h,l)
			m_3 = func_m_0(tau3,r3,z3,h,l)

			#calculates values for the quantities for next step of the runge kutta integration
			r_new = r_vals[i-1] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
			z_new = z_vals[i-1] + (1/6)*(l_0+2*l_1+2*l_2+l_3)
			tau_new = tau_vals[i-1] + (1/6)*(m_0+2*m_1+2*m_2+m_3)
			phi_new = phi_vals[i-1] + h

			#appends the values to the appropriate arrays to be used in next iteration of the loop
			r_vals.append(r_new)
			z_vals.append(z_new)
			tau_vals.append(tau_new)
			phi_vals.append(phi_new)

			r_test = r_new

			#else if statement to test whether the particle passes the Schwarzschild radius or effectively escapes to infinity
			if ((r_test <=1)):
				#appends angular momentum value for radial plunge to appropriate array
				rad_plunge_l.append(l)
				#appends proper time for radial plunge to occur to appropriate array
				plunge_t.append(tau_vals[i])
				#sets boolean to state that a radial plunge has occurred
				plunge=True
				#breaks the loop
				break	
			elif((r_test>= 1e6)):
				#breaks the loop
				break
		#if an orbit occurs for which a radial plunge does not occur, the loop is broken
		if(plunge==False):
			break

	#prints the bounds of the angular momentum values for which a radial plunge orbit will occur
	print('l values for radial plunge orbit are between 0 and ' + str(rad_plunge_l[-1]))
	#sets value for critical angular momentum
	final_l = rad_plunge_l[-1]
	#appends value to array for critical angular momenta
	l_crit.append(final_l)
	#appends initial radius value to array for instance in which the critical angular momentum is not within bounds set previously such that arrays are of two different sizes
	r_crit_vals.append(r_init)
	#calculates the squared function to detemine the angular momentum for a circular orbit
	circ_l_sq = (func_circ_orbit(r_init))
	#tests to see if the value is negative such that the square root of the value will produce an imaginary number. if so, prints that circular orbit not possible for given radius
	if (circ_l_sq <0):
		print('A particle cannot achieve a circular orbit at this radial position.')
	elif(circ_l_sq>0):
		#calculates value of circular orbit angular momentum
		circ_l = np.sqrt(circ_l_sq)
		#appends value to array for circular orbit angular momenta
		circ_l_vals.append(circ_l)
		#calculates value of critical perturbation, i.e. perturbation required to disturb from circular orbit to radial plunge
		l_diff = circ_l - rad_plunge_l[-1]
		#appends value to array for critical perturbation
		var_l.append(l_diff)
		#appends value of initial radius to array for consideration with the circular orbits in case a circular orbit is not possible for a given radius and arrays are of different size
		circ_r_vals.append(r_init)
		#prints value of angular momentum for circular orbit to occur
		print('For a circular orbit and l value of l = ' + str(circ_l) + ' is required.')
	elif(r_init==1.5):
		#prints that for an initial radius of 1.5, that the particle is at the radius of the photon sphere where only photons can achieve circular orbits
		print('Only a photon can achieve a circular orbit at this radial position. This radius is that of the photon sphere')

	
	#saves the radial plunge angular momenta and proper time for plunge to occur to csv file 
	dict = {'Radial Plunge l': rad_plunge_l, 'Proper Time': plunge_t}

	df=pd.DataFrame(dict)

	df.to_csv('radial_plunge_values_r='+str(r_init)+'_new.csv')

	#prints iteration of loop for inital radius values to act as milestone marker
	print(m)

#saves circular orbit angular momenta, corresponding initial radii, and critical perturbation to cause radial plunge from circular orbit
dict = {'Circular Orbit l': circ_l_vals, 'Radius': circ_r_vals, 'Variation in l': var_l}
df2 = pd.DataFrame(dict)
df2.to_csv('circular_l_values_values_new.csv')

#creates plot
ax = plt.subplot(111)

#plots the critical angular momentum as a function of initial radius
ax.plot(r_crit_vals,l_crit, label='Critical Angular Momentum as a Function of Initial Radius')
ax.plot(circ_r_vals,circ_l_vals, label='Angular Momentum for Circular Orbit')


ax.set_xlabel('Initial Radius $R$', fontsize=14)
ax.set_ylabel('Critical Angular Momentum $l$', fontsize=14)
ax.tick_params(labelsize=12)

ax.grid(True,alpha=0.5)
#adds legend and shows the plot
ax.legend(loc='upper right', fontsize='large')
plt.show()

#saves the critical angular momenta and corresponding initial radii to a csv file
dict = {'Critical Angular momentum': l_crit, 'Radius': r_crit_vals}
df3 = pd.DataFrame(dict)
df3.to_csv('critical_l_values_new.csv')

#creates plot
ax = plt.subplot(111)

#plots critical angular momentum perturbation as a function of initial radius of the orbit
ax.plot(r_init_vals,var_l, label='Perturbation to cause radial plunge')


ax.set_xlabel('Initial Radius $R$', fontsize=14)
ax.set_ylabel('Angular Momentum Perturbation $\Delta l$', fontsize=14)
ax.tick_params(labelsize=12)
ax.grid(True,alpha=0.5)
#adds legend and shows plot
ax.legend(loc='upper right', fontsize='large')
plt.show()

