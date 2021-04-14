"""

Code for Questions 2 and 3 of Project 14.1: Particle or Photon Orbits near a Black Hole

Name: Luke Timmons
Student Number: 304757457

"""


#import libraries to be used
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
plunge_t=[]
phi_vals=[]


#initial values for co-ordinates for numerical integration
r_init = 6.0
z_init = 0.0
tau_init =0.0
phi_init=0.0

#appends initial values to arrays for numerical integration
r_vals.append(r_init)
z_vals.append(z_init)
tau_vals.append(tau_init)
phi_vals.append(phi_init)


#creates an arrays for values of angular momentum
l_vals= np.arange(1.00001, 5.00001, 0.00001)


#initialises array for radial plunge angular momentum values
rad_plunge_l=[]

#boolean used for test to determine if a radial plunge orbit had occured
plunge = False

#sets value for angular momentum
l=2.0

#for loop in which the runge-kutta method is implemented
for i in range(N-1):
	tau=tau_vals[i]
	r=r_vals[i]
	z=z_vals[i]
	phi = phi_vals[i]

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


	#calculates the next values in the runge-kutta method
	r_new = r_vals[i] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
	z_new = z_vals[i] + (1/6)*(l_0+2*l_1+2*l_2+l_3)
	tau_new = tau_vals[i] + (1/6)*(m_0+2*m_1+2*m_2+m_3)
	phi_new = phi_vals[i] + h

	#appends next values in runge-kutta method into arrays
	r_test = r_new
	r_vals.append(r_new)
	z_vals.append(z_new)
	tau_vals.append(tau_new)
	phi_vals.append(phi_new)

	#if statement to test if particle has plunged to schwarzschild radius
	if r_test <=1:
		print(tau_new)
		plunge_time = tau_new
		plunge=True
		#breaks out of loop ending runge-kutta integration
		break
	else:
		plunge = False
	#if statement to test if particle has effectively escaped to infinity
	if r_test>=1e6:
		break


#creates polar plot
ax = plt.subplot(111,projection='polar')


#adds circle to polar plot to represent black hole
circle= plt.Circle((0,0), radius= 1,color='black',transform=ax.transData._b)
ax.add_artist(circle)

#if statement that will add the proper time taken for a radial plunge orbit to occur to the legend of the plot
if (plunge==False):
	ax.plot(phi_vals, r_vals, label='Particle Path (Initial R = ' +str(r_init)+' , l = ' + str(l) + ')')
else:
	ax.plot(phi_vals, r_vals, label='Particle Path (Initial R = ' +str(r_init)+' , l = ' + str(l) + ', Plunge Proper Time = '+str(plunge_time)+')')

#adds legend to the plot
ax.legend(loc='lower center', fontsize='large')
#sets aspect of the plot
ax.set_aspect('equal')
ax.tick_params(labelsize=15)
#shows the polar plot
plt.show()


#for loop to iterate through each value for the angular momentum
for j in range(len(l_vals)):
	#initialies array for the r,z,tau, and phi values for the numerical integration
	r_vals=[]
	z_vals=[]
	tau_vals=[]
	phi_vals=[]

	plunge=False

	#appends the initial values for the quantites to the appropriate arrays
	r_vals.append(r_init)
	z_vals.append(z_init)
	tau_vals.append(tau_init)
	phi_vals.append(phi_init)#sets the value for the angular momentum

	#sets the value for the angular momentum
	l = l_vals[j]

	#for loop to carry out the runge-kutta numerical integation
	for i in range(1,N):
		#sets the values for the proper time, r co-ordinate, phi co-ordinate, and the rate of change of r wrt phi
		tau=tau_vals[i-1]
		r=r_vals[i-1]
		z=z_vals[i-1]
		phi=phi_vals[i-1]


	

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

		#else if statement to test whether the particle passes the Schwarzschild radius
		if ((r_test <=1)):
			#appends angular momentum value for radial plunge to appropriate array
			rad_plunge_l.append(l)
			#appends proper time for radial plunge to occur to appropriate array
			plunge_t.append(tau_vals[i])
			#breaks the loop
			plunge = True
			break	

	if plunge==False:
		break

#prints the bounds of the angular momentum values for which a radial plunge orbit will occur
print('l values for radial plunge orbit are between 0 and ' + str(rad_plunge_l[-1]))
#calculates the squared function to detemine the angular momentum for a circular orbit
circ_l_sq = (func_circ_orbit(r_init))
#tests to see if the value is negative such that the square root of the value will produce an imaginary number. if so, prints that circular orbit not possible for given radius
if (circ_l_sq <0):
	print('A particle cannot achieve a circular orbit at this radial position.')
elif (circ_l_sq>0):
	#calculates value of circular orbit angular momentum
	circ_l = np.sqrt(circ_l_sq)
	#prints value of angular momentum for circular orbit to occur
	print('For a circular orbit and l value of l = ' + str(circ_l) + ' is required.')
elif (r_init==1.5):
	print('Only photons can achieve a circular orbit for an initial radius of R = 1.5')

#creates a plot
ax = plt.subplot(111)

#plots the proper time taken for a radial plunge to occur as a function of the angular momentum
ax.plot(rad_plunge_l,plunge_t, label='Proper Time for Radial Plunge (R =' + str(r_init) + ')')




ax.set_xlabel(r'Angular Momentum $l$', fontsize=16)
ax.set_ylabel(r'Proper Time $\tau$', fontsize=16)
ax.tick_params(labelsize=14)

#adds a grid to the plot
ax.grid(True,alpha=0.5)
#adds a legend to the plot
ax.legend(loc='upper right', fontsize='large')
#shows the plot
plt.show()




#saves the radial plunge angular momenta and proper time for plunge to occur to csv file 
dict = {'Radial Plunge l': rad_plunge_l, 'Proper Time': plunge_t}

df=pd.DataFrame(dict)

df.to_csv('radial_plunge_values_r='+str(r_init)+'.csv')

