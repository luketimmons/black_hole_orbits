"""

Code for Question 5 of Project 14.1: Particle or Photon Orbits near a Black Hole

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

#function for the rate of change of radial positiion wrt co-ordinate time
def func_f(z):
	return(z)

#function for snd derivative of r wrt t 
def func_g(tau,r,z,l,q):
	return(2*(z**2)/r + ((1-1/r)/(2*r))*((6/r - 4) + (r**4/(l*l*(1/r -1)))*q*q*(9*l*l/(r*r*r) - 6*l*l/(r*r) + 7/r - 4)))


#function for derivative of phi co-ordinate wrt co-ordinate time
def func_a(tau,r,z,l):
	return ((l/(r*r))*np.sqrt(((1-1/r)**2 - z**2)/((1-1/r)*(1+l*l/(r*r)))))

#function for derivative of proper time wrt co-ordinate time
def func_b(tau,r,z,l,q):
	return(q*r*r/l)




#defines functions for numerical integration via runge kutta method
def func_k_0(tau_i,r_i,z_i,h):
	return (h*func_f(z_i))


def func_l_0(tau_i,r_i,z_i,h,l,q):
	return (h*func_g(tau_i,r_i,z_i,l,q))

def func_m_0(tau_i,r_i,z_i,h,l):
	return (h*func_a(tau_i,r_i,z_i,l))

def func_n_0(tau_i,r_i,z_i,h,l,q):
	return(h*func_b(tau_i,r_i,z_i,l,q))

#defines function to determine angular momentum required for a given angular momentum
def func_circ_orbit(r_i):
	return (r_i**2/(2*(r_i-1.5)))



pi = np.pi
print(pi)

#sets the bounds for the co-ordinate time over which the numerical integration will take place as well as the number of steps for the  
a=0
b=1000
N=1000
h=(b-a)/N


#initialises arrays for numerical integration
r_vals=[]
z_vals=[]
t_vals=[]
plunge_t=[]
phi_vals=[]
tau_vals=[]

#sets the value for the angular momentum
l=2.0


#sets the initial values for the quantities
r_init = 6.0
z_init = 0.0
t_init =0.0
phi_init=0.0
tau_init=0.0

#appends the initial values to the appropriate arrays
r_vals.append(r_init)
z_vals.append(z_init)
t_vals.append(t_init)
phi_vals.append(phi_init)
tau_vals.append(tau_init)

#initialises array for angular momentum for which a radial plunge orbit will occur
rad_plunge_l=[]

#initialises array for proper time taken for a radial plunge to occur
plunge_tau=[]

#creates an array for angular momentum values
l_vals = np.arange(1.00001,5.00001,0.00001)


#for loop within which the numerical integration is carried out
for i in range(1,N):
	#sets the values for t,r, dr/dt, phi,dphi/dt, and tau
	t=t_vals[i-1]
	r=r_vals[i-1]
	z=z_vals[i-1]
	phi = phi_vals[i-1]
	#calculates the derivative of phi wrt t
	q=func_a(phi,r,z,l)
	tau=tau_vals[i-1]
	
	k_0 = func_k_0(phi,r,z,h)
	l_0 = func_l_0(phi,r,z,h,l,q)
	m_0 = func_m_0(phi,r,z,h,l)
	n_0 = func_n_0(phi,r,z,h,l,q)

	r1 = r+0.5*k_0
	z1 = z+0.5*l_0
	phi1 = phi+0.5*m_0
	q1 = func_a(phi1,r1,z1,l)
	tau1 = tau + 0.5*n_0


	k_1 = func_k_0(phi1,r1,z1,h)
	l_1 = func_l_0(phi1,r1,z1,h,l,q1)
	m_1 = func_m_0(phi1,r1,z1,h,l)
	n_1 = func_n_0(phi1,r1,z1,h,l,q1)

	r2 = r+0.5*k_1
	z2 = z+0.5*l_1
	phi2 = phi+0.5*m_1
	q2 = func_a(phi2,r2,z2,l)
	tau2 = tau + 0.5*n_1

	k_2 = func_k_0(phi2,r2,z2,h)
	l_2 = func_l_0(phi2,r2,z2,h,l,q2)
	m_2 = func_m_0(phi2,r2,z2,h,l)
	n_2 = func_n_0(phi2,r2,z2,h,l,q2)


	r3 = r+k_2
	z3 = z+l_2
	phi3 = phi+m_2
	q3 = func_a(phi3,r3,z3,l)
	tau3 = tau + 0.5*n_2

	k_3 = func_k_0(phi3,r3,z3,h)
	l_3 = func_l_0(phi3,r3,z3,h,l,q3)
	m_3 = func_m_0(phi3,r3,z3,h,l)
	n_3 = func_n_0(phi3,r3,z3,h,l,q3)

	#calculates the quantites for the next step of the numerical integration
	r_new = r_vals[i-1] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
	z_new = z_vals[i-1] + (1/6)*(l_0+2*l_1+2*l_2+l_3)
	phi_new = phi_vals[i-1] + (1/6)*(m_0+2*m_1+2*m_2+m_3)
	t_new = t_vals[i-1] + h
	tau_new = tau_vals[i-1] + (1/6)*(n_0+2*n_1+2*n_2+n_3)


	#appends the values to the appropriate arrays
	r_test = r_new
	r_vals.append(r_new)
	z_vals.append(z_new)
	t_vals.append(t_new)
	phi_vals.append(phi_new)
	tau_vals.append(tau_new)
	
	#checks if a NaN value has been determined for the radial position of the particle
	nan_test = math.isnan(r_new)
	nan_tau_test = math.isnan(tau_new)

	#if statement to test if the radial positiion of the particle has passed the Schwarzschild radius, an NaN value was encountered
	if r_test <=1:
		plunge_time = tau_vals[i-1]
		break
	elif nan_test == True:
		plunge_time = tau_vals[i-1]
		break
	elif nan_tau_test == True:
		plunge_time = tau_vals[i-1]
		break
	else:
		plunge_time = 0.0

	#if statement to test if the particle effectively escaped to infinity
	if r_test>=1e6:
		break


#creates a polar plot
ax = plt.subplot(111,projection='polar')


#adds circle to centre of plot to represent the black hole
circle= plt.Circle((0,0), radius= 1,color='black',transform=ax.transData._b)
ax.add_artist(circle)

#plots the particle motion for two cases: for a radial plunge or otherwise
if (plunge_time==0.0):
	ax.plot(phi_vals, r_vals, label='Particle Path (Initial R = ' +str(r_init)+' , l = ' + str(l) + ')')
else:
	ax.plot(phi_vals, r_vals, label='Particle Path (Initial R = ' +str(r_init)+' , l = ' + str(l) + ', Plunge Proper Time = '+str(plunge_time)+')')

#adds legend to the plot
ax.legend(loc='lower center', fontsize='large')
ax.set_aspect('equal')
ax.tick_params(labelsize=15)

#shows the plot
plt.show()



#for loop to iterate through each value for the angular momentum
for j in range(len(l_vals)):
	#initialises arrays for numerical integration
	r_vals=[]
	z_vals=[]
	t_vals=[]
	phi_vals=[]
	tau_vals=[]

	#appends the initial values to the appropriate arrays
	r_vals.append(r_init)
	z_vals.append(z_init)
	t_vals.append(t_init)
	phi_vals.append(phi_init)
	tau_vals.append(tau_init)


	#sets the value for the angular momentum
	l = l_vals[j]

	#boolean to test if a radial plunge has occured
	plunge=False

	#for loop within which the numerical integration is carried out
	for i in range(1,N):
		#sets the values for t,r, dr/dt, phi,dphi/dt, and tau
		t=t_vals[i-1]
		r=r_vals[i-1]
		z=z_vals[i-1]
		phi = phi_vals[i-1]
		#calculates the derivative of phi wrt t
		q=func_a(phi,r,z,l)
		tau=tau_vals[i-1]
	
		k_0 = func_k_0(phi,r,z,h)
		l_0 = func_l_0(phi,r,z,h,l,q)
		m_0 = func_m_0(phi,r,z,h,l)
		n_0 = func_n_0(phi,r,z,h,l,q)

		r1 = r+0.5*k_0
		z1 = z+0.5*l_0
		phi1 = phi+0.5*m_0
		q1 = func_a(phi1,r1,z1,l)
		tau1 = tau + 0.5*n_0


		k_1 = func_k_0(phi1,r1,z1,h)
		l_1 = func_l_0(phi1,r1,z1,h,l,q1)
		m_1 = func_m_0(phi1,r1,z1,h,l)
		n_1 = func_n_0(phi1,r1,z1,h,l,q1)

		r2 = r+0.5*k_1
		z2 = z+0.5*l_1
		phi2 = phi+0.5*m_1
		q2 = func_a(phi2,r2,z2,l)
		tau2 = tau + 0.5*n_1

		k_2 = func_k_0(phi2,r2,z2,h)
		l_2 = func_l_0(phi2,r2,z2,h,l,q2)
		m_2 = func_m_0(phi2,r2,z2,h,l)
		n_2 = func_n_0(phi2,r2,z2,h,l,q2)


		r3 = r+k_2
		z3 = z+l_2
		phi3 = phi+m_2
		q3 = func_a(phi3,r3,z3,l)
		tau3 = tau + 0.5*n_2

		k_3 = func_k_0(phi3,r3,z3,h)
		l_3 = func_l_0(phi3,r3,z3,h,l,q3)
		m_3 = func_m_0(phi3,r3,z3,h,l)
		n_3 = func_n_0(phi3,r3,z3,h,l,q3)

		#calculates the quantites for the next step of the numerical integration
		r_new = r_vals[i-1] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
		z_new = z_vals[i-1] + (1/6)*(l_0+2*l_1+2*l_2+l_3)
		phi_new = phi_vals[i-1] + (1/6)*(m_0+2*m_1+2*m_2+m_3)
		t_new = t_vals[i-1] + h
		tau_new = tau_vals[i-1] + (1/6)*(n_0+2*n_1+2*n_2+n_3)


		#appends the values to the appropriate arrays
		r_test = r_new
		r_vals.append(r_new)
		z_vals.append(z_new)
		t_vals.append(t_new)
		phi_vals.append(phi_new)
		tau_vals.append(tau_new)
	
		#checks if a NaN value has been determined for the radial position of the particle or the proper time
		nan_test = math.isnan(r_new)
		nan_tau_test = math.isnan(tau_new)

		#if statement to test if the radial positiion of the particle has passed the Schwarzschild radius, an NaN value was encountered for the radial position or proper time. if so the angular momentum value, and proper time are appended to arrays and loop is broken
		if r_test <=1:
			rad_plunge_l.append(l)
			plunge_time = tau_vals[i-1]
			plunge_tau.append(plunge_time)
			plunge=True
			break
		elif nan_test == True:
			rad_plunge_l.append(l)
			plunge_time = tau_vals[i-1]
			plunge_tau.append(plunge_time)
			plunge=True
			break
		elif nan_tau_test==True:
			rad_plunge_l.append(l)
			plunge_time = tau_vals[i-1]
			plunge_tau.append(plunge_time)
			plunge=True
			break

	#if statement to test if the particle effectively escaped to infinity
	if r_test>=1e6:
		break
	elif plunge==False:
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
	print('Only photons can achieve a circular orbit for an initial radius of R=1.5')

#creates a plot
ax = plt.subplot(111)

#plots the proper time for a radial plunge to occur as a function of angular momentum
ax.plot(rad_plunge_l,plunge_tau, label='Proper Time for Radial Plunge (R =' + str(r_init) + ')')




ax.set_xlabel('Angular Momentum $l$', fontsize=16)
ax.set_ylabel(r'Proper Time $\tau$', fontsize=16)
ax.tick_params(labelsize=14)

#adds a grid to the plot
ax.grid(True,alpha=0.5)

#adds a legend to the plot
ax.legend(loc='upper right', fontsize='large')
#shows the plot
plt.show()


#saves the radial plunge angular momenta and proper time for plunge to occur to csv file 
dict = {'Radial Plunge l': rad_plunge_l, 'Proper Time': plunge_tau}

df=pd.DataFrame(dict)

df.to_csv('radial_plunge_values_r='+str(r_init)+'_second_approach.csv')



