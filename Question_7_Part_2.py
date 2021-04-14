"""

Code for Question 7: Part 2 of Project 14.1: Particle or Photon Orbits near a Black Hole

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
import math
import pylab
from lmfit import Model
import os

#defines function for rate of change of the radial co-ordinate with respect to the phi co-ordinate
def func_f(r_i,b,v,phi):
	return(-1*(r**2)*np.sqrt(1/(b**2) + 1/(r**3) - 1/(r**2) + (1-v**2)/(v*v*b*b*r)))

#defines function for term inside sqrt of func_f to test if turning point has been reached (i.e. if func_r_test <= 0)
def func_r_test(r,b,v,phi):
	return(1/(b**2) + 1/(r**3) - 1/(r**2) + (1-v**2)/(v*v*b*b*r))

#function for runge-kutta numerical integration of func_f
def func_k_0(r_i,b,v,phi,h):	
	return (h*func_f(r_i,b,v,phi))

#function for conitinuation of numerical integration after turning point has been reached (i.e. as the photon moves away from the black hole)
def func_k_0_pos(r_i,b,v,phi,h):
	return (-1*h*func_f(r_i,b,v,phi))




pi = np.pi
print(pi)

#sets bounds for phi co-ordinate for numerical integration and the number of the steps of the numerical integration
a=0
b=4*pi
N=1000000
h=(b-a)/N



#sets an initial radial position of the photon
r_init =  100000


#sets a value for the particle speed, the impact parameter, and the initial deflection angle
v = 1
b = 3.2
phi_diff=0.0

#initialises arrays for the r and phi co-ordinates and the test values to determine if the turning point of the orbit has been achieved
r_vals = []
phi_vals = []
turn_pt_vals=[]

#calculates the initial phi co-ordinate based on the geometry wrt the impact parameter and the initial radial position  of the photon
phi_init = np.arcsin(b/r_init)


#calculates test value to determine if the turning point of the orbit has been reached, i.e. if the term in the sqrt in func_f <=0
turn_pt_init = func_r_test(r_init,b,v,phi_init)

#appends the value to the array for the test values
turn_pt_vals.append(turn_pt_init)

#appends the initial r and phi co-ordinates to the appropriate arrays
r_vals.append(r_init)
phi_vals.append(phi_init)

#for loop to calculate the quantities for the runge-kutta method for the instance in which the photon is infalling
for m in range(1,N):

			
	r=r_vals[m-1]
	phi = phi_vals[m-1]
			
	
			
	k_0 = func_k_0(r,b,v,phi,h)

	r1 = r+0.5*k_0

	k_1 = func_k_0(r1,b,v,phi,h)
	

	r2 = r+0.5*k_1
	
	k_2 = func_k_0(r2,b,v,phi,h)
	
	r3 = r+k_2

	k_3 = func_k_0(r3,b,v,phi,h)
	
	#calculates the new values for the r and phi co-ordinate of the runge kutta method
	r_new = r_vals[m-1] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
	phi_new = phi_vals[m-1] + h

	
	#tests if photon has passed the Schwarzschild radius
	if(r_new<=1):
		#calculates the deflection angle in the case of a radial plunge orbit
		phi_diff=phi_new - phi_vals[0] - pi
		break

	#calculates the value for testing whether the turning point of the orbit has been reached
	turn_pt = func_r_test(r_new,b,v,phi_new)
	#appends the value to array for the test values
	turn_pt_vals.append(turn_pt)


	#elseif statement to determine if the turning point of the orbit has been reached
	if(m==1):
		pass
	elif(turn_pt_vals[m]<=0):
		#for loop that calculates the r and phi co-ordinates of the photon as it moves away from the black hole for the remaining steps of te numerical integration
		for i in range(m-1,N):

			#sets values for r and phi co-ordinates
			r=r_vals[i]
			phi = phi_vals[i]
			r_i = r_vals[m-1]
			
			
			k_0 = func_k_0_pos(r,b,v,phi,h)

			r1 = r+0.5*k_0

			k_1 = func_k_0_pos(r1,b,v,phi,h)
	

			r2 = r+0.5*k_1
	
			k_2 = func_k_0_pos(r2,b,v,phi,h)
	
			r3 = r+k_2

			k_3 = func_k_0_pos(r3,b,v,phi,h)
	
			#values for r and phi for next step of the numerical integration
			r_new = r_vals[i] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
			phi_new = phi_vals[i] + h

			#appends values to appropriate array
			r_vals.append(r_new)
			phi_vals.append(phi_new)

			#elseif statement that breaks the loop if the photon crosses the schwarzschild radius or escapes to its original position
			if(r_new<=1):
				break
			elif(r_new>=r_init):
				break

		
		#breaks loop if the photon does not escape to original position or plunge to centre of attraction but the numerical integration has been completed for all steps
		break


	#appends values to the appropriate arrays 
	r_vals.append(r_new)
	phi_vals.append(phi_new)



#calculates angle of deflection from RK4 method
phi_diff = phi_vals[-1] - phi_init - pi


#creates polar plot
ax = plt.subplot(111,projection='polar')


#plots circle at centre of polar plot to represent the black hole
circle= plt.Circle((0,0), radius= 1,color='black',transform=ax.transData._b)
ax.add_artist(circle)
#plots the motion of the particle in (r,phi) space using the values determined from the runge kutta method
ax.plot(phi_vals, r_vals, label='Particle Path (Initial R = ' +str(r_init)+', b = '+ str(b) +', Angle Deflected = '+str(phi_diff) + ' Rad)')
#adds legend to the plot
ax.legend(loc='lower center', fontsize='large')
#sets aspect ratio of the plot
ax.set_aspect('equal')
ax.tick_params(labelsize=15)
#shows the plot
plt.show()


#creates polar plot
ax = plt.subplot(111,projection='polar')


#plots circle at centre of polar plot to represent the black hole
circle= plt.Circle((0,0), radius= 1,color='black',transform=ax.transData._b)
ax.add_artist(circle)
#plots the motion of the particle in (r,phi) space using the values determined from the runge kutta method
ax.plot(phi_vals, r_vals, label='Particle Path (Initial R = ' +str(r_init)+', b = '+ str(b) +', Angle Deflected = '+str(phi_diff) + ' Rad)')
#adds legend to the plot
ax.legend(loc='lower center', fontsize='large')
#sets aspect ratio of the plot
ax.set_aspect('equal')
ax.tick_params(labelsize=15)
ax.set_rmax(10.0)
#shows the plot
plt.show()


