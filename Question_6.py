"""

Code for Question 6 of Project 14.1: Particle or Photon Orbits near a Black Hole

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


#defines a power law function
def powfit(v,a,b):
	return(a*v**b)

#defines a hyperbolic function
def hypbolfit(x,a,b):
	return(a + b/x)

#sets a power law and hyperbolic model fit
powmodel = Model(powfit)
hypmodel = Model(hypbolfit)

#defines function for rate of change of the radial co-ordinate with respect to the phi co-ordinate
def func_f(r_i,b,v,phi):
	return(-1*(r**2)*np.sqrt(1/(b**2) + 1/(r**3) - 1/(r**2) + (1-v**2)/(v*v*b*b*r)))

#defines function for term inside sqrt of func_f to test if turning point has been reached (i.e. if func_r_test <= 0)
def func_r_test(r,b,v,phi):
	return(1/(b**2) + 1/(r**3) - 1/(r**2) + (1-v**2)/(v*v*b*b*r))

#function for runge-kutta numerical integration of func_f
def func_k_0(r_i,b,v,phi,h):
	#print (func_f(r_i,b,v,phi))
	return (h*func_f(r_i,b,v,phi))

#function for conitinuation of numerical integration after turning point has been reached (i.e. as the photon moves away from the black hole)
def func_k_0_pos(r_i,b,v,phi,h):
	return (-1*h*func_f(r_i,b,v,phi))




pi = np.pi
print(pi)

#defines the bounds of the phi co-ordinate over which the numerical integration will take place as well as the number of steps of the numerical integration
a=0
b=4*pi
N=10000
h=(b-a)/N


#initialies arrays for the scattering cross-section and the particle speeds at which the critical impact parameter (accounts for instance in which critical impact parameter exceeds bounds of test value)
sigma_vals = []
v_crit_vals =[]



#creates arrays for particle speeds and impact parameters to be tested
v_vals= np.arange(0.03, 1.01, 0.01)
b_vals=np.arange(0.01,60.01,0.01)

rad_plunge_l=[]

#initial radial position of the particle
r_init =  1000

#initialises the array for the critical impact parameter 
b_crit_val=[]


#for loop over which the particle speeds are iterated through
for i in range(len(v_vals)):
	#sets the value of the particle speed
	v=v_vals[i]

	#for loop to iterate through the impact parameters
	for j in range(len(b_vals)):
		#initialises arrays for r and phi co-ordinates and test values to determine if turning point has been reached
		r_vals = []
		phi_vals = []
		turn_pt_vals=[]

		#sets the impact parameter
		b = b_vals[j]
		#calculates the initial phi co-ordinate based on the geometry wrt the impact parameter and initial radius
		phi_init = np.arcsin(b/r_init)

		#calculates the test value to determine if the turning point, i.e. the term in the sqrt in func_f is less than zero
		turn_pt_init = func_r_test(r_init,b,v,phi_init)

		#appends value to the array for test values
		turn_pt_vals.append(turn_pt_init)

		#appends the initial r and phi co-ordinates to the appropriate arrays
		r_vals.append(r_init)
		phi_vals.append(phi_init)

		#sets boolean to say if a radial plunge to occur
		plunge=False

	
		#for loop to calculate the quantities for the runge-kutta method for the instance in which the photon is infalling
		for m in range(1,N):

			
			r=r_vals[m-1]
			phi = phi_vals[m-1]
			#print(r)
	
			
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
			
			#elseif statement that breaks the loop if the photon crosses the schwarzschild radius or escapes to its original position
			if(r_new<=1):
				#sets boolean to say that a radial plunge has occured
				plunge=True
				#breaks the loop
				break
			elif(r_new>=r_init):
				plunge=False
				#breaks the loop
				break

			#calculates the value of term in sqrt in func_f
			turn_pt = func_r_test(r_new,b,v,phi_new)
			#appends value to array
			turn_pt_vals.append(turn_pt)

			#elseif statement to determine if the turning point of the orbit has been reached
			if(m==1):
				pass
			elif(turn_pt_vals[m]<=0):
				#for loop that calculates the r and phi co-ordinates of the photon as it moves away from the black hole for the remaining steps of te numerical integration
				for q in range(m-1,N):
					
					#sets values for r and phi co-ordinates
					r=r_vals[q]
					phi = phi_vals[q]
				
					k_0 = func_k_0_pos(r,b,v,phi,h)

					r1 = r+0.5*k_0

					k_1 = func_k_0_pos(r1,b,v,phi,h)
	

					r2 = r+0.5*k_1
	
					k_2 = func_k_0_pos(r2,b,v,phi,h)
	
					r3 = r+k_2

					k_3 = func_k_0_pos(r3,b,v,phi,h)
	
					#values for r and phi for next step of the numerical integration
					r_new = r_vals[q] + (1/6)*(k_0+2*k_1+2*k_2+k_3)
					phi_new = phi_vals[q] + h


					#appends values to appropriate array
					r_vals.append(r_new)
					phi_vals.append(phi_new)

					
					#elseif statement that breaks the loop if the photon crosses the schwarzschild radius or escapes to its original position
					if(r_new<=1):
						plunge=True
						break
					elif(r_new>=r_init):
						plunge=False
						break

		
				#breaks loop if the photon does not escape to original position or plunge to centre of attraction but the numerical integration has been completed for all steps
				break

			#appends values to the appropriate arrays 
			r_vals.append(r_new)
			phi_vals.append(phi_new)

		#if statement for case of radial plunge not occuring
		if (plunge==False):

			print(b)
			print(v)
			#appends critical impact parameter and corresponding particle speeds to appropriate arrays
			b_crit_val.append(b)
			v_crit_vals.append(v)
			#calculates scattering cross section and appends to array
			sigma = pi*b*b
			sigma_vals.append(sigma)
			#breaks the loop
			break




#saves the critical impact parameters, corresponding particle speeds, and scattering cross-section to csv file
dict = {'Speed v': v_crit_vals, 'Critical Impact Parameter': b_crit_val, 'Cross-section':sigma_vals}

df=pd.DataFrame(dict)

df.to_csv('cross_section_r='+str(r_init)+'_new.csv')


#creates a plot
ax = plt.subplot(111)

result = powmodel.fit(sigma_vals,v=v_crit_vals,a=13.2,b=-1.98)

#plots the scattering cross-section as a function of critical impact parameter 
ax.plot(v_crit_vals,sigma_vals, label='Scattering cross-section as a function of particle speed')
ax.plot(v_crit_vals,result.best_fit, label=r'Power Law Model: $\sigma \propto v^{\alpha}$, $\alpha = -1.981923$')


ax.set_xlabel('Particle Speed $v$', fontsize=14)
ax.set_ylabel('Cross-section $\sigma$', fontsize=14)
ax.tick_params(labelsize=12)

print(result.params)

#adds a grid to the plot
ax.grid(True,alpha=0.5)
#adds the legend to the plot
ax.legend(loc='upper right', fontsize='large')
#shows the plot
plt.show()

