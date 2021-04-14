"""

Code for Question 7: Part 1 of Project 14.1: Particle or Photon Orbits near a Black Hole

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

def powfit(v,a,b):
	return(a*v**b)

def hypbolfit(x,a,b):
	return(a + b/x)


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
N=1000000
h=(b-a)/N


#creates array for values for impact parameter
b_vals=np.arange(10,300,1)

#initialises array for angular momentum values for which a radial plunge orbit will occur
rad_plunge_l=[]


#sets the initial radial position of the photon
r_init =  100000


#intiialises array for the critical impact parameters values
b_crit_val=[]

#sets the speed of the photon
v = 1
b = 3.2
phi_diff=0.0

#initialises the arrays for the deflection angles calculated via the RK4 method, and from the large impact parameter analytical solution
phi_diff_vals=[]
theory_vals=[]

#for loop through which the deflection angle of the photon is determined as the impact parameter is incremented
for j in range(len(b_vals)):
	#initialises the array for the quantities to be calculated via the runge kutta method
	r_vals = []
	phi_vals = []
	turn_pt_vals=[]

	#sets the values for impact parameter
	b = b_vals[j]
	#sets the initial value for the phi co-ordinate calculated from the geometry wrt to the impact parameter and initial radial position 
	phi_init = np.arcsin(b/r_init)

	#calculates the initial test value to determine whether the turning point of the orbit has been reached
	turn_pt_init = func_r_test(r_init,b,v,phi_init)

	#appends the initial test value to an array
	turn_pt_vals.append(turn_pt_init)

	#appends initial values of the r and phi co-ordinates to their respective arrays
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
	#calculates angle of deflection expected from large impact parameter approximation
	phi_diff_theory = 2/b
	#appends RK4 value to array
	phi_diff_vals.append(phi_diff)
	#prints iteration of impact parameter loop (acts as mile marker)
	print(j)
	#appends theory value to array
	theory_vals.append(phi_diff_theory)
	
	print(phi_diff)


#creates plot
ax = plt.subplot(111)

result = hypmodel.fit(phi_diff_vals,x=b_vals,a=-0.0005825,b=2.3224)

#plots rk4 deflection angles as a function of impact parameter
ax.plot(b_vals,phi_diff_vals, label='Runge-Kutta 4th Order Method')
#plots theoretical deflection angles as a function of impact parameter
ax.plot(b_vals,theory_vals, label = 'Analytical Solution')
#plots the hyperbolic model fit
ax.plot(b_vals,result.best_fit, label=r'Hyperbolic Model: $\delta \phi = \alpha + \gamma b^{-1}$, $\alpha = -0.0005825$; $\gamma=2.3224$')


#sets y-axis label,x-axis label, x-axis bounds, and y-axis bounds
ax.set_xlabel('Impact Parameter $b$', fontsize=16)
ax.set_ylabel('Deflection Angle $\delta \phi$ (Rad)', fontsize=16)
ax.tick_params(labelsize=14)
ax.set_xlim(0,300)
ax.set_ylim(0, 0.5)

print(result.params)

#applies a grid to the plot
ax.grid(True,alpha=0.5)

#adds a legends to the plot
ax.legend(loc='upper right', fontsize='x-large')

#shows the resulting plot
plt.show()



#saves the arrays for the impact parameter, the rk4 deflection angle, and theoretical deflection angles to a csv file
dict = {'Impact Parameter': b_vals, 'Deflection Angle': phi_diff_vals, 'Deflection Angle (Theory)': theory_vals}

df=pd.DataFrame(dict)

df.to_csv('deflection_angles_r='+str(r_init)+'_new.csv')
