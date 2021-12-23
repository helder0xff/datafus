from Kalman.Kalman import Kalman
import matplotlib.pyplot as plt
from random import gauss
import numpy as np
import random
import argparse

INITIAL_P 				= -20
INITIAL_V 				= 10
INITIAL_A 				= -0.1
DISCRETIZATION_MODEL 	= "Euler" 					# discretization model
MODEL 					= "ConstantAcceleration1D"	# model
SAMPLE_TIME 			= 1							# sample time 
INITIAL_STATE 			= [	INITIAL_P, 
							INITIAL_V, 
							INITIAL_A]	 			# initial state
VARIANCES 				= [10, .2] 				# variances
PROCESS_NOISE 			= [0.5, 1, .001]
MEASURED_STATES 		= [True, True, False]		# measured states
INPUTS_MATRIXES			= [[0,0,0.5]]


def getMeasurements(t):
	if t > 150 and t <= 250:
		u = .05
	elif t > 250 and t <= 350:
		u = -.06
	else:
		u = 0

	if t > 0:
		a = INITIAL_A
		getMeasurements.a += u*INPUTS_MATRIXES[0][2]
		a = getMeasurements.a
		getMeasurements.v += a
		getMeasurements.p += getMeasurements.v
	
	return gauss(getMeasurements.p, VARIANCES[0]), gauss(getMeasurements.v, VARIANCES[1]), getMeasurements.a, u
getMeasurements.p = INITIAL_P
getMeasurements.v = INITIAL_V
getMeasurements.a = INITIAL_A

def main():
	p, v, a, u = list(getMeasurements(0))
	getMeasurements.p = INITIAL_P
	getMeasurements.v = INITIAL_V
	getMeasurements.a = INITIAL_A

	
	pArr 	= []
	vArr 	= []
	aArr 	= []
	pPredictedArr	= []
	vPredictedArr 	= []
	apArr 	= []
	tArr 	= []
	for i in range(500):
		p, v, a, u = list(getMeasurements(i))
		measurements = [p, v]
		if i == 300:
			MEASURED_STATES[1] = False
			k.update(MEASURED_STATES)
		if i == 400:
			MEASURED_STATES[1] = True
			k.update(MEASURED_STATES)

		if i == 0:
			k =Kalman(	DISCRETIZATION_MODEL,
						MODEL,
						SAMPLE_TIME,
						[p, v, 0],
						VARIANCES,
						PROCESS_NOISE,
						MEASURED_STATES,
						INPUTS_MATRIXES)
		else:
			k.computeNextState(measurements, [u])
		x, P = k.getState()
		pArr.append(getMeasurements.p)
		vArr.append(getMeasurements.v)
		aArr.append(getMeasurements.a)
		pPredictedArr.append(x[0])
		vPredictedArr.append(x[1])
		apArr.append(x[2])
		tArr.append(i)

	plt.subplot(1, 3, 1)
	plt.plot( tArr, pArr, label = "p", color = 'blue' )
	plt.plot( tArr, pPredictedArr, label = "pPredicted", color = 'orange' )
	plt.legend( )
	plt.title("meaSig: " + str(VARIANCES[0]) + " modSig: " + str(PROCESS_NOISE[0]))
	plt.subplot(1, 3, 2)	
	plt.plot( tArr, vArr, label = "v", color = 'blue' )
	plt.plot( tArr, vPredictedArr, label = "vPredicted", color = 'orange' )
	plt.legend( )
	plt.title("meaSig: " + str(VARIANCES[1]) + " modSig: " + str(PROCESS_NOISE[1]))
	plt.subplot(1, 3, 3)
	plt.plot( tArr, aArr, label = "a", color = 'blue' )
	plt.plot( tArr, apArr, label = "aPredicted", color = 'orange' )
	plt.legend( )
	plt.title("modSig: " + str(PROCESS_NOISE[2]))
	plt.legend( )
	plt.show( )

main()