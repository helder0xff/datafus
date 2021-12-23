from .Models import Models
import numpy as np

class Kalman:
	def __init__( 	self,
					discretizeMethod,
					model,
					sampleTime,
					initialState, 
					variances, 
					processNoise,
					measuredStates,
					inputMatrixes):
		self.__A = self.__discretizeMatrix(	discretizeMethod,
											Models(model).getModel(), 
											sampleTime)
		self.__x 	= np.array(initialState)
		self.__R 	= self.__computeR(variances)
		self.__P 	= self.__computeInitialP(self.__R, self.__A.shape[0])
		self.__Q 	= self.__computeQ(processNoise)
		self.__H 	= self.__computeH(measuredStates)
		self.__Bs  	= self.__computeBs(inputMatrixes)
		self.__variances = variances

	def update(self, measuredStates):
		self.__H = self.__computeH(measuredStates)

		nMeasurements = self.__H.shape[0]
		variances = self.__variances[:nMeasurements]
		self.__R = self.__computeR(variances)

	def __computeQ(self, processNoise):
		Q = np.zeros(self.__P.shape)
		for i in range(len(processNoise)):
			Q[i][i] = processNoise[i]

		return Q

	def __computeBs( self, inputMatrixes):
		Bs = []
		for matrix in inputMatrixes:
			Bs.append(np.array(matrix))

		return Bs

	def __computeSinv( self, Pp ):
		S = self.__H.dot( Pp ).dot( self.__H.T ) + self.__R
		try:
			Sinv = np.linalg.inv(S)
		except:
			Sinv = np.identity(S.shape[0])

		return Sinv

	def __computeK( self, Pp ):
		Sinv = self.__computeSinv(Pp)
		
		return (Pp.dot( self.__H.T).dot(Sinv))

	def __predictState( self,us ):
		xp = self.__A.dot( self.__x )
		for i in range(len(us)):
			xp += self.__Bs[i] * us[i]
		Pp = self.__computePp( )

		return xp, Pp

	def __computePp( self ):
		return (self.__A.dot( self.__P.dot(self.__A.T)) + self.__Q)

	def computeNextState( self, measurements, u ):
		xp, Pp = self.__predictState(u)
		K = self.__computeK(Pp)

		nMeasurements = self.__H.shape[0]
		z = np.array(measurements[:nMeasurements])
		
		self.__x = xp + K.dot(z - self.__H.dot(xp))
		self.__P = Pp - K.dot(self.__H.dot(Pp))

	def getState( self ):
		return self.__x, self.__P

	def __computeH( self, measuredStates):
		H = []
		for i in range(measuredStates.count(True)):
			row = np.zeros(len(measuredStates))
			row[i] = 1
			H.append(row)

		return np.array(H)

	def __computeInitialP( self, R, size ):
		pads = size - R.shape[0]
		return np.pad(R,((0, pads),(0, pads)), mode = 'constant', constant_values = 0) 
		

	def __discretizeMatrix( self, method, matrix, sampleTime ):
		"""
		Convert continuous model into discrete one.
		"""
		discreteMatrix = ''
		if "Euler" == method:
			discreteMatrix = self.__discretizeMatrixEuler(matrix, sampleTime)
		elif "Exact" == method:
			discreteMatrix = self.__discretizeMatrixExact(matrix, sampleTime)
		else:
			raise ValueError("ERROR: Not a proper discretization method")

		return discreteMatrix

	def __discretizeMatrixEuler( self, matrix, sampleTime ):
		return (np.identity(matrix.shape[0]) + matrix * sampleTime)

	def __discretizeMatrixExact( self, matrix ):
		discreteMatrix = self.__discretizeMatrixEuler(matrix, sampleTime)
		iterations = self.__exactTerms - 2
		for i in range( iterations ):
			aux 	= np.linalg.matrix_power( self.__A, i + 2 )
			auxsampleTime 	= sampleTime ** ( i + 2 )
			discreteMatrix +=  (( aux * auxsampleTime) / ( i + 2 ) )
		return discreteMatrix
	def __computeR(self, variances):
		return self.__computeCovarianceMatrix(variances)

	def __computeCovarianceMatrix( self, variances ):
		variancesCopy = []
		for variance in variances:
			if None != variance:
				variancesCopy.append(variance)
			else:
				variancesCopy.append(0)
		covMatrix = []
		for variance1 in variancesCopy:
			variancesCopy = np.delete( variancesCopy, 0 ) 
			covariance = [ variance1 * variance1 ]
			for variance2 in variancesCopy:
				covariance.append( variance1 * variance2 )
			covMatrix.append( covariance )
		for i in range( len( variances ) ):
			line = []
			for j in range( i ):
				line.append( 0 )
			covMatrix[ i ] = line + covMatrix[ i ]
		for i in range( len( variances ) ):
			for j in range( len( variances ) ):
				if j > i:
					covMatrix[ j ][ i ] = covMatrix[ i ][ j ]
		
		return np.array( covMatrix )