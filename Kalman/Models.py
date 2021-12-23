import numpy as np

class ConstantVelocity1D:
	def __init__( self ):
		self.__A = np.array( [ [ 0, 1 ], [ 0, 0 ] ] )

	def getModel( self ):
		return self.__A

class ConstantAcceleration1D:
	def __init__( self ):
		self.__A = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 0, 0, 0] ] )

	def getModel( self ):
		return self.__A

class Custom:
	def __init__( self, A = [ ] ):
		self.__A = np.array( A )

	def getModel( self ):
		return self.__A	

class Models:
	def __init__( self, model, A = None ):
		if"ConstantVelocity1D" == model:
			self.__A = ConstantVelocity1D().getModel()
		elif"ConstantAcceleration1D" == model:
			self.__A = ConstantAcceleration1D().getModel()
		elif"Custom" == model:
			self.__A = Custom(A).getModel()
		else:
			raise ValueError('ERROR: Unknown model.')

	def getModel( self ):
		return self.__A	

#### end of file ####
