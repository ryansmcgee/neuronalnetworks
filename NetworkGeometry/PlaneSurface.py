from __future__ import division

from NetworkGeometry import *

import numpy as numpy

class PlaneSurface(NetworkGeometry):

	numDimensions = 2

	geometry = "PlaneSurface"

	# Flags indicating the surface plane dimensions that have torroidal wrapping:
	torroidal = [False, False]

	def __init__(self, x, y, origin=None):

		NetworkGeometry.__init__(self)

		if(x is None and y is None):
			exitOnNetworkGeometryError("PlaneSurface must have specified x (i.e. width) and y (i.e. length).")

		self.x = x
		self.y = y

		# For functions referencing the dimensions of the surface plane coordinate space in terms of width/height to be consistent with Cylinder & Torus plane geometries:
		self.w = self.x
		self.h = self.y

		# For the plane surface, parametric and surface plane coordinates are defined to b eequivalent.
		# PlaneSurface parametric coords: (x, y)
		self.parametricCoords		= numpy.full(shape=[0, PlaneSurface.numDimensions], fill_value=numpy.nan)
		# PlaneSurface surface plane coords: (x, y) relative to rectangular plane representing plane surface
		self.surfacePlaneCoords		= numpy.full(shape=[0, PlaneSurface.numDimensions], fill_value=numpy.nan)
		# PlaneSurface cartesian coords: (x, y, z)
		self.cartesianCoords		= numpy.full(shape=[0, 3], fill_value=numpy.nan)

		# Matrix of distances between neuron positions:
		self.distances 	= numpy.full(shape=[0, 0], fill_value=numpy.nan)

		# Local origin of the geometry coordinate system relative to the global coordinate system, defined in 3d cartesian coordinates:
		self.origin 	= origin if (origin is not None) else numpy.zeros(3)


	def convert_to_parametric_coords(self, origCoords, origCoordType='cartesian'):

		paramCoords = numpy.full(shape=[len(origCoords), PlaneSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'cartesian'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				z = coord[2] - self.origin[2]
				paramCoords[i][0] = x
				paramCoords[i][1] = y

			elif(origCoordType.lower() == 'surface_plane'):
				x = coord[0]
				y = coord[1]
				paramCoords[i][0] = x
				paramCoords[i][1] = y

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return paramCoords


	def convert_to_cartesian_coords(self, origCoords, origCoordType='parametric'):

		cartCoords = numpy.full(shape=[len(origCoords), 3], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'parametric'):
				x 	= coord[0]
				y	= coord[1]
				cartCoords[i][0] = x + self.origin[0]
				cartCoords[i][1] = y + self.origin[1]
				cartCoords[i][2] = self.origin[2]

			elif(origCoordType.lower() == 'surface_plane'):
				x 	= coord[0]
				y	= coord[1]
				cartCoords[i][0] = x + self.origin[0]
				cartCoords[i][1] = y + self.origin[1]
				cartCoords[i][2] = self.origin[2]

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return cartCoords


	def convert_to_surfaceplane_coords(self, origCoords, origCoordType='parametric'):
		planeCoords = numpy.full(shape=[len(origCoords), PlaneSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'cartesian'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				z = coord[2] - self.origin[2]
				planeCoords[i][0] = x
				planeCoords[i][1] = y

			elif(origCoordType.lower() == 'parametric'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				planeCoords[i][0] = x
				planeCoords[i][1] = y

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return planeCoords


	def calculate_distances(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Calculates the distance between all neuron positions *along the surface* (using surface plane coordiante system) ~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		for i in range(self.N):
			for j in range(self.N):
				manhattanDists	= abs(self.surfacePlaneCoords[i] - self.surfacePlaneCoords[j])
				# Calculate the euclidean distance between these points i and j:
				# (works for 2D/3D/ND)
				dist 	= numpy.linalg.norm(manhattanDists)
				self.distances[i,j]	= dist


	def add_neurons(self, numNeuronsToAdd):
		#--------------------
		# Increment the count of total neurons in the network:
		#--------------------
		self.N += numNeuronsToAdd

		C_temp = numpy.full(shape=(self.N, PlaneSurface.numDimensions), fill_value=numpy.nan)
		C_temp[:(self.N-numNeuronsToAdd), :] = self.parametricCoords
		self.parametricCoords = C_temp

		C_temp = numpy.full(shape=(self.N, PlaneSurface.numDimensions), fill_value=numpy.nan)
		C_temp[:(self.N-numNeuronsToAdd), :] = self.surfacePlaneCoords
		self.surfacePlaneCoords = C_temp

		D_temp = numpy.full(shape=(self.N, self.N), fill_value=numpy.nan)
		D_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.distances
		self.distances = D_temp

		return


	def position_neurons(self, positioning='random', coords=None, bounds={}, neuronIDs=None):
		#---------------------------------------------------------------------------------------
		# Define effective positioning intervals according to bounds given in this method call:
		#---------------------------------------------------------------------------------------
		# Initialize intervals to the full range of the constructed geometry:
		intervals 	= { 'x':(0, self.x), 'y':(0, self.y) }
		if('x' in bounds.keys()):
			intervals['x'][LO] = min( max(bounds['x'][LO], 0), self.x )
			intervals['x'][HI] = max( min(bounds['x'][HI], self.x), 0 )
		if('y' in bounds.keys()):
			intervals['y'][LO] = min( max(bounds['y'][LO], 0), self.y )
			intervals['y'][HI] = max( min(bounds['y'][HI], self.y), 0 )

		# If no neuron IDs were specified, default to positioning all neurons in the network:
		if(neuronIDs is None):
			neuronIDs	= range(self.N)
		numNeuronsToPosition	= len(neuronIDs)

		###################################
		# Position the specified neurons:
		###################################
		# positioning: 'random' | 'even' | 'given_parametric_coords' | 'given_surfaceplane_coords'

		#----------------------------------------------
		# Randomly position neurons on the intervals: -
		#----------------------------------------------
		if(positioning.lower() == 'random'):
			# Randomly generate parametric coords for the neurons:
			for i, nID in enumerate(neuronIDs):
				self.parametricCoords[nID][0] = numpy.random.uniform(low=intervals['x'][LO], high=intervals['x'][HI])
				self.parametricCoords[nID][1] = numpy.random.uniform(low=intervals['y'][LO], high=intervals['y'][HI])
			# Calculate the equivalent surface plane and cartesian coords for the positioned neurons:
			self.surfacePlaneCoords 	= self.convert_to_surfaceplane_coords(self.parametricCoords, 'parametric')
			self.cartesianCoords 		= self.convert_to_cartesian_coords(self.parametricCoords, 'parametric')

		#--------------------------------------------
		# Evenly position neurons on the intervals: -
		#--------------------------------------------
		elif(positioning.lower() == 'even'):
			# Generate evenly spaced surface plane coords for the neurons:
			# 1) Determine how many rows/cols to use to create a nearly square grid over the effective intervals:
			#    - (using https://math.stackexchange.com/questions/1039482/how-to-evenly-space-a-number-of-points-in-a-rectangle)
			int_x	= intervals['x'][HI] - intervals['x'][LO] 	# The width of the effective surface plane interval
			int_y	= intervals['y'][HI] - intervals['y'][LO] 	# The height of the effective surface plane interval
			n_x 	= numpy.rint( numpy.sqrt( (int_x/int_y)*numNeuronsToPosition + (numpy.square(int_x-int_y)/(4*int_y**2)) ) - (int_x - int_y)/(2*int_y) )# num cols along surface plane width
			n_y 	= numpy.ceil( numNeuronsToPosition / n_x)	# num rows along surface plane height
			# 2) Calculate the space between each row and between each column:
			d_x 	= int_x / (n_x-1)		# For NO torroidal wrapping along 'x' axis, neurons are on BOTH one vertical edge so there are n_x-1 gaps between columns
			d_y 	= int_y / (n_y-1)	# For NO torroidal wrapping along 'y' axis, neurons are on BOTH horizontal edges so there are n_y-1 gaps between columns
			# 3) Iterate through this evenly spaced grid assigning positions to neurons:
			for i, nID in enumerate(neuronIDs):
				# Determine the row and col index for the ith neuron to be placed:
				r_i = i // n_x
				c_i = i %  n_x
				self.surfacePlaneCoords[nID][0] = c_i*d_x
				self.surfacePlaneCoords[nID][1] = r_i*d_y
			# 4) Calculate the equivalent parametric and cartesian coords for the positioned neurons:
			self.parametricCoords 	= self.convert_to_parametric_coords(self.surfacePlaneCoords, 'surface_plane')
			self.cartesianCoords 	= self.convert_to_cartesian_coords(self.surfacePlaneCoords, 'surface_plane')

		#--------------------------------------------------------
		# Position neurons at the given parametric coords: -
		#--------------------------------------------------------
		elif(positioning.lower() == 'given_parametric_coords'):
			# Check that the list of neuron_ids to update and the list of given positions are the same length:
			if(numNeuronsToPosition != len(coords)):
				for i, coord in enumerate(coords):
					# Check that the coordinate matches the number of dimensions of this geometry:
					if(len(coord) == PlaneSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['x'][LO] and coord[0]<=intervals['x'][HI] and coord[1]>=intervals['y'][LO] and coord[1]<=intervals['y'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the PlaneSurface geometry ("+str(PlaneSurface.numDimensions)+")")
			else:
				exitOnNetworkGeometryError("The number of given coordinate tuples ("+str(len(coords))+") does not match the given number of neuron IDs to position ("+str(numNeuronsToPosition)+")")

		#------------------------------------------------------
		# Position neurons at the given surface plane coords: -
		#------------------------------------------------------
		elif(positioning.lower() == 'given_surfaceplane_coords'):
			# Check that the list of neuron_ids to update and the list of given positions are the same length:
			if(numNeuronsToPosition != len(coords)):
				for i, coord in enumerate(coords):
					# Check that the coordinate matches the number of dimensions of this geometry:
					if(len(coord) == PlaneSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['x'][LO] and coord[0]<=intervals['x'][HI] and coord[1]>=intervals['y'][LO] and coord[1]<=intervals['y'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the PlaneSurface geometry ("+str(PlaneSurface.numDimensions)+")")
			else:
				exitOnNetworkGeometryError("The number of given coordinate tuples ("+str(len(coords))+") does not match the given number of neuron IDs to position ("+str(numNeuronsToPosition)+")")

		#---------------------------------
		# Unrecognized positioning mode:
		else:
			exitOnNetworkGeometryError("Neuron positioning mode \'"+str(positioning)+"\' is not recognized. Expected one of ['random'|'even'|'given_parametric_planes'|'given_surfaceplane_coords']")

		###################################################
		# Calculate Distances between positioned neurons: #
		###################################################
		self.calculate_distances()

		return