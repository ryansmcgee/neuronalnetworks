from __future__ import division

from NetworkGeometry import *

import numpy as numpy

class CylinderSurface(NetworkGeometry):

	numDimensions = 2

	# Flags indicating the surface plane dimensions that have torroidal wrapping:
	torroidal = [True, False]

	def __init__(self, h, r=None, w=None, origin=None):

		NetworkGeometry.__init__(self)

		self.h 	= h

		if(r is None and w is None):
			exitOnNetworkGeometryError("CylinderSurface must have specified radius (constructor arg 'r') or surface plane width (constructor arg 'w').")
		elif(r is not None): #(and w is provided):
			self.r = r
			# Calculate the surface plane width (w) equivalent to given radius (r):
			self.w = 2*numpy.pi * self.r
		elif(w is not None): #(and r is provided):
			self.w = w
			# Calculate the radius (r) equivalent to given surface plane width (w):
			self.r = self.w / (2*numpy.pi)

		# CylinderSurface parametric coords: (theta, h)
		self.parametricCoords		= numpy.full(shape=[0, CylinderSurface.numDimensions], fill_value=numpy.nan)
		# CylinderSurface surface plane coords: (x, y) relative to rectangular plane representing unwrapped cylinder surface
		self.surfacePlaneCoords		= numpy.full(shape=[0, CylinderSurface.numDimensions], fill_value=numpy.nan)
		# CylinderSurface cartesian coords: (x, y, z)
		self.cartesianCoords		= numpy.full(shape=[0, 3], fill_value=numpy.nan)

		# Matrix of distances between neuron positions:
		self.distances 	= numpy.full(shape=[0, 0], fill_value=numpy.nan)

		# Local origin of the geometry coordinate system relative to the global coordinate system, defined in 3d cartesian coordinates:
		self.origin 	= origin if (origin is not None) else numpy.zeros(3)


	def convert_to_parametric_coords(self, origCoords, origCoordType='cartesian'):

		paramCoords = numpy.full(shape=[len(origCoords), CylinderSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'cartesian'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				z = coord[2] - self.origin[2]
				h = z
				theta = numpy.tan(x/y)
				if(x<0 and y>0):
					theta = numpy.pi - theta
				elif(x<0 and y<0):
					theta = numpy.pi + theta
				elif(x>0 and y<0):
					theta = 2*numpy.pi - theta
				paramCoords[i][0] = theta
				paramCoords[i][1] = h

			elif(origCoordType.lower() == 'surface_plane'):
				w = coord[0]
				h = coord[1]
				theta = w / self.r 	# arc_length = r * central_angle
				paramCoords[i][0] = theta
				paramCoords[i][1] = h

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return paramCoords


	def convert_to_cartesian_coords(self, origCoords, origCoordType='parametric'):

		cartCoords = numpy.full(shape=[len(origCoords), 3], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'parametric'):
				theta 	= coord[0]
				h 		= coord[1]
				cartCoords[i][0] = self.r*numpy.cos(theta) + self.origin[0]
				cartCoords[i][1] = self.r*numpy.sin(theta) + self.origin[1]
				cartCoords[i][2] = h + self.origin[2]

			elif(origCoordType.lower() == 'surface_plane'):
				w = coord[0]
				h = coord[1]
				theta = w / self.r 	# arc_length = r * central_angle
				cartCoords[i][0] = self.r*numpy.cos(theta) + self.origin[0]
				cartCoords[i][1] = self.r*numpy.sin(theta) + self.origin[1]
				cartCoords[i][2] = h + self.origin[2]

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return cartCoords


	def convert_to_surfaceplane_coords(self, origCoords, origCoordType='parametric'):
		planeCoords = numpy.full(shape=[len(origCoords), CylinderSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'cartesian'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				z = coord[2] - self.origin[2]
				h = z
				theta = numpy.tan(x/y)
				if(x<0 and y>0):
					theta = numpy.pi - theta
				elif(x<0 and y<0):
					theta = numpy.pi + theta
				elif(x>0 and y<0):
					theta = 2*numpy.pi - theta
				w = self.r*theta
				planeCoords[i][0] = w
				planeCoords[i][1] = h

			elif(origCoordType.lower() == 'parametric'):
				theta 	= coord[0]
				h 		= coord[1]
				w 		= self.r*theta
				print "----------------"
				print "theta "+str(theta)
				print "h "+str(h)
				print "r "+str(self.r)
				print "w "+str(w)
				planeCoords[i][0] = w
				planeCoords[i][1] = h

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
				# Handle torroidal wrapping of cylinder plane or annulus rect. prism in x dimension for euclidean distance calculation:
				if(manhattanDists[0] > self.w/2):
					manhattanDists[0]	= self.w - manhattanDists[0]
				# Calculate the euclidean distance between these points i and j:
				# (works for 2D/3D/ND)
				dist 	= numpy.linalg.norm(manhattanDists)
				self.distances[i,j]	= dist


	def add_neurons(self, numNeuronsToAdd):
		#--------------------
		# Increment the count of total neurons in the network:
		#--------------------
		self.N += numNeuronsToAdd

		C_temp = numpy.full(shape=(self.N, CylinderSurface.numDimensions), fill_value=numpy.nan)
		C_temp[:(self.N-numNeuronsToAdd), :] = self.parametricCoords
		self.parametricCoords = C_temp

		C_temp = numpy.full(shape=(self.N, CylinderSurface.numDimensions), fill_value=numpy.nan)
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
		intervals 	= { 'theta':(0, 2*numpy.pi), 'h':(0, self.h), 'w':(0, self.w) }
		if('theta' in bounds.keys()):
			intervals['theta'][LO] = min( max(bounds['theta'][LO], 0), 2*numpy.pi )
			intervals['theta'][HI] = max( min(bounds['theta'][HI], 2*numpy.pi), 0 )
		if('h' in bounds.keys()):
			intervals['h'][LO] = min( max(bounds['h'][LO], 0), self.h )
			intervals['h'][HI] = max( min(bounds['h'][HI], self.h), 0 )
		if('w' in bounds.keys()):
			intervals['w'][LO] = min( max(bounds['w'][LO], 0), self.w )
			intervals['w'][HI] = max( min(bounds['w'][HI], self.w), 0 )

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
				self.parametricCoords[nID][0] = numpy.random.uniform(low=intervals['theta'][LO], high=intervals['theta'][HI])
				self.parametricCoords[nID][1] = numpy.random.uniform(low=intervals['h'][LO], high=intervals['h'][HI])
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
			int_w	= intervals['w'][HI] - intervals['w'][LO] 	# The width of the effective surface plane interval
			int_h	= intervals['h'][HI] - intervals['h'][LO] 	# The height of the effective surface plane interval
			n_w 	= numpy.rint( numpy.sqrt( (int_w/int_h)*numNeuronsToPosition + (numpy.square(int_w-int_h)/(4*int_h**2)) ) - (int_w - int_h)/(2*int_h) )# num cols along surface plane width
			n_h 	= numpy.ceil( numNeuronsToPosition / n_w)	# num rows along surface plane height
			# 2) Calculate the space between each row and between each column:
			d_w 	= int_w / n_w		# For torroidal wrapping along 'w' axis, neurons are on ONLY one vertical edge so there are n_w gaps between columns
			d_h 	= int_h / (n_h-1)	# For torroidal wrapping along 'w' axis, neurons are on BOTH horizontal edges so there are n_h-1 gaps between columns
			# 3) Iterate through this evenly spaced grid assigning positions to neurons:
			for i, nID in enumerate(neuronIDs):
				# Determine the row and col index for the ith neuron to be placed:
				r_i = i // n_w
				c_i = i %  n_w
				self.surfacePlaneCoords[nID][0] = c_i*d_w
				self.surfacePlaneCoords[nID][1] = r_i*d_h
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
					if(len(coord) == CylinderSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['r'][LO] and coord[0]<=intervals['r'][HI] and coord[1]>=intervals['h'][LO] and coord[1]<=intervals['h'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the geometry \'"+str(self.geometry)+"\' ("+str(CylinderSurface.numDimensions)+")")
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
					if(len(coord) == CylinderSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['w'][LO] and coord[0]<=intervals['w'][HI] and coord[1]>=intervals['h'][LO] and coord[1]<=intervals['h'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the geometry \'"+str(self.geometry)+"\' ("+str(CylinderSurface.numDimensions)+")")
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