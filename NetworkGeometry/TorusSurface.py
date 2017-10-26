from __future__ import division

from NetworkGeometry import *

import numpy as numpy

class TorusSurface(NetworkGeometry):

	numDimensions = 2

	# Flags indicating the surface plane dimensions that have torroidal wrapping:
	torroidal = [True, True]

	def __init__(self, r_major=None, r_minor=None, w=None, h=None, origin=None):

		NetworkGeometry.__init__(self)

		if(r_major is not None and r_minor is not None):
			self.r_major 	= r_major
			self.r_minor	= r_minor
			# Calculate the surface plane width and height equivalent to given radii:
			self.w 			= 2*numpy.pi * self.r_major
			self.h 			= 2*numpy.pi * self.r_minor
		elif(w is not None and h is not None):
			self.w = w
			self.h = h
			# Calcualte the radii equivalent to given surface plane width and height:
			self.r_major 	= self.w / (2*numpy.pi)
			self.r_minor 	= self.h / (2*numpy.pi)
		else:
			exitOnNetworkGeometryError("TorusSurface must have specified either both r_major & r_minor OR both w & h.")

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# TorusSurface parametric coords: (theta_major, phi_minor)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self.parametricCoords		= numpy.full(shape=[0, TorusSurface.numDimensions], fill_value=numpy.nan)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# TorusSurface surface plane coords: (x, y) relative to rectangular plane representing unwrapped torus surface
		self.surfacePlaneCoords		= numpy.full(shape=[0, TorusSurface.numDimensions], fill_value=numpy.nan)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# TorusSurface cartesian coords: (x, y, z)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self.cartesianCoords		= numpy.full(shape=[0, 3], fill_value=numpy.nan)

		# Matrix of distances between neuron positions:
		self.distances 	= numpy.full(shape=[0, 0], fill_value=numpy.nan)

		# Local origin of the geometry coordinate system relative to the global coordinate system, defined in 3d cartesian coordinates:
		self.origin 	= origin if (origin is not None) else numpy.zeros(3)


	def convert_to_parametric_coords(self, origCoords, origCoordType='cartesian'):

		paramCoords = numpy.full(shape=[len(origCoords), TorusSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'cartesian'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				z = coord[2] - self.origin[2]
				# Identiry the angle (relative to center of torus loop) of the local ring on which this point lies:
				theta_major = numpy.tan(x/y)
				if(x<0 and y>0):
					theta_major = numpy.pi - theta_major
				elif(x<0 and y<0):
					theta_major = numpy.pi + theta_major
				elif(x>0 and y<0):
					theta_major = 2*numpy.pi - theta_major
				# Given this local ring at this major angle, calculate the minor angle from the local ring origin to the point:
				localRingOrigin = (self.r_major*numpy.cos(theta_major), self.r_major*numpy.sin(theta_major), 0)
				localVecXYZ		= numpy.array([x-localRingOrigin[0], y-localRingOrigin[1], z-localRingOrigin[2]])	# vector from the local ring origin to the point (in 3d)
				localVecXY 		= numpy.array([x-localRingOrigin[0], y-localRingOrigin[1], localRingOrigin[2]])		# vector from the local ring origin to the xy plane projection of the point
				# Calculate the angle between the vector from the local ring center to the point and the projection of that vector in the xy plane:
				phi_minor 		= numpy.arccos(numpy.dot(localVecXYZ, localVecXY) / (np.linalg.norm(localVecXYZ) * np.linalg.norm(localVecXY)))
				# numpy returns the acute angle between the two vectors. adjust as necessary to store the angle (range 0,2pi) relative to the vector pointing from the center of the local ring away from the center of the torus:
				if(z <= localRingOrigin):
					phi_minor 	= 2*numpy.pi - phi_minor

				paramCoords[i][0] = theta_major
				paramCoords[i][1] = phi_minor

			elif(origCoordType.lower() == 'surface_plane'):
				w = coord[0]
				h = coord[1]
				theta_major = w / self.r_major 	# arc_length = r * central_angle
				phi_minor 	= h / self.r_minor 	# arc_length = r * central_angle
				paramCoords[i][0] = theta_major
				paramCoords[i][1] = phi_minor

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return paramCoords


	def convert_to_cartesian_coords(self, origCoords, origCoordType='parametric'):

		cartCoords = numpy.full(shape=[len(origCoords), 3], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'parametric'):
				theta_major 	= coord[0]
				phi_minor 		= coord[1]

				localRingOrigin = (self.r_major*numpy.cos(theta_major) + self.origin[0], self.r_major*numpy.sin(theta_major) + self.origin[1], self.origin[2])
				d_z 	= self.r_minor*numpy.sin(phi_minor)	# displacment of the point from the local ring origin in the z axis
				d_xy 	= self.r_minor*numpy.cos(phi_minor)	# displacment of the point from the local ring origin in the xy plane
				# d_xy is displacement along the vector from the local rign origin to the point in the xy plane, which points in some arbitrary direction.
				# To get d_x and d_y (displacements along axis directions), we need to take into account the angle of the point from the torus center:
				if(theta_major >= -1*numpy.pi/2 and theta_major <= numpy.pi*2):
					d_x = d_xy*numpy.cos(theta_major)
					d_y = d_xy*numpy.sin(theta_major)
				else:
					d_x = d_xy*numpy.cos(numpy.pi - theta_major)
					d_y = d_xy*numpy.sin(numpy.pi - theta_major)

				cartCoords[i][0] = self.r_major*numpy.cos(theta_major) + d_x + self.origin[0]
				cartCoords[i][1] = self.r_major*numpy.sin(theta_major) + d_y + self.origin[1]
				cartCoords[i][2] = d_z + self.origin[2]

			elif(origCoordType.lower() == 'surface_plane'):
				# First, convert to parametric coordinates:
				w = coord[0]
				h = coord[1]
				theta_major = w / self.r_major 	# arc_length = r * central_angle
				phi_minor 	= h / self.r_minor 	# arc_length = r * central_angle
				# Then do the same parametric->cartesian conversion as above:
				localRingOrigin = (self.r_major*numpy.cos(theta_major) + self.origin[0], self.r_major*numpy.sin(theta_major) + self.origin[1], self.origin[2])
				d_z 	= self.r_minor*numpy.sin(phi_minor)	# displacment of the point from the local ring origin in the z axis
				d_xy 	= self.r_minor*numpy.cos(phi_minor)	# displacment of the point from the local ring origin in the xy plane
				# d_xy is displacement along the vector from the local rign origin to the point in the xy plane, which points in some arbitrary direction.
				# To get d_x and d_y (displacements along axis directions), we need to take into account the angle of the point from the torus center:
				if(theta_major >= -1*numpy.pi/2 and theta_major <= numpy.pi*2):
					d_x = d_xy*numpy.cos(theta_major)
					d_y = d_xy*numpy.sin(theta_major)
				else:
					d_x = d_xy*numpy.cos(numpy.pi - theta_major)
					d_y = d_xy*numpy.sin(numpy.pi - theta_major)

				cartCoords[i][0] = self.r_major*numpy.cos(theta_major) + d_x + self.origin[0]
				cartCoords[i][1] = self.r_major*numpy.sin(theta_major) + d_y + self.origin[1]
				cartCoords[i][2] = d_z + self.origin[2]

			else:
				exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not recognized. Expected ('cartesian'|'surface_plane')")

		return cartCoords


	def convert_to_surfaceplane_coords(self, origCoords, origCoordType='parametric'):
		planeCoords = numpy.full(shape=[len(origCoords), TorusSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			if(origCoordType.lower() == 'cartesian'):
				x = coord[0] - self.origin[0]
				y = coord[1] - self.origin[1]
				z = coord[2] - self.origin[2]
				# First, do the same cartesian->parametric coordinates as above:
				# Identiry the angle (relative to center of torus loop) of the local ring on which this point lies:
				theta_major = numpy.tan(x/y)
				if(x<0 and y>0):
					theta_major = numpy.pi - theta_major
				elif(x<0 and y<0):
					theta_major = numpy.pi + theta_major
				elif(x>0 and y<0):
					theta_major = 2*numpy.pi - theta_major
				# Given this local ring at this major angle, calculate the minor angle from the local ring origin to the point:
				localRingOrigin = (self.r_major*numpy.cos(theta_major), self.r_major*numpy.sin(theta_major), 0)
				localVecXYZ		= numpy.array([x-localRingOrigin[0], y-localRingOrigin[1], z-localRingOrigin[2]])	# vector from the local ring origin to the point (in 3d)
				localVecXY 		= numpy.array([x-localRingOrigin[0], y-localRingOrigin[1], localRingOrigin[2]])		# vector from the local ring origin to the xy plane projection of the point
				# Calculate the angle between the vector from the local ring center to the point and the projection of that vector in the xy plane:
				phi_minor 		= numpy.arccos(numpy.dot(localVecXYZ, localVecXY) / (np.linalg.norm(localVecXYZ) * np.linalg.norm(localVecXY)))
				# numpy returns the acute angle between the two vectors. adjust as necessary to store the angle (range 0,2pi) relative to the vector pointing from the center of the local ring away from the center of the torus:
				if(z <= localRingOrigin):
					phi_minor 	= 2*numpy.pi - phi_minor
				# Now, convert parametric->surface_plane
				planeCoords[i][0] = w
				planeCoords[i][1] = h

			elif(origCoordType.lower() == 'parametric'):
				theta_major = coord[0]
				phi_minor 	= coord[1]
				w 			= self.r_major*theta_major
				h 			= self.r_minor*phi_minor

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
				# Handle torroidal wrapping of surface plane in each dimension for euclidean distance calculation:
				if(manhattanDists[0] > self.w/2):
					manhattanDists[0]	= self.w - manhattanDists[0]
				if(manhattanDists[1] > self.h/2):
					manhattanDists[1]	= self.h - manhattanDists[1]
				# Calculate the euclidean distance between these points i and j:
				# (works for 2D/3D/ND)
				dist 	= numpy.linalg.norm(manhattanDists)
				self.distances[i,j]	= dist


	def add_neurons(self, numNeuronsToAdd):
		#--------------------
		# Increment the count of total neurons in the network:
		#--------------------
		self.N += numNeuronsToAdd

		C_temp = numpy.full(shape=(self.N, TorusSurface.numDimensions), fill_value=numpy.nan)
		C_temp[:(self.N-numNeuronsToAdd), :] = self.parametricCoords
		self.parametricCoords = C_temp

		C_temp = numpy.full(shape=(self.N, TorusSurface.numDimensions), fill_value=numpy.nan)
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
		intervals 	= { 'theta_major':(0, 2*numpy.pi), 'phi_minor':(0, 2*numpy.pi), 'h':(0, self.h), 'w':(0, self.w) }
		if('theta_major' in bounds.keys()):
			intervals['theta_major'][LO] = min( max(bounds['theta_major'][LO], 0), 2*numpy.pi )
			intervals['theta_major'][HI] = max( min(bounds['theta_major'][HI], 2*numpy.pi), 0 )
		if('phi_minor' in bounds.keys()):
			intervals['phi_minor'][LO] = min( max(bounds['phi_minor'][LO], 0), 2*numpy.pi )
			intervals['phi_minor'][HI] = max( min(bounds['phi_minor'][HI], 2*numpy.pi), 0 )
		if('h' in bounds.keys()):
			intervals['h'][LO] = min( max(bounds['h'][LO], 0), self.h )
			intervals['h'][HI] = max( min(bounds['h'][HI], self.h), 0 )
			if('theta_major' not in bounds.keys()):
				# There wasn't a specified theta_major bound, so calculate the appropriate bound corresponding to the given h bound:
				intervals['phi_minor'][LO] = intervals['h'][LO] / self.r_minor	# arc_length = r * central_angle
				intervals['phi_minor'][HI] = intervals['h'][HI] / self.r_minor	# arc_length = r * central_angle
		if('w' in bounds.keys()):
			intervals['w'][LO] = min( max(bounds['w'][LO], 0), self.w )
			intervals['w'][HI] = max( min(bounds['w'][HI], self.w), 0 )
			if('theta_major' not in bounds.keys()):
				# There wasn't a specified theta_major bound, so calculate the appropriate bound corresponding to the given h bound:
				intervals['theta_major'][LO] = intervals['w'][LO] / self.r_major	# arc_length = r * central_angle
				intervals['theta_major'][HI] = intervals['w'][HI] / self.r_major	# arc_length = r * central_angle

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
				self.parametricCoords[nID][0] = numpy.random.uniform(low=intervals['theta_major'][LO], high=intervals['theta_major'][HI])
				self.parametricCoords[nID][1] = numpy.random.uniform(low=intervals['phi_minor'][LO], high=intervals['phi_minor'][HI])
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
			d_h 	= int_h / n_h		# For torroidal wrapping along 'h' axis, neurons are on ONLY one horizontal edges so there are n_h gaps between columns
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
					if(len(coord) == TorusSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['r'][LO] and coord[0]<=intervals['r'][HI] and coord[1]>=intervals['h'][LO] and coord[1]<=intervals['h'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the geometry \'"+str(self.geometry)+"\' ("+str(TorusSurface.numDimensions)+")")
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
					if(len(coord) == TorusSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['w'][LO] and coord[0]<=intervals['w'][HI] and coord[1]>=intervals['h'][LO] and coord[1]<=intervals['h'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the geometry \'"+str(self.geometry)+"\' ("+str(TorusSurface.numDimensions)+")")
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