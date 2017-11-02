from __future__ import division

from NetworkGeometry import *

import numpy as numpy

class SpheroidSurface(NetworkGeometry):

	numDimensions = 2

	geometry = "SpheroidSurface"

	def __init__(self, r_xy, r_z, origin=None):

		NetworkGeometry.__init__(self)

		if(r_xy is None and r_z is None):
			exitOnNetworkGeometryError("SpheroidSurface must have specified xy-radius (constructor arg 'r_xy') and z-radius (constructor arg 'r_z').")

		self.r_xy	= r_xy
		self.r_z 	= r_z

		# SpheroidSurface parametric coords: (theta (longitude angle), phi (latitude angle))
		self.parametricCoords		= numpy.full(shape=[0, SpheroidSurface.numDimensions], fill_value=numpy.nan)
		# SpheroidSurface cartesian coords: (x, y, z)
		self.cartesianCoords		= numpy.full(shape=[0, 3], fill_value=numpy.nan)

		# Matrix of distances between neuron positions:
		self.distances 	= numpy.full(shape=[0, 0], fill_value=numpy.nan)

		# Local origin of the geometry coordinate system relative to the global coordinate system, defined in 3d cartesian coordinates:
		self.origin 	= origin if (origin is not None) else numpy.zeros(3)


	def convert_to_parametric_coords(self, origCoords, origCoordType='cartesian'):

		if(origCoordType != "parametric"):
			exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not supported. Expected ('cartesian'), which is the default parameter. The SpheroidSurface geometry class only stores and converts between parametric (theta, phi) and cartesian (x,y,z) coordinate systems.")

		paramCoords = numpy.full(shape=[len(origCoords), SpheroidSurface.numDimensions], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			# Based on inverting the convert_to_cartesian_coords, see that function (known to work) for more info:
			x = coord[0] - self.origin[0]
			y = coord[1] - self.origin[1]
			z = coord[2] - self.origin[2]

			phi 	= numpy.arccos( z/self.r_z ) 					# z = r_z*cos(phi)
			phi 	= phi - numpy.pi/2 								# reverse of angle adjustment in convert_to_cartesian_coords
			theta 	= numpy.arccos( x/(self.r_xy*numpy.sin(phi)) )	# x = r_x*cos(theta)*sin(phi) 	using phi found in lines above

			paramCoords[i][0] = theta
			paramCoords[i][1] = phi

		return paramCoords


	def convert_to_cartesian_coords(self, origCoords, origCoordType='parametric'):

		if(origCoordType != "parametric"):
			exitOnNetworkGeometryError("The provided origCoordType \'"+str(origCoordType)+"\' is not supported. Expected ('parametric'), which is the default parameter. The SpheroidSurface geometry class only stores and converts between parametric (theta, phi) and cartesian (x,y,z) coordinate systems.")

		cartCoords = numpy.full(shape=[len(origCoords), 3], fill_value=numpy.nan)

		for i, coord in enumerate(origCoords):
			# The lon,lat --> x,y,z conversion used below assumes domains lon[0,2pi], lat[0,pi]
			# The geodesic library assumes domains lon[-pi(-180), pi(180)], lat[-pi/2(-90), pi/2(90)]
			#	- these domains are used for positioning the neurons
			# - Adjusting lon/lat domain (starting from geodesic domain) as below seems to give cartesian positions that align with the corresponding lon/lat coords

			# print "coord = " + str(numpy.degrees(coord))
			# coord = coord + [0, numpy.pi/2]
			theta 	= coord[0] + 0.0
			phi 	= coord[1] + numpy.pi/2
			# print "coord = " + str(numpy.degrees(coord)) + " ***"

			x 	= self.r_xy*numpy.sin(phi)*numpy.cos(theta)	# x = r_x*cos(theta)*sin(phi)	# from http://mathworld.wolfram.com/Spheroid.html
			y	= self.r_xy*numpy.sin(phi)*numpy.sin(theta)	# y = r_y*sin(theta)*sin(phi)	r_x = r_y for spheroid
			z	= self.r_z*numpy.cos(phi)					# z = r_z*cos(phi)
			z 	*= -1 # further adjustment to give cartesian positions that align with the corresponding lon/lat coords (flip along z so negative lat angles give negative z's)

			cartCoords[i][0] = x + self.origin[0]
			cartCoords[i][1] = y + self.origin[1]
			cartCoords[i][2] = z + self.origin[2]

		return cartCoords


	def calculate_distances(self):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Calculates the distance between all neuron positions *along the surface* (using geodesic distances) ~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# The geographiclib.geodesic module is needed to calculate geodesic distance along the surface of a spheroid.
		# - Library is based on Karney 2013 (seemingly written by Karney himself)
		# - See https://geographiclib.sourceforge.io/1.48/python/
		try:
			from geographiclib.geodesic import Geodesic
		except:
			exitOnNetworkGeometryError("The geographiclib.geodesic module is needed to calculate geodesic distance along the surface of a spheroid, but this module could not be imported.")
		# The Geodesic object constructor has two parameters:
		#	- a: the equatorial (xy) radius of the ellipsoid
		#	- f: the flattening of the ellipsoid. b = a(1-f) --> f = 1 - b/a  (where b is the polar (z) axis)
		a 			= self.r_xy
		b 			= self.r_z
		f 			= 1.0 - b/a
		geodesic 	= Geodesic(a, f)

		# Convert parametric coordinates (theta(lon), phi(lat)) from radian to degree angles
		degreecoords	= numpy.degrees(self.parametricCoords)
		# print self.coordinates
		# print "degrees"
		# print degreecoords

		for i in range(self.N):
			for j in range(self.N):
				# Geodesic.Inverse(...) calculates the geodesic distance between two points along a spheroid surface.
				#	arguments: 	(lat1, lon1, lat2, lon2, outmask=1025)
				#				- expects lat1, lon1, lat2, lon2 to be latitude(theta)/longitude(phi) angles in degrees
				#				- outmask=1025 tells it to return the distance measure
				#	returns:	a Geodesic dictionary including key-value pair 's12', which is the calculated distance between the points.
				lat1	= degreecoords[i][1]
				lon1	= degreecoords[i][0]
				lat2	= degreecoords[j][1]
				lon2	= degreecoords[j][0]
				# print geodesic.Inverse(lat1, lon1, lat2, lon2, outmask=1025)
				# print "lat" + str(lat1) + " lon" + str(lon1) + "  ::  lat" + str(lat2) + " lon" + str(lon2)
				dist 	= geodesic.Inverse(lat1, lon1, lat2, lon2, outmask=1025)['s12']
				self.distances[i,j]	= dist


	def add_neurons(self, numNeuronsToAdd):
		#--------------------
		# Increment the count of total neurons in the network:
		#--------------------
		self.N += numNeuronsToAdd

		C_temp = numpy.full(shape=(self.N, SpheroidSurface.numDimensions), fill_value=numpy.nan)
		C_temp[:(self.N-numNeuronsToAdd), :] = self.parametricCoords
		self.parametricCoords = C_temp

		D_temp = numpy.full(shape=(self.N, self.N), fill_value=numpy.nan)
		D_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.distances
		self.distances = D_temp

		return


	def position_neurons(self, positioning='random', coords=None, bounds={}, neuronIDs=None):
		#---------------------------------------------------------------------------------------
		# Define effective positioning intervals according to bounds given in this method call:
		#---------------------------------------------------------------------------------------
		# Initialize intervals to the full range of the constructed geometry:
		intervals 	= { 'theta':(-1*numpy.pi, numpy.pi), 'phi':(-1*numpy.pi/2, numpy.pi/2) }  #[[-180deg, 180deg], [-90deg, 90deg]]
		if('theta' in bounds.keys()):
			intervals['theta'][LO] = min( max(bounds['theta'][LO], -1*numpy.pi), numpy.pi )
			intervals['theta'][HI] = max( min(bounds['theta'][HI], numpy.pi), -1*numpy.pi )
		if('phi' in bounds.keys()):
			intervals['theta'][LO] = min( max(bounds['theta'][LO], -1*numpy.pi/2), numpy.pi/2 )
			intervals['theta'][HI] = max( min(bounds['theta'][HI], numpy.pi/2), -1*numpy.pi/2 )

		# If no neuron IDs were specified, default to positioning all neurons in the network:
		if(neuronIDs is None):
			neuronIDs	= range(self.N)
		numNeuronsToPosition	= len(neuronIDs)

		###################################
		# Position the specified neurons:
		###################################
		# positioning: 'random' | 'even' | 'given_parametric_coords'

		#----------------------------------------------
		# Randomly position neurons on the intervals: -
		#----------------------------------------------
		if(positioning.lower() == 'random'):
			# New method 8/9/17:
			# Following the procedure for randomly picking points on ellipsoide described in this post:
			#	https://math.stackexchange.com/a/982833/355885
			for i, nID in enumerate(neuronIDs):
				acceptableRandPosGenerated	= False
				while(acceptableRandPosGenerated == False):
					# 1) Genereate a point uniformly on the sphere:
					#		This is done as described here: http://mathworld.wolfram.com/SpherePointPicking.html
					u 	= numpy.random.uniform(low=0.0, high=1.0)
					v 	= numpy.random.uniform(low=0.0, high=1.0)
					randTheta	= 2*numpy.pi*u
					randPhi		= numpy.arccos(2*v - 1)
					# 2) Convert spherical coordinates to cartesian coordinates to continue with procedure using x,y,z
					a 	= self.r_xy	# a = b for spheroids
					b 	= self.r_xy
					c 	= self.r_z
					x 	= a*numpy.cos(randTheta)*numpy.sin(randPhi)
					y 	= b*numpy.sin(randTheta)*numpy.sin(randPhi)
					z 	= c*numpy.cos(randPhi)
					# 3) Calculate the probability of accepting this random sphere point given the spheroid mapping:
					mu_xyz	= numpy.sqrt( (a*c*y)**2 + (a*b*z)**2 + (b*c*x)**2 )
					mu_max	= b*c
					prob_accept	= mu_xyz/mu_max
					# Use the calculated probability of acceptance to determine if this point is accepted/rejected:
					if(numpy.random.uniform(low=0, high=1) < prob_accept):
						# This randomly generated point has been accepted as a random point on our ellipsoid.
						# Store the spherical coordinates representing this point:
						self.parametricCoords[nID][0]	= randTheta - numpy.pi
						self.parametricCoords[nID][1]	= randPhi - numpy.pi/2
						acceptableRandPosGenerated	= True
					else:
						# This randomly generated point has been rejected as a random point on our ellipsoid.
						# Try again:
						pass

			self.cartesianCoords = self.convert_to_cartesian_coords(self.parametricCoords, 'parametric')

		#--------------------------------------------
		# Evenly position neurons on the intervals: -
		#--------------------------------------------
		elif(positioning.lower() == 'even'):
			# Generate evenly spaced parametric (theta, phi) coords for the neurons:
			# Note: this implementation spaces neurons regularly (grid-like) but not quite evenly - neurons are closer to each other near poles.
			# 		implementing truly evenly-spaced points on a sphere is difficult ("packing points on sphere"), moreso for spheroid.
			# 1) Determine how many "rows"/"cols" to use to create a nearly square grid over the effective intervals:
			#    - (using https://math.stackexchange.com/questions/1039482/how-to-evenly-space-a-number-of-points-in-a-rectangle)
			int_lon	= intervals['theta'][HI] - intervals['theta'][LO] 	# The interval of the longitudinal angle
			int_lat	= intervals['phi'][HI] - intervals['phi'][LO] 		# The interval of the latitudinal angle
			circ_lon= self.r_xy*int_lon 	# The circumferal 'width' of the effective interval (arclen = r * angle)
			circ_lat= self.r_z*int_lat 		# The circumferal 'height' of the effective interval (arclen = r * angle)
			n_lon 	= numpy.rint( numpy.sqrt( (circ_lon/circ_lat)*numNeuronsToPosition + (numpy.square(circ_lon-circ_lat)/(4*circ_lat**2)) ) - (circ_lon - circ_lat)/(2*circ_lat) )# num cols along
			n_lat 	= numpy.ceil( numNeuronsToPosition / n_lon)	# num rows along
			# 2) Calculate the angle between each latitudinal "row" and between each longitudinal "column":
			d_lon 	= int_lon / n_lon		# For torroidal wrapping longitudinally, there are n_lon gaps between longitudinal columns
			d_lat 	= int_lat / (n_lat+1)	# There must be space above/below the top/bottom rows so that neurons are not piled on the poles (also no torroidal wrapping latitudinally) so there are n_lat+1 gaps between latitudinal rows
			# 3) Iterate through this evenly spaced grid assigning positions to neurons:
			for i, nID in enumerate(neuronIDs):
				# Determine the row and col index for the ith neuron to be placed:
				r_i = (i // n_lon) + 1 	# we need to add 1 to the r_i multiplier so that no row is at 0*d_lat, which would pile neurons on the pole
				c_i = i % n_lon
				self.parametricCoords[nID][0] = c_i*d_lon
				self.parametricCoords[nID][1] = intervals['phi'][LO] + r_i*d_lat

				# print "$$$$"
				# print "["+str(nID)+"]"
				# print r_i
				# print str(r_i*d_lat)
				# print self.parametricCoords[nID]

			# 4) Calculate the equivalent parametric and cartesian coords for the positioned neurons:
			self.cartesianCoords 	= self.convert_to_cartesian_coords(self.parametricCoords, 'parametric')

		#--------------------------------------------------------
		# Position neurons at the given parametric coords: -
		#--------------------------------------------------------
		elif(positioning.lower() == 'given_parametric_coords'):
			# Check that the list of neuronIDs to update and the list of given positions are the same length:
			if(numNeuronsToPosition != len(coords)):
				for i, coord in enumerate(coords):
					# Check that the coordinate matches the number of dimensions of this geometry:
					if(len(coord) == SpheroidSurface.numDimensions):
						# Check that the given coordinate is valid (within bounds of geometry):
						if(coord[0]>=intervals['r'][LO] and coord[0]<=intervals['r'][HI] and coord[1]>=intervals['h'][LO] and coord[1]<=intervals['h'][HI]):
							self.parametricCoords[neuronIDs] = coord
						else:
							exitOnNetworkGeometryError("The given coord "+str(coord)+" falls outside the bounds of the geometry or the specified interval.")
					else:
						exitOnNetworkGeometryError("The dimensionality of given coordinate(s) ("+str(len(coord))+") does not match the parametric dimensionality of the SpheroidSurface geometry ("+str(SpheroidSurface.numDimensions)+")")
			else:
				exitOnNetworkGeometryError("The number of given coordinate tuples ("+str(len(coords))+") does not match the given number of neuron IDs to position ("+str(numNeuronsToPosition)+")")

			self.cartesianCoords 	= self.convert_to_cartesian_coords(self.parametricCoords, 'parametric')

		#---------------------------------
		# Unrecognized positioning mode:
		else:
			exitOnNetworkGeometryError("Neuron positioning mode \'"+str(positioning)+"\' is not recognized. Expected one of ['random'|'even'|'given_parametric_coords']")

		###################################################
		# Calculate Distances between positioned neurons: #
		###################################################
		self.calculate_distances()

		return