from __future__ import division
from abc import ABCMeta, abstractmethod

import numpy as numpy

# Convenience index constants for readability:
LO = 0
HI = 1

def exitOnNetworkGeometryError(errorMsg):
	print "NetworkGeometry Error: " + errorMsg
	exit()

class NetworkGeometry(object):

	__metaclass__ = ABCMeta

	def __init__(self):

		# Total number of neurons in the network:
		self.N = 0		#Initialized to 0, incremented within the addNeurons() method when neurons are added.

		self.parametricCoords	= None
		self.cartesianCoords 	= None

		self.distances				= None

	@abstractmethod
	def convert_to_parametric_coords(self, origCoords, origCoordType='cartesian'):
		""" """
		pass


	@abstractmethod
	def convert_to_cartesian_coords(self, origCoords, origCoordType='parametric'):
		""" """
		pass


	@abstractmethod
	def calculate_distances(self):
		""" """
		pass


	@abstractmethod
	def add_neurons(self, numNeuronsToAdd):
		""" Increment the count of total neurons in the network """
		pass


	@abstractmethod
	def position_neurons(self, positioning, Coords=None, bounds=None, neuronIDs=None):
		""" """
		pass