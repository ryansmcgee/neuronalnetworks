from __future__ import division

from NeuronNetwork import NeuronNetwork

import numpy as numpy

class LIFNetwork(NeuronNetwork):

	def __init__(self):

		#############################
		# NEURON PARAMETER VECTORS: #
		#############################
		""" Left to subclasses to define according to its neuron model specifications"""

		############################
		# INPUT PARAMETER VECTORS: #
		############################
		""" Left to subclasses to define according to its neuron model specifications"""


	def add_neurons(self):
		"""  """
		pass


	def add_inputs(self):
		"""  """
		pass


	def log_current_variable_values(self):
		"""  """
		pass


	def initialize_simulation(self, T_max=None, deltaT=None, integrationMethod=None):
		"""
		Set the simulation variable attributes (e.g., T, deltaT)
		Initialize time related and length-of-simulation dependent variables and containers
		Initialize dataLogs data structures
		Any other model-specific network or dynamics initialzations
		Set the simulationInitialized flag to true
		"""
		pass


	def network_update(self):
		"""  """
		pass