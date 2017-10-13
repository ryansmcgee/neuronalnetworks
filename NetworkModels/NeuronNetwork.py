from __future__ import division
from abc import ABCMeta, abstractmethod

import numpy as numpy
import pandas as pandas

class NeuronNetwork(object):

	__metaclass__ = ABCMeta

	def __init__(self):

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Network Simulation Variables: ~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self.deltaT 			= 1000	# Initialized to default value
		self.T_max 				= 0.01	# Initialized to default value

		self.t 					= 0
		self.timeStepIndex 		= 0
		self.timeSeries 		= None 	# Initialized in initialze_simulation()
		self.numTimeSteps		= 0 	# Initialized in initialze_simulation()

		self.integrationMethod	= 'trapezoidal'	# <- Default value specified

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Total number of nuerons in the network: ~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Initialized to 0, incremented within the add_neurons() method when neurons are added.
		self.N = 0

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Total number of external inputs to the network: ~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Initialized to 0, incremented within the add_inputs() method when neurons are added.
		self.numInputs = 0

		#############################
		# NEURON PARAMETER VECTORS: #
		#############################
		""" Left to subclasses to define according to its neuron model specifications"""

		############################
		# INPUT PARAMETER VECTORS: #
		############################
		""" Left to subclasses to define according to its neuron model specifications"""

		#~~~~~~~~~~~~~~~~~~~
		# Neuron metadata: ~
		#~~~~~~~~~~~~~~~~~~~
		self.neuronIDs 			= numpy.empty(shape=[0])
		self.neuronSynapseTypes	= numpy.empty(shape=[0])
		self.neuronLabels		= numpy.empty(shape=[0])

		#~~~~~~~~~~~~~~~~~~
		# Input metadata: ~
		#~~~~~~~~~~~~~~~~~~
		self.inputIDs 			= numpy.empty(shape=[0])
		self.inputTypes			= numpy.empty(shape=[0])
		self.inputLabels		= numpy.empty(shape=[0])

		#~~~~~~~~~~~~~~~~~~~~~~~~~
		# Connectivity Matrices: ~
		#~~~~~~~~~~~~~~~~~~~~~~~~~
		# Instantiating connectivity matrices as empty 2D arrays with 0 rows and 0 cols.
		# Connectivity matrices will have rows and cols added as neurons and/or inputs are added to the network.
		self.connectionWeights_synExcit	= numpy.empty(shape=[0, 0])
		self.connectionWeights_synInhib	= numpy.empty(shape=[0, 0])
		self.connectionWeights_gap		= numpy.empty(shape=[0, 0])
		self.connectionWeights_inpExcit	= numpy.empty(shape=[0, 0])
		self.connectionWeights_inpInhib	= numpy.empty(shape=[0, 0])

		#~~~~~~~~~~~~~~~~~~~
		# Network Geometry ~
		#~~~~~~~~~~~~~~~~~~~
		self.geometry = None

		#~~~~~~~~~~~~~~~~~~~~~~~~
		# Initialization Flags: ~
		#~~~~~~~~~~~~~~~~~~~~~~~~
		self.simulationInitialized = False

		#~~~~~~~~~~~~~
		# Data Logs: ~
		#~~~~~~~~~~~~~
		# Dictionary with format: {'variable': {'enabled': True, 'data': []}, ...}
		# Subclass implementation should define the variables (ie dictionary keys) to include.
		self.neuronLogs = {}
		self.inputLogs 	= {}


	@abstractmethod
	def add_neurons(self):
		"""  """
		pass


	@abstractmethod
	def add_inputs(self):
		"""  """
		pass


	def set_network_geometry(self, geometry):
		self.geometry = geometry
		return


	def set_synaptic_connectivity(self, connectivityMatrix, updateNeurons=None, synapseType='excitatory'):
		#  TODO check if any neurons have both excitatory and inhibitory outgoing connections. print a warning if so, but don't change anything. leave up to higher level scripts to make sure their construction of network doesn't have neurons that are both, unless that's what they want

		# Check that the weight matrix provided is in the correct form, and convert the matrix to a numpy.ndarray type:
		try:
			synapticConnectivityMatrix	= connectivityMatrix
			if(not isinstance(synapticConnectivityMatrix, numpy.ndarray)):
				synapticConnectivityMatrix	= numpy.atleast_2d(synapticConnectivityMatrix)
			if(synapticConnectivityMatrix.ndim != 2):
				raise ValueError
		except ValueError:
			print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects argument 'connectivityMatrix' to be a 2D numeric weight matrix as its argument (list-of-lists or numpy.ndarray)")
			exit()

		#--------------------
		# When a list of neurons to update is not given, the connectivity for the entire network is updated.
		#	- 'connectivityMatrix' argument is expected to be an NxN matrix.
		#--------------------
		if(updateNeurons == None):
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(synapticConnectivityMatrix.shape[0] != self.N or synapticConnectivityMatrix.shape[1] != self.N):
				print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects NxN weight matrix to be given as the argument 'connectivityMatrix', where N = num neurons in network. The given matrix has dimensions "+str(synapticConnectivityMatrix.shape[1])+"x"+str(synapticConnectivityMatrix.shape[0])+", but the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the network's ConnectWeights_syn[Excit/Inhib] to the given connectivity matrix, according to the input type (excitatory/inhibitory).
			#--------------------
			if(synapseType.lower() == 'excitatory' or synapseType.lower() == 'e'):
				self.ConnectionWeights_synExcit 	= synapticConnectivityMatrix
			elif(synapseType.lower() == 'inhibitory' or synapseType.lower() == 'i'):
				self.ConnectionWeights_synInhib		= synapticConnectivityMatrix
			else:
				print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects synapseType to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
				exit()

		#--------------------
		# When a list of neurons to update is given, the connectivity for only those neurons are updated.
		#	- 'updateNeurons' is expected to be a list of integer neuron IDs
		#	- 'connectivity' argument  is expected to be an nxN matrix, where n=len(updateNeurons),
		#		and the rows of this matrix are expected to refer to network neurons in the order given by updateNeurons
		#--------------------
		else:
			# Check that the updateNeurons argument is a list of integers in the range 0-N with no duplicates:
			try:
				if(len(updateNeurons) > self.N or not all(isinstance(i, int) for i in updateNeurons) or len(set(updateNeurons)) != len(updateNeurons)):
					print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects the optional argument updateNeurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
					exit()
			except TypeError:
				print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects the optional argument updateNeurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
				exit()
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(synapticConnectivityMatrix.shape[0] != len(updateNeurons) or synapticConnectivityMatrix.shape[1] != self.N):
				print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects mxN weight matrix to be given as the argument connectivityMatrix, where m = num neurons designated for update in the updateNeurons argument, and N = num neurons in network. The given matrix has dimensions "+str(synapticConnectivityMatrix.shape[0])+"x"+str(synapticConnectivityMatrix.shape[1])+", but the num neurons to update is " +str(len(updateNeurons))+ " and the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the rows of the the network's ConnectWeights_syn[Excit/Inhib] matrix corresponding to the neurons
			# to the given connectivity values, according to the input type (excitatory/inhibitory).
			#--------------------
			if(synapseType.lower() == 'excitatory' or synapseType.lower() == 'e'):
				for i in range(len(updateNeurons)):
					updateNeuronID 	= updateNeurons[i]
					self.ConnectionWeights_synExcit[updateNeuronID]	= synapticConnectivityMatrix[i]
			elif(synapseType.lower() == 'inhibitory' or synapseType.lower() == 'i'):
				for i in range(len(updateNeurons)):
					updateNeuronID 	= updateNeurons[i]
					self.ConnectionWeights_synInhib[updateNeuronID]	= synapticConnectivityMatrix[i]
			else:
				print("(NeuronNetwork) Error: The method set_synaptic_connectivity expects synapseType to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
				exit()

		return


	def set_gapJunction_connectivity(self, connectivityMatrix, updateNeurons=None):
				# Check that the weight matrix provided is in the correct form, and convert the matrix to a numpy.ndarray type:
		try:
			gapConnectivityMatrix	= connectivityMatrix
			if(not isinstance(gapConnectivityMatrix, numpy.ndarray)):
				gapConnectivityMatrix	= numpy.atleast_2d(gapConnectivityMatrix)
			if(gapConnectivityMatrix.ndim != 2):
				raise ValueError
		except ValueError:
			print("(NeuronNetwork) Error: The method set_gapJunction_connectivity expects connectivityMatrix to be a 2D numeric weight matrix as its argument (list-of-lists or numpy.ndarray)")
			exit()

		if(updateNeurons == None):
			#--------------------
			# When a list of neurons to update is not given, the connectivity for the entire network is updated.
			#	- 'connectivity' argument is expected to be an NxN matrix.
			#--------------------
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(gapConnectivityMatrix.shape[0] != self.N or gapConnectivityMatrix.shape[1] != self.N):
				print("(NeuronNetwork) Error: The method set_gapJunction_connectivity expects NxN weight matrix to be given as the argument connectivityMatrix, where N = num neurons in network. The given matrix has dimensions "+str(gapConnectivityMatrix.shape[1])+"x"+str(gapConnectivityMatrix.shape[0])+", but the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the network's ConnectWeights_gap to the given connectivity matrix, according to the input type (excitatory/inhibitory).
			#--------------------
			self.ConnectionWeights_gap 	= gapConnectivityMatrix

		else:
			#--------------------
			# When a list of neurons to update is given, the connectivity for only those neurons are updated.
			#	- 'updateNeurons' is expected to be a list of integer neuron IDs
			#	- 'connectivity' argument  is expected to be an nxN matrix, where n=len(updateNeurons),
			#		and the rows of this matrix are expected to refer to network neurons in the order given by updateNeurons
			#--------------------
			# Check that the updateNeurons argument is a list of integers in the range 0-N with no duplicates:
			try:
				if(len(updateNeurons) > self.N or not all(isinstance(i, int) for i in updateNeurons) or len(set(updateNeurons)) != len(updateNeurons)):
					print("(NeuronNetwork) Error: The method set_gapJunction_connectivity expects the optional argument updateNeurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
					exit()
			except TypeError:
				print("(NeuronNetwork) Error: The method set_gapJunction_connectivity expects the optional argument updateNeurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
				exit()
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(gapConnectivityMatrix.shape[0] != len(updateNeurons) or gapConnectivityMatrix.shape[1] != self.N):
				print("(NeuronNetwork) Error: The method set_gapJunction_connectivity expects mxN weight matrix to be given as the argument connectivityMatrix, where m = num neurons designated for update in the updateNeurons argument, and N = num neurons in network. The given matrix has dimensions "+str(gapConnectivityMatrix.shape[0])+"x"+str(gapConnectivityMatrix.shape[1])+", but the num neurons to update is " +str(len(updateNeurons))+ " and the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the rows of the the network's ConnectWeights_gap matrix corresponding to the neurons
			# to the given connectivity values, according to the input type (excitatory/inhibitory).
			#--------------------
			self.ConnectionWeights_gap 	= gapConnectivityMatrix

		return


	def set_input_connectivity(self, connectivityMatrix, inputType='excitatory'):
		#--------------------
		# This method expects as input a IxN matrix representing the weighted mapping of I input signals to N network neurons.
		#--------------------

		inputConnectivityMatrix	= connectivityMatrix
		# Check that the weight matrix provided is in the correct form, and convert the matrix to a numpy.ndarray type:
		try:
			if(not isinstance(inputConnectivityMatrix, numpy.ndarray)):
				inputConnectivityMatrix	= numpy.array(inputConnectivityMatrix)
			if(inputConnectivityMatrix.ndim != 2):
				raise ValueError
		except:
			print("(NeuronNetwork) Error: The method set_input_connectivity expects 2D numeric weight matrix as its argument (list-of-lists or numpy.ndarray)")
			exit()

		# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
		if(inputConnectivityMatrix.shape[1] != self.N):
			print("(NeuronNetwork) Error: The method set_input_connectivity expects IxN weight matrix as its argument, where N = num neurons in network. The given matrix has dimensions I="+str(inputConnectivityMatrix.shape[1])+"xN="+str(inputConnectivityMatrix.shape[0])+", but the current num neurons is "+str(self.N))
			exit()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Set the network's ConnectWeights_inpExcit or ConnectWeights_inp to the given connectivity matrix,
		# according to the input type (excitatory/inhibitory).
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if(inputType.lower() == 'excitatory' or inputType.lower() == 'e'):
			self.ConnectionWeights_inpExcit 	= inputConnectivityMatrix
		elif(inputType.lower() == 'inhibitory' or inputType.lower() == 'i'):
			self.ConnectionWeights_inpInhib		= inputConnectivityMatrix
		else:
			print("(NeuronNetwork) Error: The method set_input_connectivity expects inputType to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
			exit()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# (Re)initialize the inputValues vectors to have lengths matching the newly specified number of inputs
		# (if the number of inputs isn't changed by the new connectivity, inputValues are reset to 0 anyways)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self.inputValues_excit	= numpy.zeros(self.numInputs)
		self.inputValues_inhib	= numpy.zeros(self.numInputs)

		return


	def dirac_delta(self):
		return 1.0/self.deltaT


	@abstractmethod
	def log_current_variable_values(self):
		"""  """
		pass

	
	def initialize_simulation(self, T_max=None, deltaT=None, integrationMethod=None):
		#~~~~~
		# For any parameter values given in the call to initializeSimulation, set the network attributes accordingly.
		# For parameters whose values are not specified in the call to initializeSimulation, continue with the default values given in the class constructor.
		#~~~~~
		if(T_max is not None):
			self.T_max 				= T_max
		if(deltaT is not None):
			self.deltaT 			= deltaT
		if(integrationMethod is not None):
			self.integrationMethod	= integrationMethod

		#~~~~~
		# Initialize time related and length-of-simulation dependent variables and containers
		#~~~~~
		self.timeSeries 		= numpy.arange(0, self.T_max, self.deltaT)
		self.numTimeSteps		= len(self.timeSeries)
		self.timeStepIndex 		= 0
		self.t 					= 0

		for variable in self.neuronLogVariables:
			self.neuronLogs[variable]['data'] = [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]

		for variable in self.inputLogVariables:
			self.inputLogs[variable]['data'] = [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.numInputs)]

		"""
		Define any other model-specific network or dynamics initialzations in subclasses
		"""

		# The simulation is now flagged as initialized:
		self.simulationInitialized	= True

		return


	@abstractmethod
	def network_update(self):
		"""  """
		pass

	def sim_state_valid(self):
		#--------------------
		# If the network has not yet been initialized, call initializeSimulation() now.
		# 	Simulation parameters will be initialized to whatever values the network attributes (eg delta, T_max, etc) currently hold;
		#	(perhaps the default values that were set for these attributes in the constructor)
		# Else, continue with the simulation step under the existing initialization.
		#--------------------
		if(not self.simulationInitialized):
			self.initialize_simulation()

		# Check that we are not going past the time T_max allocated in initializeSimulation:
		if(self.t >= self.T_max):
			print("(NeuronNetwork) Warning: The maximum simulation time has been reached. [t = "+str(self.t)+", T_max = "+str(self.T_max)+", deltaT = "+str(self.deltaT)+", max num timesteps = "+str(self.T_max/self.deltaT)+", current timestep num = " +str(self.timeStepIndex))
			return False

		return True


	def sim_step(self):

		if(not self.sim_state_valid()):
			print("(NeuronNetwork) Warning: Invalid simulation state, this sim_step() will not be executed.")
			return False

		####################################
		# Update neuron dynamics variables #
		####################################
		self.network_update()

		#~~~~~~~~~~~~~~~~~~~~
		# Log the current values of variables for which logging is enabled:
		#~~~~~~~~~~~~~~~~~~~~
		# TODO? (allow user to specify all/none/set of neuron indexes to log for each logable thing?)
		self.log_current_variable_values()

		#~~~~~~~~~~~~~~~~~~~~
		# Increment t and the timestep index:
		#~~~~~~~~~~~~~~~~~~~~
		self.t 				+= self.deltaT
		self.timeStepIndex 	+= 1

		return True


	def run_simulation(self, T_max=None, deltaT=None, integrationMethod=None):
		# Convenience method that just loops sim_steps.
		# Higher level scripts can also use their own loop/logic to implement custom simulation loops.

		if(not self.sim_state_valid()):
			print("(NeuronNetwork) Warning: Invalid simulation state, simulation aborted.")
			return False

		while(self.t < (self.T_max-(self.deltaT/2))):	# The right-hand-side of this conditional is what it is rather than just T_max to avoid numerical roundoff errors causing unexpected conditional outcomes
			if( self.simStep() ):
				pass
			else:
				print("(NeuronNetwork) Error: Invalid simulation state, simulation aborted.")
				return False

		return True


	def get_neuron_IDs(self, synapseTypes=None, labels=None):
		if(synapseTypes is None):
			synapseTypes	= numpy.unique(self.neuronSynapseTypes)
		if(labels is None):
			labels 			= numpy.unique(self.neuronLabels)

		indices_selectedSynTypes	= numpy.in1d(self.neuronSynapseTypes, synapseTypes)
		indices_selectedLabels		= numpy.in1d(self.neuronLabels, labels)

		return numpy.where(indices_selectedSynTypes * indices_selectedLabels)[0]


	def get_neurons_dataframe(self):
		neuronsDataFrame	= pandas.DataFrame( columns=list(self.neuronLogs.keys()) )
		for n, nID in enumerate(self.neuronIDs):
			neuronData	= []
			for variable, data in enumerate(self.neuronLogs):
				neuronData.append( (variable, data) )

			neuronsDataFrame	= pandas.concat([ neuronsDataFrame, pandas.DataFrame.from_items(neuronData) ])

		return neuronsDataFrame


	def get_inputs_dataframe(self):
		inputsDataFrame	= pandas.DataFrame( columns=list(self.inputLogs.keys()) )
		for n, nID in enumerate(self.inputIDs):
			inputData	= []
			for variable, data in enumerate(self.inputLogs):
				inputData.append( (variable, data) )

			inputsDataFrame	= pandas.concat([ inputsDataFrame, pandas.DataFrame.from_items(inputData) ])

		return inputsDataFrame


	def enable_neuron_log(self, variable, enabled=True):
		try:
			self.neuronLogs[variable]['enabled'] = enabled
			return True
		except KeyError:
			print("(NeuronNetwork) Warning: No log for "+str(variable)+" found, call to enable_neuron_log failed.")
			return False


	def enable_input_log(self, variable, enabled=True):
		try:
			self.inputLogs[variable]['enabled'] = enabled
			return True
		except KeyError:
			print("(NeuronNetwork) Warning: No log for "+str(variable)+" found, call to enable_input_log failed.")
			return False