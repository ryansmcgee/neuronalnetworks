from __future__ import division

import numpy as numpy
import pandas as pandas

from NetworkGeometry import NetworkGeometry

class IzhikevichNetwork(object):

	def __init__(self):

+		self.deltaT 			= 1000	# <- Default value specified
+		self.T_max 				= 0.01	# <- Default value specified
+		self.t 					= 0
+		self.timeStepIndex 		= 0
+
+
+		self.integrationMethod	= 'trapezoidal'	# <- Default value specified
+
+
+		# Total number of neurons in the network:
+		self.N 					= 0		#Initialized to 0, incremented within the addNeurons() method when neurons are added.
+
+
+		#############################
+		# NEURON PARAMETER VECTORS: #
+		#############################
+		# The following attributes are vectors that hold a list of the designated parameter values for all neurons in the network.
+		# - The ith element in the vector is the value of that parameter for the ith neuron in the network.
+
+		#--------------------------------
+		# Voltage & Related parameters: -
+		#--------------------------------
+		self.V 				= numpy.empty(shape=[0])
+
+		self.V_init 		= numpy.empty(shape=[0])
+		self.V_r 			= numpy.empty(shape=[0])
+		self.V_t 	 		= numpy.empty(shape=[0])
+		self.V_peak 		= numpy.empty(shape=[0])
+		self.V_reset 		= numpy.empty(shape=[0])
+		self.V_eqExcit 		= numpy.empty(shape=[0])
+		self.V_eqInhib 		= numpy.empty(shape=[0])
+
+		self.U 				= numpy.empty(shape=[0])
+		self.U_init			= numpy.empty(shape=[0])
+
+		self.a 				= numpy.empty(shape=[0])
+		self.b 				= numpy.empty(shape=[0])
+		self.d 				= numpy.empty(shape=[0])
+
+		self.C_membrane		= numpy.empty(shape=[0])
+		self.k 				= numpy.empty(shape=[0])
+
+		#----------------
+		# Conductances: -
+		#----------------
+		self.g_excit		= numpy.empty(shape=[0])
+		self.g_inhib 		= numpy.empty(shape=[0])
+		self.g_gap 			= numpy.empty(shape=[0])
+
+		#------------------
+		# Time constants: -
+		#------------------
+		self.tau_g_excit 	= numpy.empty(shape=[0])
+		self.tau_g_inhib 	= numpy.empty(shape=[0])
+
+		#-------------------------
+		# Integration constants: -
+		#-------------------------
+		# Terms in the dynamics update rule of a parameter that consist of only constant parameters
+		# and can be pre-calculated at initialization to save computation time in the simulation loop.
+		# For LIF, different integration methods can be put in same form differing only by these constant terms,
+		# which can be pre-calculated according to a specified integration method allowing for a common update expression in the sim loop.
+		# (Initialized to None to raise error if used without being explicitly computed)
+		self.constAlpha_g_excit	= None
+		self.constBeta_g_excit	= None
+		self.constAlpha_g_inhib	= None
+		self.constBeta_g_inhib	= None
+
+		#-------------------------
+		# Connectivity Matrices: -
+		#-------------------------
+		# Instantiating connectivity matrices as empty 2D arrays with 0 rows and 0 cols.
+		# Connectivity matrices will have rows and cols added as neurons and/or inputs are added to the network.
+		self.ConnectionWeights_synExcit	= numpy.empty(shape=[0, 0])
+		self.ConnectionWeights_synInhib	= numpy.empty(shape=[0, 0])
+		self.ConnectionWeights_gap		= numpy.empty(shape=[0, 0])
+		self.ConnectionWeights_inpExcit	= numpy.empty(shape=[0, 0])
+		self.ConnectionWeights_inpInhib	= numpy.empty(shape=[0, 0])
+
+		#---------------------------
+		# The following list of dicts stores variousinformation for each neuron in the network.
+		# Typical fields stored for each neuron include:
+		#	- id:		A unique integer ID number assigned to each neuron added to the network;
+		#				the ID number i denotes that neuron as the i-th neuron to be added to the network
+		#	- label:	A string label can optionally be given to any neuron for custom identification/filtering of neurons
+		#	- synapse_type:	A string denoting if the neuron is 'excitatory', 'inhibitory', or some other related description (eg type of neurotransmitter)
+		#		^^^ maybe handle this by a 0-1 vector or some other way not in this dict
+		#	- position:	A tuple giving the 2D or 3D coordinates of the neuron relative to some physical network topography defined elsewhere
+		#	- dynamics:	A string denoting the neuron dynamics model to be used for simulating this neuron
+		#---------------------------
+		self.neuronDescriptors	= []
+
+		self.neuronIDs 			= numpy.empty(shape=[0])
+		self.neuronSynapseTypes	= numpy.empty(shape=[0])
+		self.neuronLabels		= numpy.empty(shape=[0])
+
+
+		# self.neuronPositions	=
+
+		# self.neuronLabels		=
+
+		#----------------------------
+		#############################
+
+		self.spikeEvents		= numpy.empty(shape=[0])
+
+		self.inputValues_excit	= numpy.empty(shape=[0])
+		self.inputValues_inhib	= numpy.empty(shape=[0])
+
+
+
+		self.logs_spikeEvents			= None
+		self.logs_voltage 				= None
+		self.logs_recoveryVar			= None
+		self.logs_g_excit				= None
+		self.logs_g_inhib				= None
+		self.logs_g_gap					= None
+		self.logs_inputValues_excit		= None
+		self.logs_inputValues_inhib		= None
+
+
+		self.enableLog_spikeEvents		= True
+		self.enableLog_voltage			= True
+		self.enableLog_recoveryVar		= True
+		self.enableLog_g_leak			= True
+		self.enableLog_g_excit			= True
+		self.enableLog_g_inhib			= True
+		self.enableLog_g_gap			= True
+		self.enableLog_inputValues_excit= True
+		self.enableLog_inputValues_inhib= True
+
+		#~~~~~~~~~~~~~~~~~~~
+		# Network Geometry ~
+		#~~~~~~~~~~~~~~~~~~~
+		self.geometry 	= None
+
+
+
+		#~~~~~~~~~~~~~~~~~~~~~~~~
+		# Initialization Flags: ~
+		#~~~~~~~~~~~~~~~~~~~~~~~~
+		self.simulationInitialized	= False
+
+
	def addNeurons(self, numNeuronsToAdd,
					V_init, V_r, V_t, V_peak, V_reset, V_eqExcit, V_eqInhib, U_init,
					a, b, d, C_membrane, k,
					g_excit_init, g_inhib_init, g_gap, tau_g_excit, tau_g_inhib,
					synapse_type, label='', dynamics='Izhikevich', position=None):

		#--------------------
		# Add the given parameters for this set of neuron(s) to the network's neuron parameter vectors:
		#--------------------
		self.V 				= numpy.concatenate([self.V, [V_init for n in range(numNeuronsToAdd)]])
		self.V_init 		= numpy.concatenate([self.V_init, [V_init for n in range(numNeuronsToAdd)]])
		self.V_r 			= numpy.concatenate([self.V_r, [V_r for n in range(numNeuronsToAdd)]])
		self.V_t 			= numpy.concatenate([self.V_t, [V_t for n in range(numNeuronsToAdd)]])
		self.V_peak 		= numpy.concatenate([self.V_peak, [V_peak for n in range(numNeuronsToAdd)]])
		self.V_reset 		= numpy.concatenate([self.V_reset, [V_reset for n in range(numNeuronsToAdd)]])
		self.V_eqExcit 		= numpy.concatenate([self.V_eqExcit, [V_eqExcit for n in range(numNeuronsToAdd)]])
		self.V_eqInhib 		= numpy.concatenate([self.V_eqInhib, [V_eqInhib for n in range(numNeuronsToAdd)]])

		self.U 				= numpy.concatenate([self.U, [U_init for n in range(numNeuronsToAdd)]])
		self.U_init 		= numpy.concatenate([self.U_init, [U_init for n in range(numNeuronsToAdd)]])

		self.a 				= numpy.concatenate([self.a, [a for n in range(numNeuronsToAdd)]])
		self.b 				= numpy.concatenate([self.b, [b for n in range(numNeuronsToAdd)]])
		self.d 				= numpy.concatenate([self.d, [d for n in range(numNeuronsToAdd)]])
		self.C_membrane		= numpy.concatenate([self.C_membrane, [C_membrane for n in range(numNeuronsToAdd)]])
		self.k 				= numpy.concatenate([self.k, [k for n in range(numNeuronsToAdd)]])

		self.g_excit 		= numpy.concatenate([self.g_excit, [g_excit_init for n in range(numNeuronsToAdd)]])
		self.g_inhib 		= numpy.concatenate([self.g_inhib, [g_inhib_init for n in range(numNeuronsToAdd)]])
		self.g_gap			= numpy.concatenate([self.g_gap, [g_gap for n in range(numNeuronsToAdd)]])

		self.tau_g_excit 	= numpy.concatenate([self.tau_g_excit, [tau_g_excit for n in range(numNeuronsToAdd)]])
		self.tau_g_inhib 	= numpy.concatenate([self.tau_g_inhib, [tau_g_inhib for n in range(numNeuronsToAdd)]])

		self.neuronIDs			= numpy.concatenate([self.neuronIDs, [self.N+n for n in range(numNeuronsToAdd)]])
		self.neuronSynapseTypes	= numpy.concatenate([self.neuronSynapseTypes, [synapse_type for n in range(numNeuronsToAdd)]])
		self.neuronLabels		= numpy.concatenate([self.neuronLabels, [label for n in range(numNeuronsToAdd)]])

		self.neuronDescriptors 	= (self.neuronDescriptors
									+ [{
											'id':			self.N + n,
											'label':		label,
											'synapse_type':	synapse_type,
											'dynamics':		dynamics,
										} for n in range(numNeuronsToAdd)])

		self.spikeEvents	= numpy.concatenate([self.spikeEvents, [0 for n in range(numNeuronsToAdd)]])

		#--------------------
		# Increment the count of total neurons in the network:
		#--------------------
		self.N 	+= numNeuronsToAdd

		#--------------------
		# Expand the connectivity matrices to accomodate the added nuerons.
		# (initialize all connections for newly added neurons to 0 weight)
		#--------------------

		Wts_temp	= numpy.zeros(shape=(self.N, self.N))
		Wts_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.ConnectionWeights_synExcit
		self.ConnectionWeights_synExcit = Wts_temp

		Wts_temp	= numpy.zeros(shape=(self.N, self.N))
		Wts_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.ConnectionWeights_synInhib
		self.ConnectionWeights_synInhib = Wts_temp

		Wts_temp	= numpy.zeros(shape=(self.N, self.N))
		Wts_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.ConnectionWeights_gap
		self.ConnectionWeights_gap = Wts_temp

		#--------------------
		# If the simulation has already been initialized (i.e. variable logs already initialized,
		# such as when adding neuron(s) in middle of simulation), add list(s) to the logs for the neuron(s) being added.
		# If the simulation hasn't yet been initialized, logs will be allocated for all previously added neurons at that time.
		#--------------------
		if(self.simulationInitialized):
			self.logs_spikeEvents	+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]
			self.logs_voltage 		+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]
			self.logs_recoveryVar	+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]
			self.logs_g_leak		+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]
			self.logs_g_excit		+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]
			self.logs_g_inhib		+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]
			self.logs_g_gap			+= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]


		#--------------------
		# If the network has an established geometry, add these new neurons to that geometry:
		#--------------------
		if(self.geometry is not None):
			self.geometry.addNeurons(numNeuronsToAdd)


		return


	def setNetworkGeometry(self, geometry, dimensions):
		self.geometry 	= NetworkGeometry(geometry, dimensions)
		return

	def positionNeurons(self, coordinates, interval=None, neuron_ids=None):
		self.geometry.positionNeurons(coordinates, interval, neuron_ids)
		return


	def numInputs_excit(self):
		return self.ConnectionWeights_inpExcit.shape[0]

	def numInputs_inhib(self):
		return self.ConnectionWeights_inpInhib.shape[0]


	def setSynapticConnectivity(self, connectivity, update_neurons=None, synapse_type='excitatory'):

		#  TODO check if any neurons have both excitatory and inhibitory outgoing connections. print a warning if so, but don't change anything. leave up to higher level scripts to make sure their construction of network doesn't have neurons that are both, unless that's what they want

		# Check that the weight matrix provided is in the correct form, and convert the matrix to a numpy.ndarray type:
		try:
			synapticConnectivityMatrix	= connectivity
			if(not isinstance(synapticConnectivityMatrix, numpy.ndarray)):
				synapticConnectivityMatrix	= numpy.atleast_2d(synapticConnectivityMatrix)
			if(synapticConnectivityMatrix.ndim != 2):
				raise ValueError
		except ValueError:
			print("LIFNetwork Error: The method setSynapticConnectivity expects args['W'] to be a 2D numeric weight matrix as its argument (list-of-lists or numpy.ndarray)")
			exit()
		except KeyError:
			print("LIFNetwork Error: The method setSynapticConnectivity expects the argument 'args' to be a dictionary containing the key-value pair 'W':(2D numeric weight matrix (list-of-lists or numpy.ndarray))")
			exit()

		if(update_neurons == None):
			#--------------------
			# When a list of neurons to update is not given, the connectivity for the entire network is updated.
			#	- 'connectivity' argument is expected to be an NxN matrix.
			#--------------------
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(synapticConnectivityMatrix.shape[0] != self.N or synapticConnectivityMatrix.shape[1] != self.N):
				print("LIFNetwork Error: The method setSynapticConnectivity expects NxN weight matrix to be given as the argument args['W'], where N = num neurons in network. The given matrix has dimensions "+str(synapticConnectivityMatrix.shape[1])+"x"+str(synapticConnectivityMatrix.shape[0])+", but the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the network's ConnectWeights_syn[Excit/Inhib] to the given connectivity matrix, according to the input type (excitatory/inhibitory).
			#--------------------
			if(synapse_type.lower() == 'excitatory' or synapse_type.lower() == 'e'):
				self.ConnectionWeights_synExcit 	= synapticConnectivityMatrix
			elif(synapse_type.lower() == 'inhibitory' or synapse_type.lower() == 'i'):
				self.ConnectionWeights_synInhib		= synapticConnectivityMatrix
			else:
				print("LIFNetwork Error: The method setInputConnectivity expects input_type to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
				exit()

		else:
			#--------------------
			# When a list of neurons to update is given, the connectivity for only those neurons are updated.
			#	- 'update_neurons' is expected to be a list of integer neuron IDs
			#	- 'connectivity' argument  is expected to be an nxN matrix, where n=len(update_neurons),
			#		and the rows of this matrix are expected to refer to network neurons in the order given by update_neurons
			#--------------------
			# Check that the update_neurons argument is a list of integers in the range 0-N with no duplicates:
			try:
				if(len(update_neurons) > self.N or not all(isinstance(i, int) for i in update_neurons) or len(set(update_neurons)) != len(update_neurons)):
					print("LIFNetwork Error: The method setSynapticConnectivity expects the optional argument update_neurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
					exit()
			except TypeError:
				print("LIFNetwork Error: The method setSynapticConnectivity expects the optional argument update_neurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
				exit()
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(synapticConnectivityMatrix.shape[0] != len(update_neurons) or synapticConnectivityMatrix.shape[1] != self.N):
				print("LIFNetwork Error: The method setSynapticConnectivity expects mxN weight matrix to be given as the argument args['W'], where m = num neurons designated for update in the update_neurons argument, and N = num neurons in network. The given matrix has dimensions "+str(synapticConnectivityMatrix.shape[0])+"x"+str(synapticConnectivityMatrix.shape[1])+", but the num neurons to update is " +str(len(update_neurons))+ " and the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the rows of the the network's ConnectWeights_syn[Excit/Inhib] matrix corresponding to the neurons
			# to the given connectivity values, according to the input type (excitatory/inhibitory).
			#--------------------
			if(synapse_type.lower() == 'excitatory' or synapse_type.lower() == 'e'):
				for i in range(len(update_neurons)):
					updateNeuronID 	= update_neurons[i]
					self.ConnectionWeights_synExcit[updateNeuronID]	= synapticConnectivityMatrix[i]
			elif(synapse_type.lower() == 'inhibitory' or synapse_type.lower() == 'i'):
				for i in range(len(update_neurons)):
					updateNeuronID 	= update_neurons[i]
					self.ConnectionWeights_synInhib[updateNeuronID]	= synapticConnectivityMatrix[i]
			else:
				print("LIFNetwork Error: The method setInputConnectivity expects input_type to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
				exit()

		return

	def setGapJunctionConnectivity(self, connectivity, update_neurons=None):
		# Check that the weight matrix provided is in the correct form, and convert the matrix to a numpy.ndarray type:
		try:
			gapConnectivityMatrix	= connectivity
			if(not isinstance(gapConnectivityMatrix, numpy.ndarray)):
				gapConnectivityMatrix	= numpy.atleast_2d(gapConnectivityMatrix)
			if(gapConnectivityMatrix.ndim != 2):
				raise ValueError
		except ValueError:
			print("LIFNetwork Error: The method setGapJunctionConnectivity expects args['W'] to be a 2D numeric weight matrix as its argument (list-of-lists or numpy.ndarray)")
			exit()
		except KeyError:
			print("LIFNetwork Error: The method setGapJunctionConnectivity expects the argument 'args' to be a dictionary containing the key-value pair 'W':(2D numeric weight matrix (list-of-lists or numpy.ndarray))")
			exit()

		if(update_neurons == None):
			#--------------------
			# When a list of neurons to update is not given, the connectivity for the entire network is updated.
			#	- 'connectivity' argument is expected to be an NxN matrix.
			#--------------------
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(gapConnectivityMatrix.shape[0] != self.N or gapConnectivityMatrix.shape[1] != self.N):
				print("LIFNetwork Error: The method setGapJunctionConnectivity expects NxN weight matrix to be given as the argument args['W'], where N = num neurons in network. The given matrix has dimensions "+str(gapConnectivityMatrix.shape[1])+"x"+str(gapConnectivityMatrix.shape[0])+", but the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the network's ConnectWeights_gap to the given connectivity matrix, according to the input type (excitatory/inhibitory).
			#--------------------
			self.ConnectionWeights_gap 	= gapConnectivityMatrix

		else:
			#--------------------
			# When a list of neurons to update is given, the connectivity for only those neurons are updated.
			#	- 'update_neurons' is expected to be a list of integer neuron IDs
			#	- 'connectivity' argument  is expected to be an nxN matrix, where n=len(update_neurons),
			#		and the rows of this matrix are expected to refer to network neurons in the order given by update_neurons
			#--------------------
			# Check that the update_neurons argument is a list of integers in the range 0-N with no duplicates:
			try:
				if(len(update_neurons) > self.N or not all(isinstance(i, int) for i in update_neurons) or len(set(update_neurons)) != len(update_neurons)):
					print("LIFNetwork Error: The method setGapJunctionConnectivity expects the optional argument update_neurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
					exit()
			except TypeError:
				print("LIFNetwork Error: The method setGapJunctionConnectivity expects the optional argument update_neurons, when provided, to be a list of integer neuron ID numbers with no duplicate IDs")
				exit()
			# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
			if(gapConnectivityMatrix.shape[0] != len(update_neurons) or gapConnectivityMatrix.shape[1] != self.N):
				print("LIFNetwork Error: The method setGapJunctionConnectivity expects mxN weight matrix to be given as the argument args['W'], where m = num neurons designated for update in the update_neurons argument, and N = num neurons in network. The given matrix has dimensions "+str(gapConnectivityMatrix.shape[0])+"x"+str(gapConnectivityMatrix.shape[1])+", but the num neurons to update is " +str(len(update_neurons))+ " and the current num neurons is "+str(self.N))
				exit()

			#--------------------
			# Set the rows of the the network's ConnectWeights_gap matrix corresponding to the neurons
			# to the given connectivity values, according to the input type (excitatory/inhibitory).
			#--------------------
			self.ConnectionWeights_gap 	= gapConnectivityMatrix

		return

	def setInputConnectivity(self, connectivity, input_type='excitatory'):
		#--------------------
		# This method expects as input a IxN matrix representing the weighted mapping of I input signals to N network neurons.
		#--------------------

		inputConnectivityMatrix	= connectivity
		# Check that the weight matrix provided is in the correct form, and convert the matrix to a numpy.ndarray type:
		try:
			if(not isinstance(inputConnectivityMatrix, numpy.ndarray)):
				inputConnectivityMatrix	= numpy.array(inputConnectivityMatrix)
			if(inputConnectivityMatrix.ndim != 2):
				raise ValueError
		except:
			print("LIFNetwork Error: The method setInputConnectivity expects 2D numeric weight matrix as its argument (list-of-lists or numpy.ndarray)")
			exit()

		# Check that the dimension of the weight matrix provided matches the number of neurons in the network:
		if(inputConnectivityMatrix.shape[1] != self.N):
			print("LIFNetwork Error: The method setInputConnectivity expects IxN weight matrix as its argument, where N = num neurons in network. The given matrix has dimensions I="+str(inputConnectivityMatrix.shape[1])+"xN="+str(inputConnectivityMatrix.shape[0])+", but the current num neurons is "+str(self.N))
			exit()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Set the network's ConnectWeights_inpExcit or ConnectWeights_inp to the given connectivity matrix,
		# according to the input type (excitatory/inhibitory).
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if(input_type.lower() == 'excitatory' or input_type.lower() == 'e'):
			self.ConnectionWeights_inpExcit 	= inputConnectivityMatrix
		elif(input_type.lower() == 'inhibitory' or input_type.lower() == 'i'):
			self.ConnectionWeights_inpInhib		= inputConnectivityMatrix
		else:
			print("LIFNetwork Error: The method setInputConnectivity expects input_type to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
			exit()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# (Re)initialize the inputValues vectors to have lengths matching the newly specified number of inputs
		# (if the number of inputs isn't changed by the new connectivity, inputValues are reset to 0 anyways)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self.inputValues_excit	= numpy.zeros(self.numInputs_excit())
		self.inputValues_inhib	= numpy.zeros(self.numInputs_inhib())

		return


	def setInputValues(self, input_values, input_type):
		#--------------------
		# This method expects as input array with length I representing the values that will be mapped as inputs to neurons
		# according to the ConnectionWeights_inp[Excit/Inhib] connectivity matrix set in the setInputConnectivity method.
		#--------------------

		# Check that the array of input values provided is in the correct form, and convert the array to a numpy.ndarray type:
		try:
			if(not isinstance(input_values, numpy.ndarray)):
				input_values	= numpy.array(input_values)
			if(input_values.ndim != 1):
				raise ValueError
		except:
			print("LIFNetwork Error: The method setInputValues expects 1D array as its argument (list or numpy.ndarray)")
			exit()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Set the network's current input values to the given values, according to the input type (excitatory/inhibitory).
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if(input_type.lower() == 'excitatory' or input_type.lower() == 'e'):
			# Check that the dimension of the input values array matches the number of excitatory inputs to the network:
			if(input_values.shape[0] != self.numInputs_excit()):
				print("LIFNetwork Error: The method setInputValues expects array with length I=numInputs array as its argument. The given array has dimensions I="+str(input_values.shape[0])+", but the current network is configured for "+str(self.numInputs_excit())+" excitatory inputs. Reconfigure input connectivity by passing a new connectivity matrix to setInputConnectivity before changing number of input values provided.")
				exit()
			self.inputValues_excit 	= input_values

		elif(input_type.lower() == 'inhibitory' or input_type.lower() == 'i'):
			# Check that the dimension of the input values array matches the number of inhibitory inputs to the network:
			if(input_values.shape[0] != self.numInputs_inhib()):
				print("LIFNetwork Error: The method setInputValues expects array with length I=numInputs array as its argument. The given array has dimensions I="+str(input_values.shape[0])+", but the current network is configured for "+str(self.numInputs_inhib())+" inhibitory inputs. Reconfigure input connectivity by passing a new connectivity matrix to setInputConnectivity before changing number of input values provided.")
				exit()
			self.inputValues_excit 	= input_values

		else:
			print("LIFNetwork Error: The method setInputValues expects input_type to be specified as ['excitatory'|'e'] or ['inhibitory'|'i'])")
			exit()

		return

	def logCurrentVariableValues(self):
		#~~~~~~~~~~~~~~~~~~~~
		# Log the current values of variables for which logging is enabled:
		#~~~~~~~~~~~~~~~~~~~~
		for n in range(self.N):
			if(self.enableLog_spikeEvents):
				self.logs_spikeEvents[n][self.timeStepIndex]	= self.spikeEvents[n]
			if(self.enableLog_voltage):
				self.logs_voltage[n][self.timeStepIndex]		= self.V[n]
			if(self.enableLog_recoveryVar):
				self.logs_recoveryVar[n][self.timeStepIndex]	= self.U[n]
			if(self.enableLog_g_excit):
				self.logs_g_excit[n][self.timeStepIndex]		= self.g_excit[n]
			if(self.enableLog_g_inhib):
				self.logs_g_inhib[n][self.timeStepIndex]		= self.g_inhib[n]
			if(self.enableLog_g_gap):
				self.logs_g_gap[n][self.timeStepIndex]			= self.g_gap[n]

		if(self.enableLog_inputValues_excit):
			for i in range(self.numInputs_excit()):
				self.logs_inputValues_excit[i][self.timeStepIndex]	= self.inputValues_excit[i]
		if(self.enableLog_inputValues_inhib):
			for i in range(self.numInputs_inhib()):
				self.logs_inputValues_inhib[i][self.timeStepIndex]	= self.inputValues_inhib[i]
		return


	def initializeSimulation(self, T_max=None, deltaT=None, integrationMethod=None):
		# print "initializeSimulation()"

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

		self.logs_spikeEvents	= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]
		self.logs_voltage 		= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]
		self.logs_recoveryVar 	= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]
		self.logs_g_excit		= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]
		self.logs_g_inhib		= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]
		self.logs_g_gap			= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.N)]

		self.logs_inputValues_excit	= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.numInputs_excit())]
		self.logs_inputValues_inhib	= [[numpy.nan for t in range(self.numTimeSteps)] for n in range(self.numInputs_inhib())]

		# if(not self.neuronsPositioned):

		# if(not self.connectivityConfigured):

		#~~~~~~~~~~~~~~~~~~~~
		# Pre-calculate numerical integration constants according to the given integration method:
		#~~~~~~~~~~~~~~~~~~~~
		self.integrationMethod	= self.integrationMethod.lower()

		if(self.integrationMethod == 'euler' or self.integrationMethod == 'forwardeuler' or self.integrationMethod == 'rk1'):
			# print('+++ INTEGRATION <eul> +++')
			self.constAlpha_g_excit	= 1.0 - self.deltaT/self.tau_g_excit
			self.constBeta_g_excit	= self.deltaT/self.tau_g_excit
			self.constAlpha_g_inhib	= 1.0 - self.deltaT/self.tau_g_inhib
			self.constBeta_g_inhib	= self.deltaT/self.tau_g_inhib


		elif(self.integrationMethod == 'trapezoid' or self.integrationMethod == 'trapezoidal'):
			print('+++ INTEGRATION <trp> +++')

		elif(self.integrationMethod == 'rk2'):
			print('+++ INTEGRATION <rk2> +++')

		elif(self.integrationMethod == 'rk4'):
			print('+++ INTEGRATION <rk4> +++')

		else:
			print("The given integration method, \'"+self.integrationMethod+"\' is not recognized. Forward euler integration will be used by default.")

		#~~~~~~~~~~~~~~~~~~~~
		# Log the current values of variables for which logging is enabled:
		#~~~~~~~~~~~~~~~~~~~~
		# self.logCurrentVariableValues()



		# The simulation is now flagged as initialized:
		self.simulationInitialized	= True

		return

	def diracDeltaValue(self):
		return 1.0/self.deltaT

	def simStep(self):

		# print "+++++++++++++++++++++++++++++"
		# print "+ simStep t="+str(self.t)+"  tIndex="+str(self.timeStepIndex)+" +"
		# print "+++++++++++++++++++++++++++++"

		#--------------------
		# If the network has not yet been initialized, call initializeSimulation() now.
		# 	Sim parameters will be initialized to whatever values the network attributes (eg delta, T_max, etc) currently hold;
		#	(perhaps the default values provided to these attributes in the constructor)
		# Else, continue with the simulation step under the existing initialization.
		#--------------------
		if(not self.simulationInitialized):
			self.initializeSimulation()

		# Check that we are not going past the time T_max allocated in initializeSimulation:
		if(self.t >= self.T_max):
			print("LIFError: The maximum simulation time has been reached, this simStep call will not be executed. [t = "+str(self.t)+", T_max = "+str(self.T_max)+", deltaT = "+str(self.deltaT)+", max num timesteps = "+str(self.T_max/self.deltaT)+", current timestep num = " +str(self.timeStepIndex))
			return

		#**********************
		# Update Conductances *
		#**********************
		#----------------------
		# No need to have if statements for integration method because we're putting all update rules in the same form (g(t+1) = alpha*g(t) + beta*Ws)
		# where the only difference between the integration methods are the values of constants alpha and beta,
		# which are pre-calculated at network initialization according to integration method

		# print "ConnectionWeights_synExcit\n" + str(self.ConnectionWeights_synExcit)
		# print "spikeEvents\n" + str(self.spikeEvents)
		# print "diracDeltaValue\n" + str(self.diracDeltaValue())
		# print "spikeEvents*diracDeltaValue\n" + str(self.spikeEvents*self.diracDeltaValue())
		# print "ConnectionWeights_synExcit.dot(spikeEvents*diracDeltaValue)\n" + str(self.ConnectionWeights_synExcit.T.dot(self.spikeEvents*self.diracDeltaValue()))

		# print "**********"
		# print "ConnectionWeights_inpExcit\n" + str(self.ConnectionWeights_inpExcit)
		# print "inputValues_excit\n" + str(self.inputValues_excit)
		# print "ConnectionWeights_inpExcit.dot(inputValues_excit)\n" + str(self.ConnectionWeights_inpExcit.T.dot(self.inputValues_excit))

		# print self.ConnectionWeights_inpExcit.shape
		# print self.ConnectionWeights_inpExcit.ndim

		synpaseInducedConductanceChange_excit 	= self.ConnectionWeights_synExcit.T.dot(self.spikeEvents*self.diracDeltaValue())
		synpaseInducedConductanceChange_inhib 	= self.ConnectionWeights_synInhib.T.dot(self.spikeEvents*self.diracDeltaValue())
		inputInducedConductanceChange_excit 	= 0#self.ConnectionWeights_inpExcit.T.dot(self.inputValues_excit) if self.numInputs_excit() > 0 else numpy.zeros(self.N)
		inputInducedConductanceChange_inhib 	= 0#self.ConnectionWeights_inpInhib.T.dot(self.inputValues_inhib) if self.numInputs_inhib() > 0 else numpy.zeros(self.N)

		# print "synpaseInducedConductanceChange_excit\n" + str(synpaseInducedConductanceChange_excit)
		# print "inputInducedConductanceChange_excit\n" +str(inputInducedConductanceChange_excit)

		self.g_excit 	= self.constAlpha_g_excit*self.g_excit + self.constBeta_g_excit*(synpaseInducedConductanceChange_excit + inputInducedConductanceChange_excit)
		self.g_inhib 	= self.constAlpha_g_inhib*self.g_inhib + self.constBeta_g_inhib*(synpaseInducedConductanceChange_inhib + inputInducedConductanceChange_inhib)

		# print "g_excit> " + str(self.g_excit)

		#****************************
		# Update Recovery Variables *
		#****************************
		#----------------------------


		#******************
		# Update Voltages *
		#******************
		#------------------
		# Voltage update rules depend on primarily non-constant variable terms, so there's not really anything to pre-calculate.
		# Therfore, we case the integration method and calculate the updated voltage accordingly every sim step.

		if(self.integrationMethod == 'euler' or self.integrationMethod == 'forwardeuler' or self.integrationMethod == 'rk1'):



			# I_ext_debug	= [120, 0, 0, 0, 0] #0#70.0
			I_ext_debug = [0 * self.N]
			I_ext_debug[0]=120
			dVdt	= (1/self.C_membrane)*(self.k*(self.V - self.V_r)*(self.V - self.V_t) - self.U
						+ (self.g_excit*(self.V_eqExcit - self.V) + self.g_inhib*(self.V_eqInhib - self.V))
						+ (self.ConnectionWeights_inpExcit.T.dot(self.inputValues_excit) if self.numInputs_excit() > 0 else numpy.zeros(self.N)) )
			self.V 	= self.V + self.deltaT*dVdt

			dUdt	= self.a*( self.b*(self.V - self.V_r) - self.U)
			self.U 	= self.U + self.deltaT*dUdt



		elif(self.integrationMethod == 'trapezoid' or self.integrationMethod == 'trapezoidal'):
			print('+++ INTEGRATION <trp> +++')

		elif(self.integrationMethod == 'rk2'):
			print('+++ INTEGRATION <rk2> +++')

		elif(self.integrationMethod == 'rk4'):
			print('+++ INTEGRATION <rk4> +++')

		else:
			print("LIFError: Unrecognized integration method, \'"+self.integrationMethod+"\', referenced in simStep().")
			exit()


		# print "V      > " + str(self.V)


		for n in range(self.N):

			#**********************
			# Update Spike Events *
			#**********************
			#----------------------
			# Record which neurons have reached peak voltage and have thus spiked:
			if(self.V[n] >= self.V_peak[n]):
				# Reset neurons that have spiked to their reset voltage:
				self.spikeEvents[n]		= 1
				self.V[n]				= self.V_reset[n]
				self.U[n]				= self.U[n] + self.d[n]
				# print str(n) + " spike @ " +str(self.t)
			else:
				self.spikeEvents[n]		= 0

			# if(self.V[0] >= self.V_t[0] and self.V[0] <= self.V_t[0]+0.01):
			# 	print "0 V_t @ " +str(self.t)
			# if(self.V[0] >= self.V_r[0] and self.V[0] <= self.V_r[0]+0.01):
			# 	print "0 V_r @ " +str(self.t)
			# if(self.V[0] >= self.V_reset[0] and self.V[0] <= self.V_reset[0]+0.01):
			# 	print "0 V_reset @ " +str(self.t)

			# if(self.V[1] >= self.V_t[1] and self.V[1] <= self.V_t[1]+0.0):
			# 	print "1 V_t @ " +str(self.t)
			# if(not self.V[1] <= self.V_r[1]+0.01):
			# 	print "1 NOT V_r @ " +str(self.t) + "  " + str(self.V[1])
			# if(self.V[1] > self.V_reset[1] and self.V[1] <= self.V_reset[1]+0.01):
			# 	print "1 V_reset @ " +str(self.t)

		# if(self.V[0] < self.V_reset[0]):
			# print str(self.t) + str(self.V[0]) + str(self.V_reset[0])


		# print "spikes  > " + str(self.spikeEvents)
		# print "V(reset)> " + str(self.V)

		#~~~~~~~~~~~~~~~~~~~~
		# Log the current values of variables for which logging is enabled:
		#~~~~~~~~~~~~~~~~~~~~
		# TODO? (allow user to specify all/none/set of neuron indexes to log for each logable thing?)
		self.logCurrentVariableValues()


		# update voltages

		# update spikeEvents

		#~~~~~~~~~~~~~~~~~~~~
		# Increment t and the timestep index:
		#~~~~~~~~~~~~~~~~~~~~
		self.t 				+= self.deltaT
		self.timeStepIndex 	+= 1






		return


	def runSimulation(self, T_max=None, deltaT=None, integrationMethod=None):
		# convenience method that just loops simSteps.
		# higher level script can also use their own loop/logic to run simulation and change external inputs, network surface geometry, connectivity, add/remove neurons during its sim loop

		#--------------------
		# If the network has not yet been initialized, initialize the network with the time parameters given in this runSimulation() call;
		# 	- default values will be used for other initialization parameters (and the time parameters if no values were given to runSimulation()).
		# Else, continue with the current network initialization/parameterization.
		#--------------------
		if(not self.simulationInitialized):
			self.initializeSimulation(T_max, deltaT, integrationMethod)

		while(self.t < (self.T_max-(self.deltaT/2))):	# The right-hand-side of this conditional is what it is rather than just T_max to avoid numerical roundoff errors causing unexpected conditional outcomes
			self.simStep()

		# plot variables that are being plotted (allow user to specify all/none/set of neuron indexes to log for each logable thing)
		# 	Leave plotting to higher level scripts that instantiate this network and can access its log objects?
		return

	def getNeuronIDs(self, synapse_types=None, labels=None):
		if(synapse_types is None):
			synapse_types	= numpy.unique(self.neuronSynapseTypes)
		if(labels is None):
			labels 			= numpy.unique(self.neuronLabels)

		indices_selectedSynTypes	= numpy.in1d(self.neuronSynapseTypes, synapse_types)
		indices_selectedLabels		= numpy.in1d(self.neuronLabels, labels)

		return numpy.where(indices_selectedSynTypes * indices_selectedLabels)[0]

	# TODO: Change the individual logs_xxxx variables to a single logs dict with key-values: t:[...], g_leak[...], etc ?
	def getNeuronsDataFrame(self):
		neuronsDataFrame	= pandas.DataFrame(columns=['neuron id', 't', 'g_leak', 'g_excit', 'g_inhib', 'g_gap', 'V', 'spike', 'synapse type', 'label'])
		for n, nID in enumerate(self.neuronIDs):
			neuronDataSeries		= 	[
											('neuron id', [int(nID) for i in range(len(self.timeSeries))]),
											('t', self.timeSeries),
											('g_excit', self.logs_g_excit[n]),
											('g_inhib', self.logs_g_inhib[n]),
											('g_gap', self.logs_g_gap[n]),
											('V', self.logs_voltage[n]),
											('U', self.logs_recoveryVar[n]),
											('spike', self.logs_spikeEvents[n]),
											('synapse type', [self.neuronSynapseTypes[n] for i in range(len(self.timeSeries))]),
											('label', [self.neuronLabels[n] for i in range(len(self.timeSeries))])
										]
			neuronsDataFrame	= pandas.concat([ neuronsDataFrame, pandas.DataFrame.from_items(neuronDataSeries) ])

		return neuronsDataFrame


	def getInputsDataFrame(self):
		inputsDataFrame	= pandas.DataFrame(columns=['input id', 't', 'input value'])

		for iID in range(self.numInputs_excit()):
			inputsDataSeries		= 	[
											('input id', [str(iID)+'e' for i in range(len(self.timeSeries))]),
											('t', self.timeSeries),
											('input value', self.logs_inputValues_excit[iID])
										]
			inputsDataFrame	= pandas.concat([ inputsDataFrame, pandas.DataFrame.from_items(inputsDataSeries) ])

		for iID in range(self.numInputs_inhib()):
			inputsDataSeries		= 	[
											('input id', [str(iID)+'i' for i in range(len(self.timeSeries))]),
											('t', self.timeSeries),
											('input value', self.logs_inputValues_excit[iID])
										]
			inputsDataFrame	= pandas.concat([ inputsDataFrame, pandas.DataFrame.from_items(inputsDataSeries) ])

		return inputsDataFrame


	def getSpikeTimes(self):
		df 	= self.getNeuronsDataFrame()

		spikeTimes	= []
		for n in range(self.N):
			spikeTimes.append( df.loc[((df['neuron id'] == n) & (df['spike'] == 1)), 't'].values )

		return spikeTimes