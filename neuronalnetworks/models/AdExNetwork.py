from __future__ import division

from NeuronNetwork import NeuronNetwork

import numpy as numpy

#!!!!!!!!!!!!!!!!!!!!!!!!#
# REGARDING MODEL UNITS: #
#!!!!!!!!!!!!!!!!!!!!!!!!#
# The model equations as implemented seem to assume that all parameter units have the same metric prefix.
# Thus, for some parameters, the units that are assumed in the model equations
# differ from the units used for specifying parameter values when adding neurons.
# - The units assumed in both cases are specified for parameters here:
#				Eq'n Units:		Value Input Units:
#	V			mV				mV
#	V_init		mV				mV
#	V_thresh	mV				mV
#	V_peak		mV				mV
#	V_reset		mV				mV
#	V_eqLeak	mV				mV
#	V_eqExcit	mV				mV
#	V_eqInhib	mV				mV
#	slopeFactor	mV				mV
#	C_membrane	Ohm				MegaOhm
#	U 			mA 				nA
#	U_init		mA 				nA
#	a 			mS 				nS
#	b 			mA 				nA
# 	g_leak		mS 				nS
# 	g_excit 	mS 				nS
# 	g_inhib 	mS 				nS
#	g_gap 		mS 				nS
# 	tau_g_U 	ms 				ms
# 	tau_g_excit ms 				ms
# 	tau_g_inhib ms 				ms



class AdExNetwork(NeuronNetwork):

	def __init__(self, calculateCurrents=True):

		NeuronNetwork.__init__(self)

		#################################
		# LIF NEURON PARAMETER VECTORS: #
		#################################
		# The following attributes are vectors that hold a list of the designated parameter values for all neurons in the network.
		# - The ith element in the vector is the value of that parameter for the ith neuron in the network.

		#--------------------------------
		# Voltage & Related parameters: -
		#--------------------------------
		self.V 				= numpy.empty(shape=[0])	

		self.V_init 		= numpy.empty(shape=[0])	
		self.V_thresh 		= numpy.empty(shape=[0])	
		self.V_peak 		= numpy.empty(shape=[0])	
		self.V_reset 		= numpy.empty(shape=[0])	
		self.V_eqLeak 		= numpy.empty(shape=[0])	
		self.V_eqExcit 		= numpy.empty(shape=[0])	
		self.V_eqInhib 		= numpy.empty(shape=[0])	

		self.slopeFactor	= numpy.empty(shape=[0])	

		self.C_membrane		= numpy.empty(shape=[0])	

		#------------------------------------------
		# Adaptation Variable & Related parameters: -
		#------------------------------------------

		self.U 				= numpy.empty(shape=[0])	
		self.U_init			= numpy.empty(shape=[0])	

		self.a 				= numpy.empty(shape=[0])	
		self.b 				= numpy.empty(shape=[0])	

		#----------------
		# Conductances: -
		#----------------
		self.g_leak 		= numpy.empty(shape=[0])	
		self.g_excit		= numpy.empty(shape=[0])	
		self.g_inhib 		= numpy.empty(shape=[0])	
		self.g_gap 			= numpy.empty(shape=[0])	

		#------------
		# Currents: -
		#------------
		self.I_leak 		= numpy.empty(shape=[0])
		self.I_excit		= numpy.empty(shape=[0])
		self.I_inhib 		= numpy.empty(shape=[0])
		self.I_gap 			= numpy.empty(shape=[0])
		self.I_input		= numpy.empty(shape=[0])

		#------------------
		# Time constants: -
		#------------------
		self.tau_g_excit 	= numpy.empty(shape=[0])	
		self.tau_g_inhib 	= numpy.empty(shape=[0])	
		self.tau_U		 	= numpy.empty(shape=[0])	

		#------------------
		# Spiking events: -
		#------------------
		self.spikeEvents		= numpy.empty(shape=[0])

		#-------------------------
		# Integration constants: -
		#-------------------------
		# Terms in the dynamics update rule of a parameter that consist of only constant parameters
		# and can be pre-calculated at initialization to save computation time in the simulation loop.
		# For LIF, different integration methods can be put in same form differing only by these constant terms,
		# which can be pre-calculated according to a specified integration method allowing for a common update expression in the sim loop.
		# (Initialized to None to raise error if used without being explicitly computed)
		self.constAlpha_g_excit	= None
		self.constBeta_g_excit	= None
		self.constAlpha_g_inhib	= None
		self.constBeta_g_inhib	= None


		############################
		# INPUT PARAMETER VECTORS: #
		############################
		self.inputVals 	= numpy.empty(shape=[0])

		#----------------------------
		#############################

		self.neuronLogVariables = ['neuron_id', 't', 'spike', 'V', 'U', 'I_leak', 'I_excit', 'I_inhib', 'I_gap', 'I_input', 'g_leak', 'g_excit', 'g_inhib', 'g_gap', 'synapse_type', 'label']
		for variable in self.neuronLogVariables:
			# The data data structure will be initialized in initialize_simulation() when numTimePoints is known.
			# Enable variable logs by default, except current logs ('I_' variables) which are only enabled by default if currents are to be calculated.
			self.neuronLogs[variable]	= {'data': None, 'enabled': True}

		# print self.neuronLogs['V']
		# print self.neuronLogs['I_leak']
		# exit()

		self.inputLogVariables = ['input_id', 't', 'input_val', 'label']
		for variable in self.inputLogVariables:
			self.inputLogs[variable]	= {'data': None, 'enabled': False}	# The data data structure will be initialized in initialize_simulation() when numTimePoints is known.


	def add_neurons(self, numNeuronsToAdd,
					V_init, V_thresh, V_peak, V_reset, V_eqLeak, V_eqExcit, V_eqInhib, slopeFactor, C_membrane,
					U_init, a, b,
					g_leak, g_excit_init, g_inhib_init, g_gap,
					tau_g_excit, tau_g_inhib, tau_U,
					synapse_type, label=''):

		#--------------------
		# Convert specified parameter units to units used in the model equations, where applicable
		#--------------------
		# a 			= a*1e-
		# b 			= b*1e-6
		# g_leak 		= g_leak*1e-6
		# g_excit_init = g_excit_init*1e-6
		# g_inhib_init = g_inhib_init*1e-6
		# # g_gap 		= g_gap*1e-6
		# C_membrane 	= C_membrane*1e-6


		#--------------------
		# Add the given parameters for this set of neuron(s) to the network's neuron parameter vectors:
		#--------------------
		self.V 				= numpy.concatenate([self.V, [V_init for n in range(numNeuronsToAdd)]])

		self.V_init 		= numpy.concatenate([self.V_init, [V_init for n in range(numNeuronsToAdd)]])
		self.V_thresh 		= numpy.concatenate([self.V_thresh, [V_thresh for n in range(numNeuronsToAdd)]])
		self.V_peak 		= numpy.concatenate([self.V_peak, [V_peak for n in range(numNeuronsToAdd)]])
		self.V_reset 		= numpy.concatenate([self.V_reset, [V_reset for n in range(numNeuronsToAdd)]])
		self.V_eqLeak 		= numpy.concatenate([self.V_eqLeak, [V_eqLeak for n in range(numNeuronsToAdd)]])
		self.V_eqExcit 		= numpy.concatenate([self.V_eqExcit, [V_eqExcit for n in range(numNeuronsToAdd)]])
		self.V_eqInhib 		= numpy.concatenate([self.V_eqInhib, [V_eqInhib for n in range(numNeuronsToAdd)]])

		self.slopeFactor 	= numpy.concatenate([self.slopeFactor, [slopeFactor for n in range(numNeuronsToAdd)]])

		self.U 				= numpy.concatenate([self.U, [U_init for n in range(numNeuronsToAdd)]])
		self.U_init 		= numpy.concatenate([self.U_init, [U_init for n in range(numNeuronsToAdd)]])
		self.a 				= numpy.concatenate([self.a, [a for n in range(numNeuronsToAdd)]])
		self.b 				= numpy.concatenate([self.b, [b for n in range(numNeuronsToAdd)]])

		self.C_membrane		= numpy.concatenate([self.C_membrane, [C_membrane for n in range(numNeuronsToAdd)]])

		self.g_leak 		= numpy.concatenate([self.g_leak, [g_leak for n in range(numNeuronsToAdd)]])
		self.g_excit 		= numpy.concatenate([self.g_excit, [g_excit_init for n in range(numNeuronsToAdd)]])
		self.g_inhib 		= numpy.concatenate([self.g_inhib, [g_inhib_init for n in range(numNeuronsToAdd)]])
		self.g_gap			= numpy.concatenate([self.g_gap, [g_gap for n in range(numNeuronsToAdd)]])

		self.tau_g_excit 	= numpy.concatenate([self.tau_g_excit, [tau_g_excit for n in range(numNeuronsToAdd)]])
		self.tau_g_inhib 	= numpy.concatenate([self.tau_g_inhib, [tau_g_inhib for n in range(numNeuronsToAdd)]])
		self.tau_U 			= numpy.concatenate([self.tau_U, [tau_U for n in range(numNeuronsToAdd)]])

		self.neuronIDs			= numpy.concatenate([self.neuronIDs, [self.N+n for n in range(numNeuronsToAdd)]])
		self.neuronSynapseTypes	= numpy.concatenate([self.neuronSynapseTypes, [synapse_type for n in range(numNeuronsToAdd)]])
		self.neuronLabels		= numpy.concatenate([self.neuronLabels, [label for n in range(numNeuronsToAdd)]])

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
		Wts_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.connectionWeights_synExcit
		self.connectionWeights_synExcit = Wts_temp

		Wts_temp	= numpy.zeros(shape=(self.N, self.N))
		Wts_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.connectionWeights_synInhib
		self.connectionWeights_synInhib = Wts_temp

		Wts_temp	= numpy.zeros(shape=(self.N, self.N))
		Wts_temp[:(self.N-numNeuronsToAdd), :(self.N-numNeuronsToAdd)] = self.connectionWeights_gap
		self.connectionWeights_gap = Wts_temp

		Wts_temp	= numpy.zeros(shape=(self.numInputs, self.N))
		Wts_temp[:(self.numInputs), :(self.N-numNeuronsToAdd)] = self.connectionWeights_inputs
		self.connectionWeights_inputs = Wts_temp

		#--------------------
		# If the simulation has already been initialized (i.e. variable logs already initialized,
		# such as when adding neuron(s) in middle of simulation), add list(s) to the logs for the neuron(s) being added.
		# If the simulation hasn't yet been initialized, logs will be allocated for all previously added neurons at that time.
		#--------------------
		if(self.simulationInitialized):
			for variable in self.neuronLogVariables:
				self.neuronLogs[variable]['data'] += [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numNeuronsToAdd)]

		# print self.neuronLogs['V']
		# print self.neuronLogs['I_leak']
		# exit()

		#--------------------
		# If the network has an established geometry, add these new neurons to that geometry:
		#--------------------
		# TODO: Make sure this function call is justified
		if(self.geometry is not None):
			self.geometry.add_neurons(numNeuronsToAdd)

		return

	def add_inputs(self, numInputsToAdd, label=''):

		#--------------------
		# Add the given metadata for this set of input(s) to the network's input parameter vectors:
		#--------------------
		self.inputVals 		= numpy.concatenate([self.inputVals, [0.0 for n in range(numInputsToAdd)]])

		self.inputIDs		= numpy.concatenate([self.inputIDs, [self.numInputs+i for i in range(numInputsToAdd)]]) # Assign new IDs before incrementing self.numInputs
		self.inputLabels	= numpy.concatenate([self.inputLabels, [label for i in range(numInputsToAdd)]])

		#--------------------
		# Increment the count of total inputs in the network:
		#--------------------
		self.numInputs 	+= numInputsToAdd

		#--------------------
		# Expand the connectivity matrices to accomodate the added nuerons.
		# (initialize all connections for newly added inputs to 0 weight)
		#--------------------
		Wts_temp	= numpy.zeros(shape=(self.numInputs, self.N))
		Wts_temp[:(self.numInputs-numInputsToAdd), :(self.N)] = self.connectionWeights_inputs
		self.connectionWeights_inputs = Wts_temp

		#--------------------
		# If the simulation has already been initialized (i.e. variable logs already initialized,
		# such as when adding neuron(s) in middle of simulation), add list(s) to the logs for the neuron(s) being added.
		# If the simulation hasn't yet been initialized, logs will be allocated for all previously added neurons at that time.
		#--------------------
		if(self.simulationInitialized):
			for variable in self.inputLogVariables:
				self.inputLogs[variable]['data'] += [[numpy.nan for t in range(self.numTimeSteps)] for n in range(numInputsToAdd)]

		return


	def set_input_vals(self, vals, inputIDs=None):
		if(inputIDs is not None):
			self.inputVals = numpy.asarray(vals)
		else:
			self.inputVals[inputIDs] = vals



	def log_current_variable_values(self):
		# return
		#~~~~~~~~~~~~~~~~~~~~
		# Log the current values of variables for which logging is enabled:
		#~~~~~~~~~~~~~~~~~~~~
		# print self.neuronLogs['I_leak']['data'][0][self.timeStepIndex]
		# print self.I_leak
		# print self.neuronIDs
		for n in range(self.N):
			if(self.neuronLogs['neuron_id']['enabled']):
				self.neuronLogs['neuron_id']['data'][n][self.timeStepIndex]	= self.neuronIDs[n]
			if(self.neuronLogs['t']['enabled']):
				self.neuronLogs['t']['data'][n][self.timeStepIndex]	= self.t
			if(self.neuronLogs['spike']['enabled']):
				self.neuronLogs['spike']['data'][n][self.timeStepIndex]	= self.spikeEvents[n]
			if(self.neuronLogs['V']['enabled']):
				self.neuronLogs['V']['data'][n][self.timeStepIndex]	= self.V[n]
			if(self.neuronLogs['U']['enabled']):
				self.neuronLogs['U']['data'][n][self.timeStepIndex]	= self.U[n]
			if(self.neuronLogs['I_leak']['enabled']):
				self.neuronLogs['I_leak']['data'][n][self.timeStepIndex]	= self.I_leak[n]
			if(self.neuronLogs['I_excit']['enabled']):
				self.neuronLogs['I_excit']['data'][n][self.timeStepIndex]	= self.I_excit[n]
			if(self.neuronLogs['I_inhib']['enabled']):
				self.neuronLogs['I_inhib']['data'][n][self.timeStepIndex]	= self.I_inhib[n]
			if(self.neuronLogs['I_gap']['enabled']):
				self.neuronLogs['I_gap']['data'][n][self.timeStepIndex]	= self.I_gap[n]
			if(self.neuronLogs['I_input']['enabled']):
				self.neuronLogs['I_input']['data'][n][self.timeStepIndex]	= self.I_input[n]
			if(self.neuronLogs['g_leak']['enabled']):
				self.neuronLogs['g_leak']['data'][n][self.timeStepIndex]	= self.g_leak[n]
			if(self.neuronLogs['g_excit']['enabled']):
				self.neuronLogs['g_excit']['data'][n][self.timeStepIndex]	= self.g_excit[n]
			if(self.neuronLogs['g_inhib']['enabled']):
				self.neuronLogs['g_inhib']['data'][n][self.timeStepIndex]	= self.g_inhib[n]
			if(self.neuronLogs['g_gap']['enabled']):
				self.neuronLogs['g_gap']['data'][n][self.timeStepIndex]	= self.g_gap[n]
			if(self.neuronLogs['synapse_type']['enabled']):
				self.neuronLogs['synapse_type']['data'][n][self.timeStepIndex]	= self.neuronSynapseTypes[n]
			if(self.neuronLogs['label']['enabled']):
				self.neuronLogs['label']['data'][n][self.timeStepIndex]	= self.neuronLabels[n]

		for i in range(self.numInputs):
			if(self.inputLogs['input_id']['enabled']):
				self.inputLogs['input_id']['data'][i][self.timeStepIndex]	= self.inputIDs[i]
			if(self.inputLogs['t']['enabled']):
				self.inputLogs['t']['data'][i][self.timeStepIndex]	= self.t
			if(self.inputLogs['input_val']['enabled']):
				self.inputLogs['input_val']['data'][i][self.timeStepIndex]	= self.inputVals[i]
			if(self.inputLogs['label']['enabled']):
				self.inputLogs['label']['data'][i][self.timeStepIndex]	= self.inputLabels[i]

		return


	def initialize_simulation(self, T_max=None, deltaT=None, integrationMethod=None):
		# Call the standard network simulation initialization:
		super(self.__class__, self).initialize_simulation(T_max, deltaT, integrationMethod)

		########################################
		# LIFNetwork-specific initializations: #
		########################################
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
			pass
		elif(self.integrationMethod == 'rk2'):
			pass
		elif(self.integrationMethod == 'rk4'):
			pass
		else:
			print("The given integration method, \'"+self.integrationMethod+"\' is not recognized. Forward euler integration will be used by default.")

		return


	def network_update(self):

		#**********************
		# Update Conductances *
		#**********************
		#----------------------
		# No need to have if statements for integration method because we're putting all update rules in the same form (g(t+1) = alpha*g(t) + beta*Ws)
		# where the only difference between the integration methods are the values of constants alpha and beta,
		# which are pre-calculated at network initialization according to integration method

		# print "****************************************************************"
		# print "* update @ t = " +str(self.t)

		synpaseInducedConductanceChange_excit 	= self.connectionWeights_synExcit.T.dot(self.spikeEvents*self.dirac_delta())
		synpaseInducedConductanceChange_inhib 	= self.connectionWeights_synInhib.T.dot(self.spikeEvents*self.dirac_delta())

		self.g_excit 	= self.constAlpha_g_excit*self.g_excit + self.constBeta_g_excit*(synpaseInducedConductanceChange_excit)  # + inputInducedConductanceChange_excit)
		self.g_inhib 	= self.constAlpha_g_inhib*self.g_inhib + self.constBeta_g_inhib*(synpaseInducedConductanceChange_inhib)  # + inputInducedConductanceChange_inhib)

		#******************
		# Update Voltages *
		#******************
		#------------------
		# Voltage update rules depend on primarily non-constant variable terms, so there's not really anything to pre-calculate.
		# Therfore, we case the integration method and calculate the updated voltage accordingly every sim step.
		if(self.integrationMethod == 'euler' or self.integrationMethod == 'forwardeuler' or self.integrationMethod == 'rk1'):
			#---------------------
			# Calculate Currents :
			#---------------------
			self.I_leak		= self.g_leak*(self.V_eqLeak - self.V)
			self.I_excit 	= self.g_excit*(self.V_eqExcit - self.V)
			self.I_inhib 	= self.g_inhib*(self.V_eqInhib - self.V)
			self.I_gap 		= self.g_gap*(self.connectionWeights_gap.T.dot(self.V) - self.connectionWeights_gap.sum(axis=0)*self.V)
			self.I_input	= self.inputVals.dot(self.connectionWeights_inputs)

			# print "R_m     = " + str(self.C_membrane[0])
			# print " "
			# print "V       = " + str(self.V[5])
			# print "V_r     = " + str(self.V_r[0])
			# print "V_t     = " + str(self.V_t[0])
			# print "(V-V_r) = " + str(self.V[0] - self.V_r[0])
			# print "(V-V_t) = " + str(self.V[0] - self.V_t[0])
			# print " "
			# print "U       = " + str(self.U[5])
			# print " "
			# print "I_excit = " + str(self.I_excit[5])
			# print "I_inhib = " + str(self.I_inhib[5])
			# print "I_gap   = " + str(self.I_gap[5])
			# print "I_input = " + str(self.I_input[5])

			dVdt 	= (1/self.C_membrane)*( -1*self.g_leak*(self.V - self.V_eqLeak) + self.g_leak*self.slopeFactor*numpy.exp((self.V - self.V_thresh)/self.slopeFactor) - self.U + (self.I_excit + self.I_inhib + self.I_gap + self.I_input) )
			
			dUdt 	= (1/self.tau_U)*( self.a*(self.V - self.V_eqLeak) - self.U )

			self.V 	= self.V + self.deltaT*dVdt

			
			self.U 	= self.U + self.deltaT*dUdt

			# print " "
			# print "dVdt    = " + str(dVdt[5])
			# print "V'      = " + str(self.V[5])
			# print " "
			# print "dUdt    = " + str(dUdt[5])
			# print "U'      = " + str(self.U[5])

		elif(self.integrationMethod == 'trapezoid' or self.integrationMethod == 'trapezoidal'):
			pass

		elif(self.integrationMethod == 'rk2'):
			pass

		elif(self.integrationMethod == 'rk4'):
			pass

		else:
			print("LIFError: Unrecognized integration method, \'"+self.integrationMethod+"\', referenced in simStep().")
			exit()

		for n in range(self.N):
			#**********************
			# Update Spike Events *
			#**********************
			#----------------------
			# Record which neurons have reached peak voltage and have thus spiked:
			# self.spikeEvents 	= (self.V >= self.V_thresh).astype(int)
			if(self.V[n] >= self.V_peak[n]):
				# Reset neurons that have spiked to their reset voltage:
				self.spikeEvents[n]		= 1
				self.V[n]				= self.V_reset[n]
				self.U[n]				= self.U[n] + self.b[n]
				# print str(n) + " spike @ " +str(self.t)
			else:
				self.spikeEvents[n]		= 0

		return


	def get_spike_times(self):
		spikeTimes = [numpy.nonzero(self.neuronLogs['spike']['data'][nID])[0]*self.deltaT for nID in range(self.N)]
		return spikeTimes