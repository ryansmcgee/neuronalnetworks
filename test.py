from __future__ import division

from neuronalnetworks import *

import numpy as numpy

import time

numpy.random.seed(346911)



#~~~~~~~~~~
# Let's instantiate a network that uses Izhikevich model dynamics:
#~~~~~~~~~~
network = IzhikevichNetwork()
# network = LIFNetwork()

#~~~~~~~~~~
# Let's add a batch of excitatory neurons and a batch of inhibitory neurons to our network:
#~~~~~~~~~~
N_excit = 100    # number of excitatory neurons
N_inhib = 20    # number of inhibitory neurons

network.add_neurons(numNeuronsToAdd=N_excit,
    V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0,
    U_init=0.0, a=0.02, b=0.2, d=8, k=0.7, R_membrane=0.01,
    g_excit_init=0.0, g_inhib_init=0.0, g_gap=1.0, tau_g_excit=2.0, tau_g_inhib=2.0,
    synapse_type='excitatory')

network.add_neurons(numNeuronsToAdd=N_inhib,
    V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0,
    U_init=0.0, a=0.02, b=0.2, d=8, k=0.7, R_membrane=0.01,
    g_excit_init=0.0, g_inhib_init=0.0, g_gap=1.0, tau_g_excit=2.0, tau_g_inhib=2.0,
    synapse_type='inhibitory')

# network.add_neurons(numNeuronsToAdd=N_excit,
# 					V_init=-68.0, V_thresh=-50.0, V_reset=-70.0, V_eqLeak=-68.0, V_eqExcit=0.0, V_eqInhib=-70.0, R_membrane=1.0,
# 					g_leak=0.3, g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5,
# 					tau_g_excit=2.0, tau_g_inhib=2.0, refracPeriod=3.0,
# 					synapse_type='excitatory')

# network.add_neurons(numNeuronsToAdd=N_inhib,
# 					V_init=-68.0, V_thresh=-50.0, V_reset=-70.0, V_eqLeak=-68.0, V_eqExcit=0.0, V_eqInhib=-70.0, R_membrane=1.0,
# 					g_leak=0.3, g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5,
# 					tau_g_excit=2.0, tau_g_inhib=2.0, refracPeriod=3.0,
# 					synapse_type='inhibitory')

#~~~~~~~~~~
# We can get lists of the neuron IDs for the excitatory and inhibitory neurons.
# This is convenient for referencing and handling excitatory and inhibitory neurons seperately later.
#~~~~~~~~~~
#DELE neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
#DELE neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]
neuronIDs_excit	= network.get_neuron_ids(synapseTypes='excitatory')
neuronIDs_inhib	= network.get_neuron_ids(synapseTypes='inhibitory')

#~~~~~~~~~~
# Let's have our network reside on the surface of a cylinder:
#~~~~~~~~~~
network.geometry = CylinderSurface(r=1.5, h=10)

# Now, we add neurons to the geometry instance, since the network's geometry was not instantiated
# until after we added our batches of neurons:
network.geometry.add_neurons(numNeuronsToAdd=N_excit+N_inhib)

#~~~~~~~~~~
# We will position our network's excitatory neurons evenly over the full cylinder surface,
# and position the inhibitory neurons randomly in a limited band around the midline of the cylinder:
#~~~~~~~~~~
network.geometry.position_neurons(positioning='even', neuronIDs=neuronIDs_excit)
network.geometry.position_neurons(positioning='random', bounds={'h':[4,6]}, neuronIDs=neuronIDs_inhib)

# Let's take a look at our network's geometry and neuron positions.
# (A number of network plots are provided, we use one here. Excitatory neurons are colored blue, inhibitory red):
# axsyn3d = pyplot.subplot(projection='3d')
# synapse_network_diagram_3d(axsyn3d, network)
# pyplot.show()

# Distances between neurons in our network were already calculated when the neurons were positioned:
# print network.geometry.distances

#~~~~~~~~~~
# Let's configure our synaptic connectivities such that probability of neuron adjacency decays with their distance.
# Excitatory synaptic connections will have weights that decrease linearly in distance,
# while inhibitory synaptic connections will have weights that decrease exponentially in distance:
#~~~~~~~~~~
W_synE = generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='uniform',
							            args={'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.9,
							                  'low':20.0, 'high':40.0,
							                  'distances':network.geometry.distances[neuronIDs_excit]} )

print numpy.count_nonzero(W_synE[:N_excit,:])
print numpy.count_nonzero(W_synE[:N_excit,:])/N_excit
# exit()

W_synI = generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='distance',
							            args={'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.6,
							                  'init_weight_dist_fn':'exponential', 'p0_w':40, 'sigma_w':5.0,
							                  'distances':network.geometry.distances[neuronIDs_inhib]} )

print numpy.count_nonzero(W_synI[:N_inhib,:])
print numpy.count_nonzero(W_synI[:N_inhib,:])/N_inhib
# exit()

network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)

# Let's take a look at our network's synaptic connectivity:
# axsyn3d = pyplot.subplot(projection='3d')
# synapse_network_diagram_3d(axsyn3d, network)
# pyplot.show()

#~~~~~~~~~~
# We can also configure our network's gap junction connectivity.
# For gap junctions, let's use a nearest-neighbors adjacency scheme.
# But let's add an additional element to this connectivity:
# Let's make the gap junction connectivity increasingly sparse as you move up the length of the cylinder:
#~~~~~~~~~~
W_synG = numpy.zeros(shape=(network.N, network.N))    # starting with empty container for connectivity matrix

for nID in network.get_neuron_ids():
    # Generate and set connectivity vectors one neuron at a time,
    # using their percentile height on the cylinder as a parameter
    # for the additional sparsity applied to the generated vector.
    heightPercentile = network.geometry.cartesianCoords[nID][2]/network.geometry.h
    W_synG[nID] = generate_connectivity_vector(N=network.N, neuronID=nID,
                                               adjacencyScheme='nearest_neighbors', initWeightScheme='constant',
                                               args={'k':4,'c_w':1.0, 'distances':network.geometry.distances[int(nID)]},
                                               sparsity=heightPercentile )

network.set_gapjunction_connectivity(connectivity=W_synG)

# Now let's take a look at our network's gap junction connectivity:
# axgap3d = pyplot.subplot(projection='3d')
# gapjunction_network_diagram_3d(axgap3d, network)
# pyplot.show()

# print network.geometry.cartesianCoords[:,2]

# print numpy.argpartition(network.geometry.cartesianCoords[:,2], -10)[-10:]
# print network.geometry.cartesianCoords[:,2][numpy.argpartition(network.geometry.cartesianCoords[:,2], -10)[-10:]]

neuronIDs_geoTop10 	= numpy.argpartition(network.geometry.cartesianCoords[:,2], -10)[-10:]
neuronIDs_outputs 	= neuronIDs_geoTop10
network.neuronLabels[neuronIDs_outputs] = 'output'

neuronIDs_geoBotHalf = numpy.argpartition(network.geometry.cartesianCoords[:,2], int(network.N/2))[:int(network.N/2)]
print neuronIDs_geoBotHalf
neuronIDs_nonOutputExcit 	= network.get_neuron_ids()[(network.neuronLabels!='output') & (network.neuronSynapseTypes!='inhibitory')]
print neuronIDs_nonOutputExcit
neuronIDs_inputs 	= numpy.random.choice( numpy.intersect1d(neuronIDs_geoBotHalf, neuronIDs_nonOutputExcit), 10, replace=False )
network.neuronLabels[neuronIDs_inputs] = 'input'

print neuronIDs_inputs
print neuronIDs_outputs
# exit()


# axsyn3d = pyplot.subplot(projection='3d')
# synapse_network_diagram_3d(axsyn3d, network)
# pyplot.show()

numInputs = 2
constantInput 	= ConstantInput(constVal=250.0)
linearInput		= LinearInput(0, slope=0.5)

network.add_inputs(numInputsToAdd=numInputs)

W_inp = numpy.zeros(shape=(numInputs, network.N))
W_inp[0, numpy.sort(neuronIDs_inputs)[0]] 	= 1.0
W_inp[1, numpy.sort(neuronIDs_inputs)[1:]]	= 1.0

print numpy.sort(neuronIDs_inputs)[0]
print numpy.sort(neuronIDs_inputs)[1:]

network.set_input_connectivity(connectivity=W_inp)

# axinpmat = pyplot.subplot()
# input_connectivity_matrix(axinpmat, network.connectionWeights_inputs)
# pyplot.show()



network.initialize_simulation(T_max=500, deltaT=0.5)
# network.initialize_simulation(T_max=50, deltaT=0.1)


R_baseline 	= network.geometry.r
dRdt	 	= 0.005
r_temp      = R_baseline

print "sim start"
while(network.sim_state_valid()):

	# newtime_start = time.time()

	print "tttttt = " + str( network.t )

	network.geometry.set_r(network.geometry.r+dRdt)

	# print network.geometry.distances.shape
	# print neuronIDs_inhib
	# print network.geometry.distances[neuronIDs_inhib]

	# print network.connectionWeights_synInhib[-1]
	# print network.connectionWeights_synInhib[neuronIDs_inhib][-1,:]
	# print numpy.sum(network.connectionWeights_synInhib)
	# print numpy.sum(network.connectionWeights_synInhib[neuronIDs_inhib])
	# exit()

	W_synI = generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='given', initWeightScheme='distance',
															args={	'given_adj':network.connectionWeights_synInhib[neuronIDs_inhib],
																	'init_weight_dist_fn':'exponential', 'p0_w':10000, 'sigma_w':10000,
							                  						'distances': network.geometry.distances[neuronIDs_inhib] } )


	# W_synI = generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='distance',
	# 							            args={'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.7,
	# 							                  'init_weight_dist_fn':'exponential', 'p0_w':40, 'sigma_w':5.0,
	# 							                  'distances':network.geometry.distances} )

	network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)


	print "R      = " + str( network.geometry.r )
	print "SUMWTI = " + str( numpy.sum(network.connectionWeights_synInhib[neuronIDs_inhib]) )
	print "AVGWTI = " + str( numpy.mean(network.connectionWeights_synInhib[neuronIDs_inhib]) )
	print "avgwte = " + str( numpy.mean(network.connectionWeights_synExcit[neuronIDs_excit]) )

	# newtime_end = time.time()
	# print ("new: " + str(newtime_end - newtime_start))

	# oldtime_start = time.time()

	r_temp += dRdt
	network.set_input_vals( vals=[constantInput.val(network.t), 100*(r_temp - R_baseline)] )

	network.sim_step()

	# oldtime_end = time.time()
	# print ("old: " + str(oldtime_end - oldtime_start))
print "sim end"


network_overview_figure(network, synapseDiagram2D=True, gapjunctionDiagram2D=True, spikerateDiagram2D=True)#, neuronIDs_traces=neuronIDs_inputs)
pyplot.show()