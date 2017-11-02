from __future__ import division

from os import chdir    
chdir('../') # Navigate to the path containing the package directory from this script location

from neuronalnetworks import *

import numpy as numpy




network = IzhikevichNetwork()

N_excit = 100    # number of excitatory neurons
N_inhib = 20    # number of inhibitory neurons

network.add_neurons(numNeuronsToAdd=N_excit,
    V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0, 
    U_init=0.0, a=0.02, b=0.2, d=8, k=0.7, R_membrane=0.01, 
    g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=2.0, tau_g_inhib=2.0,
    synapse_type='excitatory') 

network.add_neurons(numNeuronsToAdd=N_inhib,
    V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0, 
    U_init=0.0, a=0.02, b=0.2, d=8, k=0.7, R_membrane=0.01, 
    g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=2.0, tau_g_inhib=2.0,
    synapse_type='inhibitory') 

neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]

network.geometry = CylinderSurface(r = 3, h = 10)
# network.geometry = SpheroidSurface(r_xy=5, r_z=10)

network.geometry.add_neurons(numNeuronsToAdd=N_excit+N_inhib)

network.geometry.position_neurons(positioning='even', neuronIDs=neuronIDs_excit)
network.geometry.position_neurons(positioning='random', bounds={'h':[4,6]}, neuronIDs=neuronIDs_inhib)

# W_synE 	= generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='distance',
# 													args={
# 															'distances':network.geometry.distances,
# 															'k':5,
# 															'init_weight_dist_fn':'exponential', 'p0_w':40, 'sigma_w':5.0
# 													} )

W_synE 	= generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='uniform',
													args={
															'distances':network.geometry.distances,
															'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':1.2,
															'low':20.0, 'high':40.0
															# 'init_weight_dist_fn':'linear', 'p0_w':40, 'sigma_w':5.0,
													} )

print "***"+str(numpy.count_nonzero(W_synE))+" * "+str(numpy.count_nonzero(W_synE)/(N_excit*1.0))

W_synI 	= generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='distance',
													args={
															'distances':network.geometry.distances,
															'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.9,
															'init_weight_dist_fn':'exponential', 'p0_w':40, 'sigma_w':5.0,
													} )

# W_synG 	= generate_connectivity_matrix(N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='constant',
# 										args={
# 												'distances':network.geometry.distances,
# 												'k':6,
# 												'c_w':1.0
# 										} )
# print network.neuronIDs

W_synG = numpy.zeros(shape=(network.N, network.N))

for nID in network.neuronIDs:
	nID = int(nID)
	heightPercentile = network.geometry.cartesianCoords[int(nID)][2]/network.geometry.h
	W_synG[nID] = generate_connectivity_vector(N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='constant', args={'distances':network.geometry.distances[int(nID)],'k':5,'c_w':1.0}, sparsity=heightPercentile, neuronID=int(nID))
	
network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)
network.set_gapjunction_connectivity(connectivity=W_synG)

axsyn3d = pyplot.subplot(projection='3d')

synapse_network_diagram_3d(axsyn3d, network)

pyplot.show()

axgap3d = pyplot.subplot(projection='3d')

gapjunction_network_diagram_3d(axgap3d, network)

pyplot.show()
