# from __future__ import division

# import numpy as numpy
# import pandas as pandas
# from matplotlib import pyplot as pyplot

# import matplotlib.gridspec as gridspec

# import plotly
# from plotly.graph_objs import *

# from LIFNetwork import LIFNetwork
# import NetworkConnectivity
# from MidpointNormalize import MidpointNormalize

from NetworkModels.LIFNetwork import LIFNetwork
from NetworkGeometry.CylinderSurface import CylinderSurface
from NetworkConnectivity.NetworkConnectivity import *

import numpy as numpy


network 	= LIFNetwork()

network.geometry = CylinderSurface(r=1, h=3)
# network.geometry = CylinderSurface(w=20, h=10)

N = 6	#numpy.random.randint(low=2, high=200)

print N

network.add_neurons(numNeuronsToAdd=N,
					V_init=-68.0, V_thresh=-50.0, V_reset=-70.0, V_eqLeak=-68.0, V_eqExcit=0.0, V_eqInhib=-70.0, R_membrane=1.0,
					g_leak=0.3, g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5,
					tau_g_excit=2.0, tau_g_inhib=2.0, refracPeriod=3.0,
					synapse_type='excitatory')

network.geometry.position_neurons(positioning='even')

W_synE 	= generate_connectivity_matrix(	N=network.N, adjacencyScheme='distance_probability', initWeightScheme='constant',
										args={
												'distances':network.geometry.distances,
												'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':1.5,
												'c_w':0.5
											} )

print W_synE

network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e')

# print network.geometry.distances

print network.connectionWeights_synExcit


exit()




network.set_synaptic_connectivity()
network.set_gapjunction_connectivity()
network.set_input_connectivity()

network.label_neurons(neuronIDs=[], label='input')
network.label_neurons(neuronIDs=[], label='output')

# network.initialize_simulation()

simulation 	= NetworkSimulation(network, )

simulate_periodic_pulse_input_sim(network, inputPeriod=2, inputValue=1.0)
