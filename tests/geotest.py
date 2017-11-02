# from __future__ import division

# import numpy as numpy
# import pandas as pandas
from matplotlib import pyplot as pyplot

# import matplotlib.gridspec as gridspec

# import plotly
# from plotly.graph_objs import *

# from LIFNetwork import LIFNetwork
# import NetworkConnectivity
# from NetworkVisualization.MidpointNormalize import MidpointNormalize

from NetworkModels.LIFNetwork import LIFNetwork
from NetworkModels.FHNNetwork import FHNNetwork
from NetworkGeometry.CylinderSurface import CylinderSurface
from NetworkGeometry.TorusSurface import TorusSurface
from NetworkGeometry.PlaneSurface import PlaneSurface
from NetworkGeometry.SpheroidSurface import SpheroidSurface
from NetworkConnectivity.NetworkConnectivity import *
from NetworkInput.ConstantInput import ConstantInput

from NetworkVisualization.NetworkPlots import *
from NetworkVisualization.OverviewFigures import *

import numpy as numpy

numpy.random.seed(6900034)

network 	= FHNNetwork()

# network.geometry = CylinderSurface(r=1, h=3)
# network.geometry = CylinderSurface(w=10, h=10)
# network.geometry = TorusSurface(r_major=3, r_minor=1.5)
# network.geometry = TorusSurface(w=10, h=10)
# network.geometry = PlaneSurface(x=10, y=10)
network.geometry = SpheroidSurface(r_xy=15, r_z=15)


N_excit = 80	#numpy.random.randint(low=2, high=200)
N_inhib = 20


network.add_neurons(numNeuronsToAdd=N_excit,
					V_init=-1.1994, V_peak=1.45, V_eqExcit=2.0, V_eqInhib=-1.25,
					W_init=-0.6243, a=0.7, b=0.8,
					g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=2.0, tau_g_inhib=2.0, tau_W=12.5,
					synapse_type='excitatory', label='')

network.add_neurons(numNeuronsToAdd=N_inhib,
					V_init=-1.1994, V_peak=1.5, V_eqExcit=0.0, V_eqInhib=-1.25,
					W_init=-0.6243, a=0.7, b=0.8,
					g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=2.0, tau_g_inhib=2.0, tau_W=12.5,
					synapse_type='inhibitory', label='')

neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]

network.geometry.position_neurons(positioning='even')
# network.geometry.position_neurons(positioning='random')




I = 3
currentInput1 = ConstantInput(constVal=1.0)
currentInput2 = ConstantInput(constVal=100)
currentInput3 = ConstantInput(constVal=100)

network.add_inputs(numInputsToAdd=I)

inputs_neuronIDs	= [[55], [], []]
# W_inpE = numpy.atleast_2d(numpy.zeros(network.N))
W_inp = numpy.zeros(shape=(I, network.N))
for i in range(I):
	for nID in inputs_neuronIDs[i]:
		W_inp[i, nID] = 1.0
		network.neuronLabels[nID] = 'input_'+str(i)
network.set_input_connectivity(connectivity=W_inp)

outputs_neuronIDs 	= numpy.random.randint(low=0, high=N_excit, size=4)
network.neuronLabels[outputs_neuronIDs] = 'output'


# print network.connectionWeights_inputs
# print network.inputVals
# exit()



# W_synE 	= generate_connectivity_matrix(	N=network.N, adjacencyScheme='distance_probability', initWeightScheme='constant',
# 										args={
# 												'distances':network.geometry.distances,
# 												'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':1.5,
# 												'c_w':0.5
# 											} )
W_synE 	= generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='uniform',
										args={
												'distances':network.geometry.distances,
												'k':4,
												'low':0.5, 'high':0.5
										} )

W_synI 	= generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='uniform',
										args={
												'distances':network.geometry.distances,
												'k':4,
												'low':0.5, 'high':0.5
										} )

W_synG 	= generate_connectivity_matrix(N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='uniform',
										args={
												'distances':network.geometry.distances,
												'k':4,
												'low':0.1, 'high':0.1
										} )


network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)
network.set_gapjunction_connectivity(connectivity=W_synG)

# print network.geometry.distances

# print network.connectionWeights_synExcit




network.initialize_simulation(T_max=10, deltaT=0.5)


# while(network.t < (network.T_max-(network.deltaT/2))):	# The right-hand-side of this conditional is what it is rather than just T_max to avoid numerical roundoff errors causing unexpected conditional outcomes
while(network.sim_state_valid()):

	network.set_input_vals( vals=[currentInput1.val(network.t), currentInput2.val(network.t), currentInput3.val(network.t)] )

	# print "t="+str(network.t)+" inputs="+str(network.inputVals)

	network.sim_step()

	# break

# network.get_neurons_dataframe().to_csv('debugging.txt', sep='\t')

# exit()

# simNeuronsDataFrame	= network.get_neurons_dataframe()
# print simNeuronsDataFrame

# simInputsDataFrame	= network.get_inputs_dataframe()
# print simInputsDataFrame

# print network.get_spike_times()




# hmm = LIF_overview_figure(simNeuronsDataFrame)

# pyplot.show()

# exit()











# print simNeuronsDataFrame.loc[(simNeuronsDataFrame['neuron_id']==nID), 'V'].values



# figrast, axrast = pyplot.subplots()

# spike_raster_plot(axrast, network)

# figtrace, axtrace = pyplot.subplots()

# nID = 1
# x_series 	= simNeuronsDataFrame.loc[(simNeuronsDataFrame['neuron_id']==nID), 't'].values
# trace_V 	= {'data':simNeuronsDataFrame.loc[(simNeuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
# trace_ge 	= {'data':simNeuronsDataFrame.loc[(simNeuronsDataFrame['neuron_id']==nID), 'g_excit'].values, 'label':'g_e', 'color':'blue', 'alpha':0.5, 'linestyle':':'}
# traces_plot(axtrace, x=x_series, y1_traces=[trace_V], y2_traces=[trace_ge], x_axis_label='t', y1_axis_label='Voltage', y2_axis_label='', y1_legend=False, y2_legend=False, fontsize=8, labelsize=6)

# figsyn2d, axsyn2d = pyplot.subplots()

# synapse_network_diagram_2d(axsyn2d, network)

axsyn3d = pyplot.subplot(projection='3d')

synapse_network_diagram_3d(axsyn3d, network)

# figgap2d, axgap2d = pyplot.subplots()

# gapjunction_network_diagram_2d(axgap2d, network)

# figgap3d, axgap3d = pyplot.subplots()

# gapjunction_network_diagram_3d(axgap3d, network)

# figsynmat, axsynmat = pyplot.subplots()

# synapse_connectivity_matrix(axsynmat, network.connectionWeights_synExcit-network.connectionWeights_synInhib)
# # synapse_connectivity_matrix(axsynmat, numpy.zeros(shape=(network.N, network.N)))

# figgapmat, axgapmat = pyplot.subplots()

# gapjunction_connectivity_matrix(axgapmat, network.connectionWeights_gap)
# # gapjunction_connectivity_matrix(axgapmat, numpy.zeros(shape=(network.N, network.N)))

# figinpmat, axinpmat = pyplot.subplots()

# input_connectivity_matrix(axinpmat, network.connectionWeights_inputs)
# # input_connectivity_matrix(axinpmat, numpy.zeros(shape=(network.N, network.N)))

# figrate2d, axrate2d = pyplot.subplots()

# rate_network_diagram_2d(axrate2d, network)



# exit()

# network_overview_figure(network, synapseDiagram2D=True, gapjunctionDiagram2D=True, spikerateDiagram2D=True)

# pyplot.savefig('_____.png', bbox_inches='tight')

pyplot.show()

#************************************
#************************************
#************************************
#************************************
#************************************
exit()
#************************************
#************************************
#************************************
#************************************
#************************************
