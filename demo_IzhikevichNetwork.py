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
from NetworkModels.IzhikevichNetwork import IzhikevichNetwork
from NetworkGeometry.CylinderSurface import CylinderSurface
from NetworkConnectivity.NetworkConnectivity import *
from NetworkInput.ConstantInput import ConstantInput

from NetworkVisualization.NetworkPlots import *
from NetworkVisualization.OverviewFigures import *

import numpy as numpy

numpy.random.seed(69000)

network 	= IzhikevichNetwork()

# network.geometry = CylinderSurface(r=1, h=3)
network.geometry = CylinderSurface(w=10, h=10)

N_excit = 100	#numpy.random.randint(low=2, high=200)
N_inhib = 0


network.add_neurons(numNeuronsToAdd=N_excit,
					V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0, U_init=0.0,
					a=0.02, b=0.2, d=8, R_membrane=0.01, k=0.7,
					g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=2.0, tau_g_inhib=2.0,
					synapse_type='excitatory', label='')

network.add_neurons(numNeuronsToAdd=N_inhib,
					V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0, U_init=0.0,
					a=0.02, b=0.2, d=8, R_membrane=0.01, k=0.7,
					g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=50.0, tau_g_inhib=50.0,
					synapse_type='inhibitory', label='')

neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]

network.geometry.position_neurons(positioning='even')




I = 3
currentInput1 = ConstantInput(constVal=400)
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
												'low':30.0, 'high':30.0
										} )

W_synI 	= generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='uniform',
										args={
												'distances':network.geometry.distances,
												'k':3,
												'low':30.0, 'high':30.0
										} )

W_synG 	= generate_connectivity_matrix(N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='uniform',
										args={
												'distances':network.geometry.distances,
												'k':4,
												'low':0.0, 'high':3.8
										} )


network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)
network.set_gapjunction_connectivity(connectivity=W_synG)

# print network.geometry.distances

# print network.connectionWeights_synExcit




network.initialize_simulation(T_max=5000, deltaT=0.5)


# while(network.t < (network.T_max-(network.deltaT/2))):	# The right-hand-side of this conditional is what it is rather than just T_max to avoid numerical roundoff errors causing unexpected conditional outcomes
while(network.sim_state_valid()):

	network.set_input_vals( vals=[currentInput1.val(network.t), currentInput2.val(network.t), currentInput3.val(network.t)] )

	# print "t="+str(network.t)+" inputs="+str(network.inputVals)

	network.sim_step()

	# break

# network.get_neurons_dataframe().to_csv('debugging.txt', sep='\t')

# exit()

simNeuronsDataFrame	= network.get_neurons_dataframe()
# print simNeuronsDataFrame

simInputsDataFrame	= network.get_inputs_dataframe()
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

# figsyn3d, axsyn3d = pyplot.subplots()

# synapse_network_diagram_3d(axsyn3d, network)

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

izhikevich_network_overview_figure(network, synapseDiagram2D=True, gapjunctionDiagram2D=True, spikerateDiagram2D=True)

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















from matplotlib import pyplot as pyplot

import matplotlib.gridspec as gridspec

import plotly
from plotly.graph_objs import *

from mpl_toolkits.mplot3d.art3d import Line3DCollection

simsummaryFigure = pyplot.figure(figsize=(16,10))

neuronsIDs_raster = range(network.N)

gs1 = gridspec.GridSpec(12, 9)
gs1.update(left=0.02, right=0.6, wspace=0.05, top=0.98, bottom=0.02, hspace=1.0)
ax_raster = pyplot.subplot(gs1[0:5, :])
ax_netSyn = pyplot.subplot(gs1[5:9, 0:3], projection='3d')
ax_netGap = pyplot.subplot(gs1[5:9, 3:6], projection='3d')
# ax_netRat = pyplot.subplot(gs1[5:9, 6:9], projection='3d')
ax_matSyn = pyplot.subplot(gs1[9:12,0:3])
ax_matGap = pyplot.subplot(gs1[9:12,3:6])
ax_matInp = pyplot.subplot(gs1[9:12,6:9])



spikeTimes 	= network.get_spike_times()
print spikeTimes
for n, spike_t in enumerate(spikeTimes):
	ax_raster.vlines(x=spike_t, ymin=n+0.5, ymax=n+1.5)
ax_raster.set_ylim(0.5, len(spikeTimes) + 0.5)
ax_raster.set_xlim(0, network.T_max)
ax_raster.set_yticks([] if network.N > 50 else range(1, network.N+1))
ax_raster.grid(False)
ax_raster.set_xlabel('t')
ax_raster.set_ylabel('Neuron Spikes')
ax_raster.invert_yaxis()


connectionWeights_synExcitAndInhib	= network.connectionWeights_synExcit - network.connectionWeights_synInhib
cmapRdBu_min	= connectionWeights_synExcitAndInhib.min()
cmapRdBu_max	= connectionWeights_synExcitAndInhib.max()
cmapRdBu_midpt	= 0.0


neuronPositions	= network.geometry.cartesianCoords
neuronSynEndpts	= []
neuronSynWts	= []
for i in range(network.N):
	for j in range(network.N):
		if(connectionWeights_synExcitAndInhib[i,j] != 0.0):
			neuronSynEndpts.append( [neuronPositions[i], neuronPositions[j]] )
			neuronSynWts.append(connectionWeights_synExcitAndInhib[i,j])

# Customize the colormap used to color the synaptic edges,
# specifically to decrease the alpha of the color range overall to lighten the RdBu colormap colors.
from matplotlib.colors import ListedColormap
cmapOrig 		= pyplot.get_cmap('RdBu')				# Choose colormap
cmapSyn 		= cmapOrig(numpy.arange(cmapOrig.N))	# Get the colormap colors
cmapSyn[:,-1] 	= numpy.linspace(1.0, 1.0, cmapOrig.N)	# Set alphas along cmap
cmapSyn 		= ListedColormap(cmapSyn)				# Create new colormap

edgesSyn = Line3DCollection(neuronSynEndpts, cmap=cmapSyn, clim=(cmapRdBu_min, cmapRdBu_max), norm=MidpointNormalize(midpoint=cmapRdBu_midpt, vmin=cmapRdBu_min, vmax=cmapRdBu_max))
edgesSyn.set_array(numpy.asarray(neuronSynWts)) # color the segments by our parameter
ax_netSyn.add_collection3d(edgesSyn)

neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])
print neuronIDs_nonIOExcit
print neuronIDs_nonIOInhib
print neuronIDs_inputExcit
print neuronIDs_inputInhib
print neuronIDs_outputExcit
print neuronIDs_outputInhib
ax_netSyn.scatter(xs=neuronPositions[neuronIDs_nonIOExcit,0], ys=neuronPositions[neuronIDs_nonIOExcit,1], zs=neuronPositions[neuronIDs_nonIOExcit,2], marker='o', c='black', edgecolors='blue')
ax_netSyn.scatter(xs=neuronPositions[neuronIDs_nonIOInhib,0], ys=neuronPositions[neuronIDs_nonIOInhib,1], zs=neuronPositions[neuronIDs_nonIOInhib,2], marker='o', c='black', edgecolors='red')
ax_netSyn.scatter(xs=neuronPositions[neuronIDs_inputExcit,0], ys=neuronPositions[neuronIDs_inputExcit,1], zs=neuronPositions[neuronIDs_inputExcit,2], marker='^', c='limegreen', edgecolors='blue', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size
ax_netSyn.scatter(xs=neuronPositions[neuronIDs_inputInhib,0], ys=neuronPositions[neuronIDs_inputInhib,1], zs=neuronPositions[neuronIDs_inputInhib,2], marker='^', c='limegreen', edgecolors='red', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)
ax_netSyn.scatter(xs=neuronPositions[neuronIDs_outputExcit,0], ys=neuronPositions[neuronIDs_outputExcit,1], zs=neuronPositions[neuronIDs_outputExcit,2], marker='s', c='orange', edgecolors='blue', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)
ax_netSyn.scatter(xs=neuronPositions[neuronIDs_outputInhib,0], ys=neuronPositions[neuronIDs_outputInhib,1], zs=neuronPositions[neuronIDs_outputInhib,2], marker='s', c='orange', edgecolors='red', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)

ax_netSyn.set_axis_off()
if(True or network.geometry.numVizDimensions == 3):		# TODO THIS CONDITIONAL HAD A OR TRUE TACKED ON FOR HACKY GETTING AROUND NUMVIZDIMENSIONS
	ax_netSyn.set_xlim3d(0.65*ax_netSyn.get_xlim3d())	# "zoom in" by clipping the 3d view limits
	ax_netSyn.set_ylim3d(0.65*ax_netSyn.get_ylim3d())
	ax_netSyn.set_zlim3d(0.65*ax_netSyn.get_zlim3d())

neuronGapEndpts	= []
neuronGapWts	= []
for i in range(network.N):
	for j in range(network.N):
		if(network.connectionWeights_gap[i,j] != 0.0):
			neuronGapEndpts.append( [neuronPositions[i], neuronPositions[j]] )
			neuronGapWts.append(network.connectionWeights_gap[i,j])

# Customize the colormap used to color the gap junction edges,
# specifically to decrease the alpha of the color range overall to lighten the RdBu colormap colors.
from matplotlib.colors import ListedColormap
cmapGap_alpha	= 1.0 if network.connectionWeights_gap.max() > 0 else 0.0
cmapOrig 		= pyplot.get_cmap('Purples')			# Choose colormap
cmapGap 		= cmapOrig(numpy.arange(cmapOrig.N))	# Get the colormap colors
cmapGap[:,-1] 	= numpy.linspace(1.0, 1.0, cmapOrig.N)	# Set alphas along cmap
cmapGap 		= ListedColormap(cmapGap)

edgesGap = Line3DCollection(neuronGapEndpts, cmap=cmapGap, clim=(0.0, network.connectionWeights_gap.max()))
# edgesGap = Line3DCollection(neuronGapEndpts, cmap=cmapGap)
edgesGap.set_array(numpy.asarray(neuronGapWts)) # color the segments by our parameter
ax_netGap.add_collection3d(edgesGap)

ax_netGap.scatter(xs=neuronPositions[neuronIDs_nonIOExcit,0], ys=neuronPositions[neuronIDs_nonIOExcit,1], zs=neuronPositions[neuronIDs_nonIOExcit,2], marker='o', c='black', edgecolors='k')
ax_netGap.scatter(xs=neuronPositions[neuronIDs_nonIOInhib,0], ys=neuronPositions[neuronIDs_nonIOInhib,1], zs=neuronPositions[neuronIDs_nonIOInhib,2], marker='o', c='black', edgecolors='k')
ax_netGap.scatter(xs=neuronPositions[neuronIDs_inputExcit,0], ys=neuronPositions[neuronIDs_inputExcit,1], zs=neuronPositions[neuronIDs_inputExcit,2], marker='^', c='limegreen', edgecolors='k', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size
ax_netGap.scatter(xs=neuronPositions[neuronIDs_inputInhib,0], ys=neuronPositions[neuronIDs_inputInhib,1], zs=neuronPositions[neuronIDs_inputInhib,2], marker='^', c='limegreen', edgecolors='k', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)
ax_netGap.scatter(xs=neuronPositions[neuronIDs_outputExcit,0], ys=neuronPositions[neuronIDs_outputExcit,1], zs=neuronPositions[neuronIDs_outputExcit,2], marker='s', c='orange', edgecolors='k', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)
ax_netGap.scatter(xs=neuronPositions[neuronIDs_outputInhib,0], ys=neuronPositions[neuronIDs_outputInhib,1], zs=neuronPositions[neuronIDs_outputInhib,2], marker='s', c='orange', edgecolors='k', s=1.5*(pyplot.rcParams['lines.markersize']**2), depthshade=False)
# ax_netGap.scatter(xs=neuronPositions[:,0], ys=neuronPositions[:,1], zs=neuronPositions[:,2], c='k', marker='o')

ax_netGap.set_axis_off()
if(True or network.geometry.numVizDimensions == 3):		# TODO THIS CONDITIONAL HAD A OR TRUE TACKED ON FOR HACKY GETTING AROUND NUMVIZDIMENSIONS
	ax_netGap.set_xlim3d(0.65*ax_netGap.get_xlim3d())	# "zoom in" by clipping the 3d view limits
	ax_netGap.set_ylim3d(0.65*ax_netGap.get_ylim3d())
	ax_netGap.set_zlim3d(0.65*ax_netGap.get_zlim3d())

############################

neuronsNumSpikes	= testNetworkNeuronsDataFrame.groupby(['neuron_id']).sum().reset_index()['spike'].tolist()
# # neuronSpikeRates	= [neuronsNumSpikes[i]/network.T_max for i in range(len(neuronsNumSpikes))]
# neuronSpikeRates	= numpy.asarray(neuronsNumSpikes) / network.T_max
# print neuronsNumSpikes
# print neuronSpikeRates
# print type(neuronsNumSpikes)
# print type(neuronSpikeRates)
# print type(neuronPositions[neuronIDs_nonIOExcit,2])
# print neuronSpikeRates.shape
# print neuronPositions[neuronIDs_nonIOExcit,2].shape

# # Customize the colormap used to color the synaptic edges on the spike rate plot,
# # specifically to decrease the alpha of the color range overall to lighten the RdBu colormap colors.
# from matplotlib.colors import ListedColormap
# cmapOrig 			= pyplot.get_cmap('RdBu')				# Choose colormap
# cmapSynLite 		= cmapOrig(numpy.arange(cmapOrig.N))	# Get the colormap colors
# cmapSynLite[:,-1] 	= numpy.linspace(0.051, 0.051, cmapOrig.N)	# Set alphas along cmap
# cmapSynLite 		= ListedColormap(cmapSynLite)				# Create new colormap

# edgesRat = Line3DCollection(neuronSynEndpts, cmap=cmapSynLite, clim=(cmapRdBu_min, cmapRdBu_max), norm=MidpointNormalize(midpoint=cmapRdBu_midpt, vmin=cmapRdBu_min, vmax=cmapRdBu_max))
# edgesRat.set_array(numpy.asarray(neuronSynWts)) # color the segments by our parameter
# ax_netRat.add_collection3d(edgesRat)

# print "heyo " + str(type(neuronIDs_nonIOExcit))

# # cmapSpikeRate	= pyplot.get_cmap('hot')
# # cmapSpikeRate.set_array(neuronSpikeRates)
# # hmm = ax_netRat.scatter(xs=neuronPositions[neuronIDs_nonIOExcit,0], ys=neuronPositions[neuronIDs_nonIOExcit,1], zs=neuronPositions[neuronIDs_nonIOExcit,2], marker='o', c=cmapSpikeRate(neuronSpikeRates[neuronIDs_nonIOExcit]), cmap='hot', edgecolors='none')
# # ax_netRat.scatter(xs=neuronPositions[neuronIDs_nonIOInhib,0], ys=neuronPositions[neuronIDs_nonIOInhib,1], zs=neuronPositions[neuronIDs_nonIOInhib,2], marker='o', c=cmapSpikeRate(neuronSpikeRates[neuronIDs_nonIOInhib]), cmap='hot', edgecolors='none')
# # ax_netRat.scatter(xs=neuronPositions[neuronIDs_inputExcit,0], ys=neuronPositions[neuronIDs_inputExcit,1], zs=neuronPositions[neuronIDs_inputExcit,2], marker='^', c=cmapSpikeRate(neuronSpikeRates[neuronIDs_inputExcit]), cmap='hot', edgecolors='none', s=1.5*(pyplot.rcParams['lines.markersize']**2))	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size
# # ax_netRat.scatter(xs=neuronPositions[neuronIDs_inputInhib,0], ys=neuronPositions[neuronIDs_inputInhib,1], zs=neuronPositions[neuronIDs_inputInhib,2], marker='^', c=cmapSpikeRate(neuronSpikeRates[neuronIDs_inputInhib]), cmap='hot', edgecolors='none', s=1.5*(pyplot.rcParams['lines.markersize']**2))
# # ax_netRat.scatter(xs=neuronPositions[neuronIDs_outputExcit,0], ys=neuronPositions[neuronIDs_outputExcit,1], zs=neuronPositions[neuronIDs_outputExcit,2], marker='s', c=cmapSpikeRate(neuronSpikeRates[neuronIDs_outputExcit]), cmap='hot', edgecolors='none', s=1.5*(pyplot.rcParams['lines.markersize']**2))
# # ax_netRat.scatter(xs=neuronPositions[neuronIDs_outputInhib,0], ys=neuronPositions[neuronIDs_outputInhib,1], zs=neuronPositions[neuronIDs_outputInhib,2], marker='s', c=cmapSpikeRate(neuronSpikeRates[neuronIDs_outputInhib]), cmap='hot', edgecolors='none', s=1.5*(pyplot.rcParams['lines.markersize']**2))

# hmm= ax_netRat.scatter(xs=neuronPositions[:,0], ys=neuronPositions[:,1], zs=neuronPositions[:,2], marker='o', c=neuronSpikeRates, cmap='hot', edgecolors='none', s=1.0*(pyplot.rcParams['lines.markersize']**2))

# ax_netRat.set_axis_off()
# if(network.geometry.numVizDimensions == 3):
# 	ax_netRat.set_xlim3d(0.65*ax_netRat.get_xlim3d())	# "zoom in" by clipping the 3d view limits
# 	ax_netRat.set_ylim3d(0.65*ax_netRat.get_ylim3d())
# 	ax_netRat.set_zlim3d(0.65*ax_netRat.get_zlim3d())

# cbar_netRat	= pyplot.colorbar(hmm, ax=ax_netRat, orientation='horizontal', pad=0.0, fraction=0.1)
# cbar_netRat.set_label('Average Spike Rate')
# cbar_netRat.ax.xaxis.set_label_position('top')
# cbar_netRat.ax.tick_params(labelsize=8)

# ax_netRat.patch.set_facecolor([0.95, 0.95, 0.95])

############################

# Customize the colormap used to color the synaptic edges,
# specifically to decrease the alpha of the color range overall to lighten the RdBu colormap colors.

#TODO: PUT THIS BACK IN; TAKEN OUT FOR HACKY RUNNING ON 10/19
# cmapInp_alpha	= 1.0 if ((network.numInputs_excit()>0 and network.connectionWeights_inpExcit.max()>0.0) or (network.numInputs_inhib()>0 and network.connectionWeights_inpInhib.max()>0.0)) else 0.0
# cmapOrig 		= pyplot.get_cmap('Greens')				# Choose colormap
# cmapInp 		= cmapOrig(numpy.arange(cmapOrig.N))	# Get the colormap colors
# cmapInp[:,-1] 	= numpy.linspace(cmapInp_alpha, cmapInp_alpha, cmapOrig.N)	# Set alphas along cmap
# cmapInp 		= ListedColormap(cmapInp)				# Create new colormap




img_matSyn 	= ax_matSyn.imshow(connectionWeights_synExcitAndInhib, cmap='RdBu', clim=(cmapRdBu_min, cmapRdBu_max), norm=MidpointNormalize(midpoint=cmapRdBu_midpt, vmin=cmapRdBu_min, vmax=cmapRdBu_max))
cbar_matSyn	= pyplot.colorbar(img_matSyn, ax=ax_matSyn)
cbar_matSyn.ax.tick_params(labelsize=(8 if ((connectionWeights_synExcitAndInhib.min()!=0.0 or connectionWeights_synExcitAndInhib.max()!=0.0)
											and connectionWeights_synExcitAndInhib.max()!=connectionWeights_synExcitAndInhib.min()) else 0))
ax_matSyn.set_xticklabels([])
ax_matSyn.set_yticklabels([])
ax_matSyn.set_xticks([])
ax_matSyn.set_yticks([])
ax_matSyn.set_title("Synaptic Weights")

img_matGap 	= ax_matGap.imshow(network.connectionWeights_gap, cmap='Purples', interpolation='nearest')
cbar_matGap	= pyplot.colorbar(img_matGap, ax=ax_matGap)
cbar_matGap.ax.tick_params(labelsize=(8 if network.connectionWeights_gap.max() > 0.0 else 0))
ax_matGap.set_xticklabels([])
ax_matGap.set_yticklabels([])
ax_matGap.set_xticks([])
ax_matGap.set_yticks([])
ax_matGap.set_title("Gap Junction Weights")

#TODO: PUT THIS BACK IN; TAKEN OUT FOR HACKY RUNNING ON 10/19
# img_matInp 	= ax_matInp.imshow(network.connectionWeights_inputs, cmap=cmapInp, interpolation='nearest')
# cbar_matInp	= pyplot.colorbar(img_matInp, ax=ax_matInp)
# cbar_matInp.ax.tick_params(labelsize=(8 if ((network.numInputs_excit()>0 and network.connectionWeights_inputs.max()>0.0) or (network.numInputs_inhib()>0 and network.connectionWeights_inputs.max()>0.0)) else 0))
# ax_matInp.set_xticklabels([])
# ax_matInp.set_yticklabels([])
# ax_matInp.set_xticks([])
# ax_matInp.set_yticks([])
# ax_matInp.set_title("Input Weights")


print "yoyoyo"
print neuronsNumSpikes
hiSpikeRateIDs	= list(set(([0] + numpy.asarray(neuronsNumSpikes).argsort()[-6:][::-1].tolist())[:5]))
loSpikeRateIDs	= numpy.asarray(neuronsNumSpikes).argsort()[:5].tolist()
print hiSpikeRateIDs
print loSpikeRateIDs
# print set(([0] + hiSpikeRateIDs)[:5] + loSpikeRateIDs)
# exit()

# neuronsIDs_voltcond	= range(10)
neuronsIDs_voltcond	= hiSpikeRateIDs + loSpikeRateIDs

gs2 = gridspec.GridSpec(len(neuronsIDs_voltcond), 1)
gs2.update(left=0.65, right=0.97, top=0.98, bottom=0.02, hspace=0.15)
for i, nID in enumerate(neuronsIDs_voltcond):
	ax_n = pyplot.subplot(gs2[i, :])

	ax_n_1 = ax_n
	ax_n_2	= ax_n_1.twinx()

	ax_n_2.plot(
				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == nID), 't'].values,
				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == nID), 'g_excit'].values,
				color='blue', alpha=0.5, linestyle=':'
				)
	ax_n_2.plot(
				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == nID), 't'].values,
				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == nID), 'g_inhib'].values,
				color='red', alpha=0.5, linestyle=':'
				)
	ax_n_2.set_ylabel("N"+str(nID)+" cond.", fontsize=8)
	# ax_n_2.set(ylabel="Neur"+str(n)+" conductance")
	ax_n_2.set(ylim=[testNetworkNeuronsDataFrame['g_excit'].min(), testNetworkNeuronsDataFrame['g_excit'].max()])
	ax_n_2.set(xlim=[0, network.T_max])
	ax_n_2.grid(False)
	ax_n_2.tick_params(axis='y', which='major', labelsize=6)


	ax_n_1.plot(
				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == nID), 't'].values,
				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == nID), 'V'].values,
				color='black', alpha=1.0, linewidth=1
				)
	ax_n_1.set_ylabel("N"+str(nID)+" volt (mv)", fontsize=8)
	# ax_n_1.set(ylabel="Neur"+str(n)+" voltage")
	ax_n_1.set(ylim=[testNetworkNeuronsDataFrame['V'].min(), testNetworkNeuronsDataFrame['V'].max()])
	ax_n_1.set(xlim=[0, network.T_max])
	ax_n_1.grid(False)
	ax_n_1.tick_params(axis='y', which='major', labelsize=6)

	if(i == neuronsIDs_voltcond[-1]):	# if on the last neuronID, label x axis...
		ax_n_1.set(xlabel="t")
		ax_n_2.legend(['g_e', 'g_i'], loc='upper right', fontsize='x-small')
	else:								# else, for all others hide x axis tick labels
		ax_n_1.set_xticklabels([])
		ax_n_2.set_xticklabels([])




# gs1.tight_layout()
simsummaryFigure.patch.set_facecolor('white')

# pyplot.show()

spikeTimes	= []
for n in range(network.N):
	spikeTimes.append( testNetworkNeuronsDataFrame.loc[((testNetworkNeuronsDataFrame['neuron_id'] == n) & (testNetworkNeuronsDataFrame['spike'] == 1)), 't'].values )
# print spikeTimes

# spikeFigure = pyplot.figure(figsize=(10,4))
# ax 	= spikeFigure.add_subplot(111)
# for n, spike_t in enumerate(spikeTimes):
# 	pyplot.vlines(x=spike_t, ymin=n+0.5, ymax=n+1.5)
# ax.set_ylim(0.5, len(spikeTimes) + 0.5)
# ax.set_yticks(range(1, network.N+1))
# # ax.set_yticks(numpy.arange(0.5, network.N+0.5, 1.0))
# # ax.grid(which='major')
# ax.grid(False)
# ax.set_xlabel('t')
# ax.set_ylabel('neuron_id')
# ax.invert_yaxis()
# spikeFigure.patch.set_facecolor('white')

# print "distances = \n" + str(network.geometry.distances)

# print "W_synE = \n" + str(W_synE)



# pyplot.savefig('_____.png', bbox_inches='tight')

pyplot.show()

exit()









connectionWeights_synExcitAndInhib	= network.connectionWeights_synExcit - network.connectionWeights_synInhib

neuronPositions	= network.geometry.cartesianCoords

from mpl_toolkits.mplot3d.art3d import Line3DCollection

x = numpy.linspace(0,1,10)
y = numpy.random.random((10,))
z = numpy.linspace(0,1,10)
t = numpy.linspace(0,1,10)

# generate a list of (x,y,z) points
points = numpy.array([x,y,z]).transpose().reshape(-1,1,3)
print(points.shape)  # Out: (len(x),1,2)
print points
# exit()

# set up a list of segments
segs = numpy.concatenate([points[:-1],points[1:]],axis=1)
print(segs.shape)  # Out: ( len(x)-1, 2, 2 )
                  # see what we've done here -- we've mapped our (x,y)
                  # points to an array of segment start/end coordinates.
                  # segs[i,0,:] == segs[i-1,1,:]
print segs
# exit()

neuronSynEndpts	= []
neuronSynWts	= []
for i in range(network.N):
	for j in range(network.N):
		if(connectionWeights_synExcitAndInhib[i,j] != 0.0):
			neuronSynEndpts.append( [neuronPositions[i], neuronPositions[j]] )
			neuronSynWts.append(connectionWeights_synExcitAndInhib[i,j])


# make the collection of segments
cmapRdBu_min	= connectionWeights_synExcitAndInhib.min()
cmapRdBu_max	= connectionWeights_synExcitAndInhib.max()
cmapRdBu_midpt	= 0.0
lc = Line3DCollection(neuronSynEndpts, cmap=pyplot.get_cmap('RdBu'), clim=(cmapRdBu_min, cmapRdBu_max), norm=MidpointNormalize(midpoint=cmapRdBu_midpt, vmin=cmapRdBu_min, vmax=cmapRdBu_max))
lc.set_array(numpy.asarray(neuronSynWts)) # color the segments by our parameter
# lc.set_array(t) # color the segments by our parameter
print t
print numpy.asarray(neuronSynWts)
# print lc
# exit()

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)
fig.colorbar(neuronSynWts)
# pyplot.show()

# exit()


# from mpl_toolkits.mplot3d import Axes3D



# neuronSynEndpts	= [[], [], []]
# for i in range(network.N):
# 	for j in range(network.N):
# 		if(network.connectionWeights_synExcit[i,j] > 0.0):
# 			neuronSynEndpts[0] += [neuronPositions[i][0], neuronPositions[j][0], None]
# 			neuronSynEndpts[1] += [neuronPositions[i][1], neuronPositions[j][1], None]
# 			neuronSynEndpts[2] += [neuronPositions[i][2], neuronPositions[j][2], None]

# dend = [(1,1), (2,3), 'r', (2,2), (4,5), 'g', (5,5), (6,7), 'b']

# networkvizFigure = pyplot.figure()
# ax = networkvizFigure.add_subplot(111)#, projection='3d')

# ax.plot(*dend)

ax.scatter(xs=neuronPositions[:,0], ys=neuronPositions[:,1], zs=neuronPositions[:,2], c='r', marker='o')


# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')



pyplot.show()




exit()



# 'frames': [	{'data': [{'x': [1, 2], 'y': [1, 2]}]},
#             {'data': [{'x': [1, 4], 'y': [1, 4]}]},
#             {'data': [{'x': [3, 4], 'y': [3, 4]}],
#                       'layout': {'title': 'End Title'}}]

# networkGraph_frames	= []

# print sorted(testNetworkNeuronsDataFrame['t'].unique())



simsummaryFigure	= pyplot.figure()







exit()






networkGraph_frames	= []

for f, curT in enumerate( sorted(testNetworkNeuronsDataFrame['t'].unique()) ):
	if(f % 50 == 0):
		print f


		neuronPositions	= network.geometry.cartesianCoords

		# # curT	= network.T_max - network.deltaT
		# print "ttttttttttttttttttt"
		# print testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['t'] == curT)]

		# print neuronPositions

		# [
		# [[ep1_x, ep2_x, None], [ep1_y, ep2_y, None], [ep1_z, ep2_z, None]]
		# ]
		# neuronSynEndpts	= [[endpt pair x coords], [endpt pair y coords], [endpt pair z coords]]
		neuronSynEndpts	= [[], [], []]
		for i in range(network.N):
			for j in range(network.N):
				if(network.connectionWeights_synExcit[i,j] > 0.0):
					neuronSynEndpts[0] += [neuronPositions[i][0], neuronPositions[j][0], None]
					neuronSynEndpts[1] += [neuronPositions[i][1], neuronPositions[j][1], None]
					neuronSynEndpts[2] += [neuronPositions[i][2], neuronPositions[j][2], None]


		neuronGapEndpts	= []
		# neuronInpEndpts	= []

		# print neuronSynEndpts[0]

		networkGraph_edges	= Scatter3d(
										x=neuronSynEndpts[0],
										y=neuronSynEndpts[1],
										z=neuronSynEndpts[2],
										mode='lines',
		               					line=Line(color='rgb(125,125,125)', width=1),
		               					name='synapses',
		               					hoverinfo='none'
										)

		networkGraph_nodes	= Scatter3d(
										x=neuronPositions[:,0],
										y=neuronPositions[:,1],
										z=neuronPositions[:,2],
										mode='markers',
										name='neurons',
										marker=Marker(
														symbol='dot',
														color=testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['t'] == curT), 'V'],
														colorscale='Viridis',
														cmin=testNetworkNeuronsDataFrame['V'].min(),
														cmax=testNetworkNeuronsDataFrame['V'].max(),
														line=Line(color=['rgb(0,0,255)' if type == 'excitatory' else 'rgb(255,0,0)' if type == 'inhibitory' else 'rgb(0,0,0)' for type in network.neuronSynapseTypes], width=1),
														showscale=True,
														colorbar=dict(	thicknessmode='pixels',
																		thickness=15,
																		lenmode='pixels',
																		len=150,
																		title='Voltage (mV)',
																		xanchor='right',
																		titleside='left'
																		)
													)
										)

		networkGraph_frames.append({'data': [networkGraph_edges, networkGraph_nodes]})



networkGraph_axis	= dict(
								showbackground=False,
	          					showline=False,
						    	zeroline=False,
						        showgrid=True,
						        showticklabels=False,
						        title=''
						        )

networkGraph_layout = Layout(
         						title="Neuron Network Simulation",
								width=1000,
								height=1000,
								showlegend=True,
								scene=Scene(
											xaxis=XAxis(networkGraph_axis),
											yaxis=YAxis(networkGraph_axis),
											zaxis=ZAxis(networkGraph_axis),
											),
								margin=Margin(
											t=100
											),
								hovermode='closest',
								# annotations=Annotations([
								# 				Annotation(
								# 							showarrow=False,
								# 							text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
								# 							xref='paper',
								# 							yref='paper',
								# 							x=0,
								# 							y=0.1,
								# 							xanchor='left',
								# 							yanchor='bottom',
								# 							font=Font(size=14)
								# 							)
								# 						]),
								)


# plotly.offline.plot({
#     					"data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
#     					"layout": Layout(title="hello world")
# 					})
# plotly.offline.plot({
#     					"data": [networkGraph_edges, networkGraph_nodes],
#     					"layout": Layout(title="hello world")
# 					})




networkGraph_figdata 	=Data([networkGraph_edges, networkGraph_nodes])
networkGraph_fig 		=Figure(data=networkGraph_figdata, layout=networkGraph_layout, frames=networkGraph_frames)
plotly.offline.plot(networkGraph_fig, filename='plotly__LIFNetwork_test_gapDThresh-'+str(_gap_d_thresh_)+'_gapCW-'+str(_gap_c_w_)+'_numInhib-'+str(numNeurons_inhib)+'_'+str(networkGeometry)+'.html')#, image='png', image_filename="/home/ryan/Projects/hydra/neuron-dynamnic-models/plotly_LIFNetwork_test")




# voltcondFigure, axarr = pyplot.subplots(network.N, sharex=True, sharey=True, figsize=(10, 10))
# for n in range(network.N):
# 	# figure.subplot(network.N, 1, n+1)
# 	ax1 = axarr[n]
# 	ax2	= ax1.twinx()

# 	ax2.plot(
# 				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == n), 't'].values,
# 				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == n), 'g_excit'].values,
# 				color='green', alpha=0.5, linestyle=':'
# 				)
# 	ax2.plot(
# 				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == n), 't'].values,
# 				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == n), 'g_inhib'].values,
# 				color='red', alpha=0.5, linestyle=':'
# 				)
# 	ax2.set(ylabel="Neur"+str(n)+" conductance")
# 	ax2.set(ylim=[testNetworkNeuronsDataFrame['g_excit'].min(), testNetworkNeuronsDataFrame['g_excit'].max()])
# 	ax2.grid(False)

# 	ax1.plot(
# 				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == n), 't'].values,
# 				testNetworkNeuronsDataFrame.loc[(testNetworkNeuronsDataFrame['neuron_id'] == n), 'V'].values,
# 				color='black', alpha=1.0, linewidth=2
# 				)
# 	ax1.set(ylabel="Neur"+str(n)+" voltage")
# 	ax1.set(ylim=[testNetworkNeuronsDataFrame['V'].min(), testNetworkNeuronsDataFrame['V'].max()])
# 	ax1.grid(False)

# 	if(n == network.N-1):
# 		ax1.set(xlabel="t")

# voltcondFigure.tight_layout()
# voltcondFigure.patch.set_facecolor('white')
# pyplot.show()


spikeTimes	= []
for n in range(network.N):
	spikeTimes.append( testNetworkNeuronsDataFrame.loc[((testNetworkNeuronsDataFrame['neuron_id'] == n) & (testNetworkNeuronsDataFrame['spike'] == 1)), 't'].values )
print spikeTimes

# spikeFigure = pyplot.figure(figsize=(10,4))
# ax 	= spikeFigure.add_subplot(111)
# for n, spike_t in enumerate(spikeTimes):
# 	pyplot.vlines(x=spike_t, ymin=n+0.5, ymax=n+1.5)
# ax.set_ylim(0.5, len(spikeTimes) + 0.5)
# ax.set_yticks(range(1, network.N+1))
# # ax.set_yticks(numpy.arange(0.5, network.N+0.5, 1.0))
# # ax.grid(which='major')
# ax.grid(False)
# ax.set_xlabel('t')
# ax.set_ylabel('neuron_id')
# ax.invert_yaxis()
# spikeFigure.patch.set_facecolor('white')

print "distances = \n" + str(network.geometry.distances)

print "W_synE = \n" + str(W_synE)



rasterweightFigure	= pyplot.figure(figsize=(12,9))
# ax1 	= weightmapFigure.add_subplot(211)
# ax1.imshow(network.connectionWeights_synExcit, cmap='hot', interpolation='nearest')
# ax2 	= weightmapFigure.add_subplot(212)
# ax2.hist(network.connectionWeights_synExcit.flatten())

connectionWeights_synExcitAndInhib	= network.connectionWeights_synExcit - network.connectionWeights_synInhib

ax1 = pyplot.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
for n, spike_t in enumerate(spikeTimes):
	pyplot.vlines(x=spike_t, ymin=n+0.5, ymax=n+1.5)
ax1.set_ylim(0.5, len(spikeTimes) + 0.5)
ax1.set_yticks([] if network.N > 50 else range(1, network.N+1))
# ax.set_yticks(numpy.arange(0.5, network.N+0.5, 1.0))
# ax.grid(which='major')
ax1.grid(False)
ax1.set_xlabel('t')
ax1.set_ylabel('neurons')
ax1.invert_yaxis()

ax2 = pyplot.subplot2grid((3, 3), (2, 0))
cmapRdBu_min	= connectionWeights_synExcitAndInhib.min()
cmapRdBu_max	= connectionWeights_synExcitAndInhib.max()
cmapRdBu_midpt	= 0.0
ax2.imshow(connectionWeights_synExcitAndInhib, cmap='RdBu', clim=(cmapRdBu_min, cmapRdBu_max), norm=MidpointNormalize(midpoint=cmapRdBu_midpt, vmin=cmapRdBu_min, vmax=cmapRdBu_max))
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = pyplot.subplot2grid((3, 3), (2, 1))
# ax3.imshow(connectionWeights_synExcitAndInhib, cmap='RdBu', interpolation='nearest')
cmapRdBu_min	= connectionWeights_synExcitAndInhib.min()
cmapRdBu_max	= connectionWeights_synExcitAndInhib.max()
cmapRdBu_midpt	= 0.0
ax3.imshow(network.connectionWeights_gap, cmap='Purples', interpolation='nearest')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_yticks([])

ax4 = pyplot.subplot2grid((3, 3), (2, 2))
ax4.imshow(network.connectionWeights_inpExcit, cmap='Greens', interpolation='nearest')
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

rasterweightFigure.patch.set_facecolor('white')



rasterweightFigure.tight_layout()

pyplot.savefig('plot__LIFNetwork_test_gapDThresh-'+str(_gap_d_thresh_)+'_gapCW-'+str(_gap_c_w_)+'_numInhib-'+str(numNeurons_inhib)+'_'+str(networkGeometry)+'.png', bbox_inches='tight')

pyplot.show()





