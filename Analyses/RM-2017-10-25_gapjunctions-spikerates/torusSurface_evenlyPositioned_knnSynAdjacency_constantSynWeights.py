from __future__ import division

import numpy as numpy
import pandas as pandas
# import seaborn as seaborn
from matplotlib import pyplot as pyplot

import sys
# sys.path.append('/Users/ryan/Projects/neuronal-networks')
sys.path.append('../../../..')
from NetworkModels.LIFNetwork import LIFNetwork
from NetworkGeometry.TorusSurface import TorusSurface
from NetworkConnectivity.NetworkConnectivity import *
from NetworkInput.ConstantInput import ConstantInput
from NetworkVisualization.NetworkPlots import *
from NetworkVisualization.OverviewFigures import *

experimentData	= []

counter = 0

numReps	= 3
for d, _CUR_D_THRESH_ in enumerate(numpy.arange(0.0, 5.0, 0.125)):
	for c, _CUR_C_W_ in enumerate(numpy.arange(0.0, 1.0, 0.025)):
		for rep in range(numReps):
			counter += 1

			print "\n\n===================="+str(counter)+"\nd_thresh = "+str(_CUR_D_THRESH_)+"  c_w = "+str(_CUR_C_W_)+"  rep = " + str(rep)

			network 	= LIFNetwork()

			network.geometry = TorusSurface(w=10, h=10)

			N_excit = 100
			network.add_neurons(numNeuronsToAdd=N_excit,
								V_init=-68.0, V_thresh=-50.0, V_reset=-70.0, V_eqLeak=-68.0, V_eqExcit=0.0, V_eqInhib=-70.0, R_membrane=1.0,
								g_leak=0.3, g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5,
								tau_g_excit=2.0, tau_g_inhib=2.0, refracPeriod=3.0,
								synapse_type='excitatory')

			neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
			neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]

			network.geometry.position_neurons(positioning='even')

			numInputs = 1
			currentInput = ConstantInput(constVal=50)
			network.add_inputs(numInputsToAdd=numInputs)
			inputs_neuronIDs	= [[55]]
			W_inp = numpy.zeros(shape=(numInputs, network.N))
			for i in range(numInputs):
				for nID in inputs_neuronIDs[i]:
					W_inp[i, nID] = 1.0
					network.neuronLabels[nID] = 'input_'+str(i)
			network.set_input_connectivity(connectivity=W_inp)

			W_synE 	= generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='constant',
													args={
															'distances':network.geometry.distances,
															'k':4,
															'c_w':1.0
													} )
			W_synG 	= generate_connectivity_matrix(N=network.N, adjacencyScheme='distance_threshold', initWeightScheme='constant',
													args={
															'distances':network.geometry.distances,
															'd_thresh':2.0,
															'c_w':0.1
													} )
			network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
			network.set_gapjunction_connectivity(connectivity=W_synG)

			network.initialize_simulation(T_max=500, deltaT=0.1)

			print "start sim"
			while(network.sim_state_valid()):

				network.set_input_vals( vals=[currentInput.val(network.t)] )

				network.sim_step()
			print "end sim"
			
			network_overview_figure(network, synapseDiagram2D=True, gapjunctionDiagram2D=True, spikerateDiagram2D=True)

			pyplot.show()

			# Store the first network, which should have no gap junctions, as a representative network for later reference and visualization:
			if(counter == 1):
				representativeNetwork = network

			print network.spikeTimes()

			exit()

			#~~~~~~~~~~~~~~~~~~~~~~~~
			#~~~~~~~~~~~~~~~~~~~~~~~~

			# networkDataframe 	= network.get_neurons_dataframe()

			NUMSPIKES	= networkDataframe['spike'].sum()
			NUMGAPJNS	= (numpy.count_nonzero(network.connectionWeights_gap)/2) / network.N # divide by 2 to not double count symmetry # divide by N to get gap jns per neuron

			experimentData.append({'d_thresh':_CUR_D_THRESH_, 'c_w':_CUR_C_W_, 'rep':rep, 'numSpikes':NUMSPIKES, 'numGapJns':NUMGAPJNS})

			network_overview_figure(network, synapseDiagram2D=True, gapjunctionDiagram2D=True, spikerateDiagram2D=True)

			pyplot.show()

			exit()

			break
		break

		#~END OF REPS FOR CURRENT D_THRESH, C_W COMBO~

	break

#~END OF EXPERIMENT LOOPS~

 

experimentDataframe	= pandas.DataFrame(experimentData)

experimentDataframe.to_csv('torusSurface_evenlyPositioned_knnSynAdjacency_constantSynWeights_RESULTS.csv', index=False)

avgNumSpikesDataframe = experimentDataframe['numSpikes'].groupby([experimentDataframe['c_w'], experimentDataframe['d_thresh']]).mean().unstack()

experimentFigure	= pyplot.figure(figsize=(6,10))
ax1	= experimentFigure.add_subplot(211)
ax2	= experimentFigure.add_subplot(212)

seaborn.heatmap(data=avgNumSpikesDataframe, annot=False, cmap='hot', ax=ax1)
ax1.invert_yaxis()

seaborn.boxplot(data=experimentDataframe, x='d_thresh', y='numGapJns', color='white', ax=ax2)
# ax2.yaxis.set_label_position("right")

experimentFigure.tight_layout()

pyplot.savefig('LIFNetwork_gapJnFirstTests_'+str(counter)+'combos'+str(numReps)+'reps_results.png', bbox_inches='tight')

pyplot.show()

exit()

spikeTimes	= []
for n in range(network.N):
	spikeTimes.append( testNetworkDataFrame.loc[((testNetworkDataFrame['neuron id'] == n) & (testNetworkDataFrame['spike'] == 1)), 't'].values )
# print spikeTimes

rasterweightFigure	= pyplot.figure(figsize=(12,9))

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
connectionWeights_synExcitAndInhib	= network.ConnectionWeights_synExcit - network.connectionWeights_synInhib
cmap_min	= connectionWeights_synExcitAndInhib.min()
cmap_max	= connectionWeights_synExcitAndInhib.max()
cmap_midpt	= 0.0
ax2.imshow(connectionWeights_synExcitAndInhib, cmap='RdBu', clim=(cmap_min, cmap_max), norm=MidpointNormalize(midpoint=cmap_midpt, vmin=cmap_min, vmax=cmap_max))
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = pyplot.subplot2grid((3, 3), (2, 1))
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

pyplot.show()

