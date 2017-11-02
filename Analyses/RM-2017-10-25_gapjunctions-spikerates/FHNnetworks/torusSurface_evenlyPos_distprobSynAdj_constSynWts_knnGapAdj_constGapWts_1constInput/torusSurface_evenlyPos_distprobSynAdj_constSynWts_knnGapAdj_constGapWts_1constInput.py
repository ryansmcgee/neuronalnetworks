from __future__ import division

import numpy as numpy
import pandas as pandas
# import seaborn as seaborn
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg') # For running matplotlib through ssh. Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as pyplot
import time

import sys
# sys.path.append('/Users/ryan/Projects/neuronal-networks')
sys.path.append('../../../..')
from NetworkModels.FHNNetwork import FHNNetwork
from NetworkGeometry.TorusSurface import TorusSurface
from NetworkConnectivity.NetworkConnectivity import *
from NetworkInput.ConstantInput import ConstantInput
from NetworkVisualization.NetworkPlots import *
from NetworkVisualization.OverviewFigures import *

numpy.random.seed(69000)

experimentData	= []

counter = 0

numReps	= 1#5
k_vals = [3]#[0,1,2,3,4,5,6,7,8]
c_vals = numpy.arange(0.0, 1.0, 0.05)
for idx_k, _CUR_GAP_K_ in enumerate(k_vals):
	for idx_c, _CUR_GAP_C_W_ in enumerate(c_vals):
		for rep in range(numReps):
			counter += 1

			print "\n\n=============================="+str(counter)+"/"+str(len(k_vals)*len(c_vals)*numReps)+"\nk = "+str(_CUR_GAP_K_)+"  c_w = "+str(_CUR_GAP_C_W_)+"  rep = " + str(rep+1)+"/"+str(numReps)
			repStartTime = time.time()

			network 	= FHNNetwork()

			network.geometry = TorusSurface(w=10, h=10)

			N_excit = 100
			network.add_neurons(numNeuronsToAdd=N_excit,
								V_init=-1.1994, V_peak=1.45, V_eqExcit=2.0, V_eqInhib=-70.0, 
								W_init=-0.6243, a=0.7, b=0.8,
								g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5, tau_g_excit=2.0, tau_g_inhib=2.0, tau_W=12.5,
								synapse_type='excitatory', label='')

			neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
			neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]

			network.geometry.position_neurons(positioning='even')

			numInputs = 1
			currentInput = ConstantInput(constVal=1.0)
			network.add_inputs(numInputsToAdd=numInputs)
			inputs_neuronIDs	= [[55]]
			W_inp = numpy.zeros(shape=(numInputs, network.N))
			for i in range(numInputs):
				for nID in inputs_neuronIDs[i]:
					W_inp[i, nID] = 1.0
					network.neuronLabels[nID] = 'input_'+str(i)
			network.set_input_connectivity(connectivity=W_inp)

			W_synE 	= generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='constant',
													args={
															'distances':network.geometry.distances,
															'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.9,
															'c_w':0.5
													} )
			W_synG 	= generate_connectivity_matrix(N=network.N, adjacencyScheme='nearest_neighbors', initWeightScheme='constant',
													args={
															'distances':network.geometry.distances,
															'k':_CUR_GAP_K_,
															'c_w':_CUR_GAP_C_W_
													} )
			network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
			network.set_gapjunction_connectivity(connectivity=W_synG)

			network.initialize_simulation(T_max=10, deltaT=0.5)

			simStartTime = time.time()
			while(network.sim_state_valid()):

				network.set_input_vals( vals=[currentInput.val(network.t)] )

				network.sim_step()
			simEndTime = time.time()
			print "  sim time: " +str(simEndTime - simStartTime)
			
			

			#~~~~~~~~~~~~~~~~~~~~~~~~
			#~~~~~~~~~~~~~~~~~~~~~~~~

			# This calculates the average spike rate of a neuron (in Hz, if the time unit is ms (1000 time units = 1sec)):
			# - Calculates rate based only on 2nd half of the simulation, assuming this is a more stable state of network activity than the beginning of the sim
			# - (# spikes in 2nd half of sim)/N = average # spikes per neuron in 2nd half of sim
			# - avg # spikes per neuron * 1000/(T_max/2) = avg # spikes per neuron per 1000 time units (1sec if time unit is ms)
			SPIKERATE 	= (sum(sum(spikeEvents_n[int(len(spikeEvents_n)/2):]) for spikeEvents_n in network.neuronLogs['spike']['data'])/network.N) * (1000/(network.T_max/2))

			NUMGAPJNS	= (numpy.count_nonzero(network.connectionWeights_gap)/2) / network.N # divide by 2 to not double count symmetry # divide by N to get gap jns per neuron

			experimentData.append({'k':_CUR_GAP_K_, 'c_w':_CUR_GAP_C_W_, 'rep':rep, 'spikeRate':SPIKERATE, 'numGapJns':NUMGAPJNS})


			repEndTime = time.time()
			print "  rep time: " + str(repEndTime - repStartTime)
			#~~~~~~~~~~~~~~~~~~~~~~~~
			# Store a representative for later reference and visulatization:
			# - choose a network that has intermediate gap junction sparsity and low non-zero gap weights 
			#.  so that gap junctions appear but characteristic gap-junction-free spiking behavior can be seen.
			if(_CUR_GAP_K_ == numpy.median(k_vals) and _CUR_GAP_C_W_ == numpy.sort(c_vals)[1] and rep == 0):
				representativeNetwork = network

			# axx = pyplot.subplot(projection='3d')
			# synapse_network_diagram_3d(axx, network)

			network_overview_figure(network, synapseDiagram2D=False, gapjunctionDiagram2D=False, spikerateDiagram2D=False)



			pyplot.show()

			exit()



		# 	break
		# break

		#~END OF REPS FOR CURRENT D_THRESH, C_W COMBO~

	# break

#~END OF EXPERIMENT LOOPS~

# print experimentData
# exit()
 
# import seaborn as seaborn

experimentDataframe	= pandas.DataFrame(experimentData)

experimentDataframe.to_csv('FHN_torusSurface_evenlyPos_distprobSynAdj_constSynWts_knnGapAdj_constGapWts_1constInput_Tmax'+str(representativeNetwork.T_max)+'_'+str(numReps)+'reps_RESULTS.csv', index=False)

spikeRateAvgOverRepsDataframe = experimentDataframe['spikeRate'].groupby([experimentDataframe['c_w'], experimentDataframe['k']]).mean().unstack()


experimentFigure 	= pyplot.figure(figsize=(16,4))
ax_netSyn 			= experimentFigure.add_subplot(141)
ax_netGap 			= experimentFigure.add_subplot(142)
ax_netRate 			= experimentFigure.add_subplot(143)
ax_avgRateHeatmap 	= experimentFigure.add_subplot(144)

synapse_network_diagram_2d(ax_netSyn, representativeNetwork)

gapjunction_network_diagram_2d(ax_netGap, representativeNetwork)

rate_network_diagram_2d(ax_netRate, representativeNetwork)
ax_netRate.set_title("Typical No-Gap-Jn Spike Rates", {'fontsize':12})

import seaborn as seaborn
seaborn.heatmap(data=spikeRateAvgOverRepsDataframe, annot=False, cmap='hot', ax=ax_avgRateHeatmap, yticklabels=(True if len(c_vals)<10 else int(len(c_vals)/2)-1), xticklabels=(True if len(k_vals)<10 else int(len(k_vals)/2)-1) )
ax_avgRateHeatmap.invert_yaxis()
ax_avgRateHeatmap.set_title("Average Spike Rates")
ax_avgRateHeatmap.set_ylabel("w (constant gap jn weight)")
ax_avgRateHeatmap.set_xlabel("k (gap jn KNN connectivity)")
# ax_avgRateHeatmap.yticklabels([numpy.min(c_vals), numpy.max(c_vals)])
# ax_avgRateHeatmap.xticklabels([numpy.min(k_vals), numpy.max(k_vals)])

experimentFigure.tight_layout()

pyplot.savefig('FHN_torusSurface_evenlyPos_distprobSynAdj_constSynWts_knnGapAdj_constGapWts_1constInput_Tmax'+str(representativeNetwork.T_max)+'_'+str(numReps)+'reps_RESULTS.png', bbox_inches='tight')

# pyplot.show()

#end