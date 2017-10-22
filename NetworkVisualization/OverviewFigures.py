import numpy as numpy

from matplotlib import pyplot as pyplot
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from NetworkVisualization.NetworkPlots import *



def LIF_overview_figure(neuronsDataFrame):
	simsummaryFigure = pyplot.figure(figsize=(16,10))

	gs1 = gridspec.GridSpec(12, 9)
	gs1.update(left=0.02, right=0.6, wspace=0.05, top=0.98, bottom=0.02, hspace=1.0)
	ax_raster = pyplot.subplot(gs1[0:5, :])
	ax_netSyn = pyplot.subplot(gs1[5:9, 0:3], projection='3d')
	ax_netGap = pyplot.subplot(gs1[5:9, 3:6], projection='3d')
	# ax_netRat = pyplot.subplot(gs1[5:9, 6:9], projection='3d')
	ax_matSyn = pyplot.subplot(gs1[9:12,0:3])
	ax_matGap = pyplot.subplot(gs1[9:12,3:6])
	ax_matInp = pyplot.subplot(gs1[9:12,6:9])

	# return

	#*************************
	# RASTER PLOT
	#*************************

	#*************************
	# SYNAPTIC CONNECTIVITY DIAGRAM
	#*************************

	#*************************
	# SYNAPTIC CONNECTIVITY MATRIX
	#*************************

	#*************************
	# GAP JUNCTION CONNECTIVITY DIAGRAM
	#*************************

	#*************************
	# GAP JUNCTION CONNECTIVITY MATRIX
	#*************************

	#*************************
	# TRACE PLOTS
	#*************************
	neuronsNumSpikes	= neuronsDataFrame.groupby(['neuron_id']).sum().reset_index()['spike'].tolist()
	hiSpikeRateIDs		= list(set(([0] + numpy.asarray(neuronsNumSpikes).argsort()[-6:][::-1].tolist())[:5]))
	loSpikeRateIDs		= numpy.asarray(neuronsNumSpikes).argsort()[:5].tolist()

	neuronsIDs_traces	= hiSpikeRateIDs + loSpikeRateIDs

	gs2 = gridspec.GridSpec(len(neuronsIDs_traces), 1)
	gs2.update(left=0.65, right=0.97, top=0.98, bottom=0.02, hspace=0.15)

	# return

	for i, nID in enumerate(neuronsIDs_traces):
		ax_n = pyplot.subplot(gs2[i, :])
		
		x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
		trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
		trace_ge 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'g_excit'].values, 'label':'g_e', 'color':'blue', 'alpha':0.5, 'linestyle':':'}

		traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_ge], x_axis_label='t', y1_axis_label='Voltage', y2_axis_label='', y1_legend=False, y2_legend=False, fontsize=8, labelsize=6)


	return

	# 	ax_n_1 = ax_n
	# 	ax_n_2	= ax_n_1.twinx()

	# 	ax_n_2.plot(
	# 				neuronDataFrame.loc[(neuronDataFrame['neuron_id'] == nID), 't'].values,
	# 				neuronDataFrame.loc[(neuronDataFrame['neuron_id'] == nID), 'g_excit'].values,
	# 				color='blue', alpha=0.5, linestyle=':'
	# 				)
	# 	ax_n_2.plot(
	# 				neuronDataFrame.loc[(neuronDataFrame['neuron_id'] == nID), 't'].values,
	# 				neuronDataFrame.loc[(neuronDataFrame['neuron_id'] == nID), 'g_inhib'].values,
	# 				color='red', alpha=0.5, linestyle=':'
	# 				)
	# 	ax_n_2.set_ylabel("N"+str(nID)+" cond.", fontsize=8)
	# 	# ax_n_2.set(ylabel="Neur"+str(n)+" conductance")
	# 	ax_n_2.set(ylim=[neuronDataFrame['g_excit'].min(), neuronDataFrame['g_excit'].max()])
	# 	ax_n_2.set(xlim=[0, testNetwork.T_max])
	# 	ax_n_2.grid(False)
	# 	ax_n_2.tick_params(axis='y', which='major', labelsize=6)


	# 	ax_n_1.plot(
	# 				neuronDataFrame.loc[(neuronDataFrame['neuron_id'] == nID), 't'].values,
	# 				neuronDataFrame.loc[(neuronDataFrame['neuron_id'] == nID), 'V'].values,
	# 				color='black', alpha=1.0, linewidth=1
	# 				)
	# 	ax_n_1.set_ylabel("N"+str(nID)+" volt (mv)", fontsize=8)
	# 	# ax_n_1.set(ylabel="Neur"+str(n)+" voltage")
	# 	ax_n_1.set(ylim=[neuronDataFrame['V'].min(), neuronDataFrame['V'].max()])
	# 	ax_n_1.set(xlim=[0, testNetwork.T_max])
	# 	ax_n_1.grid(False)
	# 	ax_n_1.tick_params(axis='y', which='major', labelsize=6)

	# 	if(i == neuronsIDs_traces[-1]):	# if on the last neuronID, label x axis...
	# 		ax_n_1.set(xlabel="t")
	# 		ax_n_2.legend(['g_e', 'g_i'], loc='upper right', fontsize='x-small')
	# 	else:								# else, for all others hide x axis tick labels
	# 		ax_n_1.set_xticklabels([])
	# 		ax_n_2.set_xticklabels([])




	# # gs1.tight_layout()
	# simsummaryFigure.patch.set_facecolor('white')

	# # pyplot.show()

	# spikeTimes	= []
	# for n in range(testNetwork.N):
	# 	spikeTimes.append( neuronDataFrame.loc[((neuronDataFrame['neuron_id'] == n) & (neuronDataFrame['spike'] == 1)), 't'].values )
	# # print spikeTimes

	# # spikeFigure = pyplot.figure(figsize=(10,4))
	# # ax 	= spikeFigure.add_subplot(111)
	# # for n, spike_t in enumerate(spikeTimes):
	# # 	pyplot.vlines(x=spike_t, ymin=n+0.5, ymax=n+1.5)
	# # ax.set_ylim(0.5, len(spikeTimes) + 0.5)
	# # ax.set_yticks(range(1, testNetwork.N+1))
	# # # ax.set_yticks(numpy.arange(0.5, testNetwork.N+0.5, 1.0))
	# # # ax.grid(which='major')
	# # ax.grid(False)
	# # ax.set_xlabel('t')
	# # ax.set_ylabel('neuron_id')
	# # ax.invert_yaxis()
	# # spikeFigure.patch.set_facecolor('white')

	# # print "distances = \n" + str(testNetwork.geometry.distances)

	# # print "W_synE = \n" + str(W_synE)



	# pyplot.savefig('_____.png', bbox_inches='tight')

	# pyplot.show()

	# exit()


