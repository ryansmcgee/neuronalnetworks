import numpy as numpy

from matplotlib import pyplot as pyplot
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from NetworkVisualization.NetworkPlots import *



def LIF_network_overview_figure(network, synapseDiagram2D=False, gapjunctionDiagram2D=False, spikerateDiagram2D=False):
	simsummaryFigure = pyplot.figure(figsize=(16,10))

	gs1 = gridspec.GridSpec(12, 9)
	gs1.update(left=0.02, right=0.6, wspace=0.05, top=0.98, bottom=0.02, hspace=1.0)
	ax_raster = pyplot.subplot(gs1[0:5, :])
	# ax_netSyn = pyplot.subplot(gs1[5:9, 0:3])
	# ax_netGap = pyplot.subplot(gs1[5:9, 3:6])
	ax_netSyn = pyplot.subplot(gs1[5:9, 0:3]) if synapseDiagram2D else pyplot.subplot(gs1[5:9, 0:3], projection='3d')
	ax_netGap = pyplot.subplot(gs1[5:9, 3:6]) if gapjunctionDiagram2D else pyplot.subplot(gs1[5:9, 3:6], projection='3d')
	ax_netRat = pyplot.subplot(gs1[5:9, 6:9]) if spikerateDiagram2D else pyplot.subplot(gs1[5:9, 6:9], projection='3d')
	ax_matSyn = pyplot.subplot(gs1[9:12,0:3])
	ax_matGap = pyplot.subplot(gs1[9:12,3:6])
	ax_matInp = pyplot.subplot(gs1[9:12,6:9])

	# return

	#*************************
	# RASTER PLOT
	#*************************
	spike_raster_plot(ax_raster, network)

	#*************************
	# SYNAPTIC CONNECTIVITY DIAGRAM
	#*************************
	if(synapseDiagram2D):
		synapse_network_diagram_2d(ax_netSyn, network)
	else:
		synapse_network_diagram_3d(ax_netSyn, network)

	#*************************
	# SYNAPTIC CONNECTIVITY MATRIX
	#*************************
	synapse_connectivity_matrix(ax_matSyn, network.connectionWeights_synExcit-network.connectionWeights_synInhib)

	#*************************
	# GAP JUNCTION CONNECTIVITY DIAGRAM
	#*************************
	if(gapjunctionDiagram2D):
		gapjunction_network_diagram_2d(ax_netGap, network)
	else:
		gapjunction_network_diagram_3d(ax_netGap, network)

	#*************************
	# GAP JUNCTION CONNECTIVITY MATRIX
	#*************************
	gapjunction_connectivity_matrix(ax_matGap, network.connectionWeights_gap)

	#*************************
	# AVERAGE RATE NETWORK DIAGRAM
	#*************************
	# rate_network_diagram_2d(ax_netRat, network)
	if(spikerateDiagram2D):
		rate_network_diagram_2d(ax_netRat, network)
	else:
		rate_network_diagram_3d(ax_netRat, network)

	#*************************
	# GAP JUNCTION CONNECTIVITY MATRIX
	#*************************
	input_connectivity_matrix(ax_matInp, network.connectionWeights_inputs)

	#*************************
	# TRACE PLOTS
	#*************************
	neuronsDataFrame 	= network.get_neurons_dataframe()
	neuronsNumSpikes	= neuronsDataFrame.groupby(['neuron_id']).sum().reset_index()['spike'].tolist()
	hiSpikeRateIDs		= list(set(([0] + numpy.asarray(neuronsNumSpikes).argsort()[-6:][::-1].tolist())[:5]))
	loSpikeRateIDs		= numpy.asarray(neuronsNumSpikes).argsort()[:5].tolist()

	neuronIDs_traces	= [0, 1, 2, 10, 11, 12]
	# neuronIDs_traces	= hiSpikeRateIDs + loSpikeRateIDs

	gs2 = gridspec.GridSpec(len(neuronIDs_traces), 1)
	gs2.update(left=0.65, right=0.97, top=0.98, bottom=0.02, hspace=0.15)


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# PLOT CONDUCTANCES VERSION ~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# y1_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].min().min(), neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].max().max()]
	# y2_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['g_excit', 'g_inhib']].min().min(), neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['g_excit', 'g_inhib']].max().max()]
	# for i, nID in enumerate(neuronIDs_traces):
	# 	ax_n = pyplot.subplot(gs2[i, :])

	# 	x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
	# 	trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
	# 	trace_ge 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'g_excit'].values, 'label':'g_e', 'color':'blue', 'alpha':0.5, 'linestyle':':'}

	# 	traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_ge], y1_lim=y1_axlim, y2_lim=y2_axlim,
	# 					x_axis_label='t', y1_axis_label='N'+str(nID)+' Voltage', y2_axis_label='N'+str(nID)+' Conductance',
	# 					y1_legend=False, y2_legend=False, fontsize=8, labelsize=6)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# PLOT CURRENTS VERSION     ~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	y1_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].min().min(),
				neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].max().max()]
	y2_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['I_leak', 'I_excit', 'I_inhib', 'I_gap', 'I_input']].min().min(),
				neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['I_leak', 'I_excit', 'I_inhib', 'I_gap', 'I_input']].max().max()*1.05] #*1.05 is to give a 5% margin at high end to make sure highest currents' lines are visible
	for i, nID in enumerate(neuronIDs_traces):
		ax_n = pyplot.subplot(gs2[i, :])

		x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
		trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
		# trace_Ilk 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_leak'].values, 'label':'I_leak', 'color':'black', 'alpha':1.0, 'linestyle':':'}
		trace_Iex 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_excit'].values, 'label':'I_excit', 'color':'blue', 'alpha':1.0, 'linestyle':':'}
		trace_Iih 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_inhib'].values, 'label':'I_inhib', 'color':'red', 'alpha':1.0, 'linestyle':':'}
		trace_Igp 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_gap'].values, 'label':'I_gap', 'color':'purple', 'alpha':1.0, 'linestyle':':'}
		trace_Iin 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_input'].values, 'label':'I_input', 'color':'green', 'alpha':1.0, 'linestyle':':'}

		traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_Iex, trace_Iih, trace_Igp, trace_Iin], y1_lim=y1_axlim, y2_lim=y2_axlim,
						x_axis_label='t', y1_axis_label='N'+str(nID)+' Voltage', y2_axis_label='N'+str(nID)+' Currents',
						y1_legend=False, y2_legend=False, fontsize=8, labelsize=6)


	gs1.tight_layout(simsummaryFigure, rect=[0, 0, 0.6, 1])
	simsummaryFigure.patch.set_facecolor('white')

	return



def network_overview_figure(network, synapseDiagram2D=False, gapjunctionDiagram2D=False, spikerateDiagram2D=False):
	simsummaryFigure = pyplot.figure(figsize=(16,10))

	gs1 = gridspec.GridSpec(12, 9)
	gs1.update(left=0.02, right=0.6, wspace=0.05, top=0.98, bottom=0.02, hspace=1.0)
	ax_raster = pyplot.subplot(gs1[0:5, :])
	# ax_netSyn = pyplot.subplot(gs1[5:9, 0:3])
	# ax_netGap = pyplot.subplot(gs1[5:9, 3:6])
	ax_netSyn = pyplot.subplot(gs1[5:9, 0:3]) if synapseDiagram2D else pyplot.subplot(gs1[5:9, 0:3], projection='3d')
	ax_netGap = pyplot.subplot(gs1[5:9, 3:6]) if gapjunctionDiagram2D else pyplot.subplot(gs1[5:9, 3:6], projection='3d')
	ax_netRat = pyplot.subplot(gs1[5:9, 6:9]) if spikerateDiagram2D else pyplot.subplot(gs1[5:9, 6:9], projection='3d')
	ax_matSyn = pyplot.subplot(gs1[9:12,0:3])
	ax_matGap = pyplot.subplot(gs1[9:12,3:6])
	ax_matInp = pyplot.subplot(gs1[9:12,6:9])

	# return

	#*************************
	# RASTER PLOT
	#*************************
	spike_raster_plot(ax_raster, network)

	#*************************
	# SYNAPTIC CONNECTIVITY DIAGRAM
	#*************************
	if(synapseDiagram2D):
		synapse_network_diagram_2d(ax_netSyn, network)
	else:
		synapse_network_diagram_3d(ax_netSyn, network)

	#*************************
	# SYNAPTIC CONNECTIVITY MATRIX
	#*************************
	synapse_connectivity_matrix(ax_matSyn, network.connectionWeights_synExcit-network.connectionWeights_synInhib)

	#*************************
	# GAP JUNCTION CONNECTIVITY DIAGRAM
	#*************************
	if(gapjunctionDiagram2D):
		gapjunction_network_diagram_2d(ax_netGap, network)
	else:
		gapjunction_network_diagram_3d(ax_netGap, network)

	#*************************
	# GAP JUNCTION CONNECTIVITY MATRIX
	#*************************
	gapjunction_connectivity_matrix(ax_matGap, network.connectionWeights_gap)

	#*************************
	# AVERAGE RATE NETWORK DIAGRAM
	#*************************
	# rate_network_diagram_2d(ax_netRat, network)
	if(spikerateDiagram2D):
		rate_network_diagram_2d(ax_netRat, network)
	else:
		rate_network_diagram_3d(ax_netRat, network)

	#*************************
	# GAP JUNCTION CONNECTIVITY MATRIX
	#*************************
	input_connectivity_matrix(ax_matInp, network.connectionWeights_inputs)

	#*************************
	# TRACE PLOTS
	#*************************
	neuronsDataFrame 	= network.get_neurons_dataframe()
	neuronsNumSpikes	= neuronsDataFrame.groupby(['neuron_id']).sum().reset_index()['spike'].tolist()
	hiSpikeRateIDs		= list(set(([0] + numpy.asarray(neuronsNumSpikes).argsort()[-6:][::-1].tolist())[:5]))
	loSpikeRateIDs		= numpy.asarray(neuronsNumSpikes).argsort()[:5].tolist()

	neuronIDs_traces	= [55, 65, 54, 56, 45, 53, 57, 64, 46, 0]
	# neuronIDs_traces	= hiSpikeRateIDs + loSpikeRateIDs

	gs2 = gridspec.GridSpec(len(neuronIDs_traces), 1)
	gs2.update(left=0.65, right=0.97, top=0.98, bottom=0.02, hspace=0.15)


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# PLOT CONDUCTANCES VERSION ~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# y1_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].min().min(), neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].max().max()]
	# y2_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['g_excit', 'g_inhib']].min().min(), neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['g_excit', 'g_inhib']].max().max()]
	# for i, nID in enumerate(neuronIDs_traces):
	# 	ax_n = pyplot.subplot(gs2[i, :])

	# 	x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
	# 	trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
	# 	trace_ge 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'g_excit'].values, 'label':'g_e', 'color':'blue', 'alpha':0.5, 'linestyle':':'}

	# 	traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_ge], y1_lim=y1_axlim, y2_lim=y2_axlim,
	# 					x_axis_label='t', y1_axis_label='N'+str(nID)+' Voltage', y2_axis_label='N'+str(nID)+' Conductance',
	# 					y1_legend=False, y2_legend=False, fontsize=8, labelsize=6)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# PLOT CURRENTS VERSION     ~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	y1_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].min().min(),
				neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['V']].max().max()]
	y2_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['I_excit', 'I_inhib', 'I_gap', 'I_input']].min().min(),
				neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_traces)), ['I_excit', 'I_inhib', 'I_gap', 'I_input']].max().max()*1.05] #*1.05 is to give a 5% margin at high end to make sure highest currents' lines are visible
	for i, nID in enumerate(neuronIDs_traces):
		ax_n = pyplot.subplot(gs2[i, :])

		x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
		trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
		trace_Iex 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_excit'].values, 'label':'I_excit', 'color':'blue', 'alpha':1.0, 'linestyle':':'}
		trace_Iih 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_inhib'].values, 'label':'I_inhib', 'color':'red', 'alpha':1.0, 'linestyle':':'}
		trace_Igp 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_gap'].values, 'label':'I_gap', 'color':'purple', 'alpha':1.0, 'linestyle':':'}
		trace_Iin 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_input'].values, 'label':'I_input', 'color':'green', 'alpha':1.0, 'linestyle':':'}

		traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_Iex, trace_Iih, trace_Igp, trace_Iin], y1_lim=y1_axlim, y2_lim=y2_axlim,
						x_axis_label='t', y1_axis_label='N'+str(nID)+' Voltage', y2_axis_label='N'+str(nID)+' Currents',
						y1_legend=False, y2_legend=False, fontsize=8, x_labelsize=(0 if i<(len(neuronIDs_traces)-1) else 6) )

		spikeTimes 	= network.get_spike_times()
		ax_n.scatter(x=spikeTimes[nID], y=neuronsDataFrame.loc[((neuronsDataFrame['neuron_id']==nID)&(neuronsDataFrame['t'].isin(spikeTimes[nID]))), 'V'].values, marker='^', c='k', edgecolors='none')
		# exit()
		


	gs1.tight_layout(simsummaryFigure, rect=[0, 0, 0.6, 1])
	simsummaryFigure.patch.set_facecolor('white')

	return



