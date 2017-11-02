import numpy as numpy

import matplotlib.pyplot as pyplot
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection




from vizutil import *


def synapse_connectivity_colormap(connectivityMatrix):
	maxWt = numpy.max(connectivityMatrix)
	minWt = numpy.min(connectivityMatrix)
	synWt_cmap = zero_midpoint_cmap(orig_cmap=matplotlib.cm.get_cmap('RdBu'), min_val=minWt, max_val=maxWt)
	# synWt_cmap.set_bad(color='white') # This forces a color for bad values and/or all-0s matrices (when all-0s matrices are fully masked as bad) (as opposed to the center of colormap being used in these cases)
	return synWt_cmap

def gapjunction_connectivity_colormap(connectivityMatrix):
	gapWt_cmap = matplotlib.cm.get_cmap('Purples')
	gapWt_cmap.set_bad(color='white') # This forces a color for bad values and/or all-0s matrices (when all-0s matrices are fully masked as bad) (as opposed to the center of colormap being used in these cases)
	return gapWt_cmap

def input_connectivity_colormap(connectivityMatrix):
	inpWt_cmap = matplotlib.cm.get_cmap('Greens')
	inpWt_cmpa = inpWt_cmap.set_bad(color='white') # This forces a color for bad values and/or all-0s matrices (when all-0s matrices are fully masked as bad) (as opposed to the center of colormap being used in these cases)
	return inpWt_cmap


# def traces_plot(ax, x_data, y1_data=None, y1_labels=None, y1_colors=None, y1_alphas=None, y1_linestyles=None, y2_data=None, y2_labels=None, y2_colors=None, y2_alphas=None, x_axis_label='', y_axis1_label='', fontsize=8, labelsize=6):
def traces_plot(ax, x, y1_traces=None, y2_traces=None, x_lim=None, y1_lim=None, y2_lim=None,
					x_axis_label='', y1_axis_label='', y2_axis_label='', y1_legend=False, y2_legend=False,
					fontsize=8, x_labelsize=6, y1_labelsize=6, y2_labelsize=6):

	yAx1 = ax
	yAx2 = yAx1.twinx()

	#~~~~~~~~~~~~~~~~~~~~~~
	# Plot y-axis-2 traces:
	#~~~~~~~~~~~~~~~~~~~~~~
	if(y2_traces is not None):
		y2_minVal = float("inf")
		y2_maxVal = -1*float("inf")
		y2_allLabels = []
		for i, trace in enumerate(y2_traces):
			#~~~~~~~~~~~~~~~~~~~~
			# Parse the trace data and parameters from the traces argument dictionary, setting defaults for non-specified parameters:
			#~~~~~~~~~~~~~~~~~~~~
			traceArgs = trace.keys()

			if('data' not in traceArgs):
				print "Trace Plot Error: Argument 'traces' expects dictionary with required key 'data' that holds the data series to be plotted."
				return
			else:
				data 		= trace['data']
				# Update the axis min and max values as necessary:
				y2_minVal = min(data) if (min(data) < y2_minVal) else y2_minVal
				y2_maxVal = max(data) if (max(data) > y2_maxVal) else y2_maxVal

			label 		= trace['label'] if ('label' in traceArgs) else ''
			y2_allLabels.append(label)

			color 		= trace['color'] if ('color' in traceArgs) else 'black'
			alpha 		= trace['alpha'] if ('alpha' in traceArgs) else '1.0'
			linestyle 	= trace['linestyle'] if ('linestyle' in traceArgs) else '-'

			yAx2.plot(x, data, color=color, alpha=alpha, linestyle=linestyle)

		yAx2.set_ylabel(y2_axis_label, fontsize=fontsize)
		yAx2.set(ylim=(y2_lim if y2_lim is not None else [y2_minVal, y2_maxVal]))
		# yAx2.set(xlim=(x_lim if x_lim is not None else [0, max(x)])) # set along with y1_axis
		yAx2.grid(False)
		yAx2.tick_params(axis='y', which='major', labelsize=y2_labelsize)

		if(y2_legend):
			yAx2.legend(y2_allLabels, loc='upper right', fontsize='x-small')

	#~~~~~~~~~~~~~~~~~~~~~~
	# Plot y-axis-1 traces:
	#~~~~~~~~~~~~~~~~~~~~~~
	if(y1_traces is not None):
		y1_minVal = float("inf")
		y1_maxVal = -1*float("inf")
		y1_allLabels = []
		for i, trace in enumerate(y1_traces):
			#~~~~~~~~~~~~~~~~~~~~
			# Parse the trace data and parameters from the traces argument dictionary, setting defaults for non-specified parameters:
			#~~~~~~~~~~~~~~~~~~~~
			traceArgs = trace.keys()

			if('data' not in traceArgs):
				print "Trace Plot Error: Argument 'traces' expects dictionary with required key 'data' that holds the data series to be plotted."
				return
			else:
				data 		= trace['data']
				# Update the axis min and max values as necessary:
				y1_minVal = min(data) if (min(data) < y1_minVal) else y1_minVal
				y1_maxVal = max(data) if (max(data) > y1_maxVal) else y1_maxVal

			label 		= trace['label'] if ('label' in traceArgs) else ''
			y1_allLabels.append(label)

			color 		= trace['color'] if ('color' in traceArgs) else 'black'
			alpha 		= trace['alpha'] if ('alpha' in traceArgs) else '1.0'
			linestyle 	= trace['linestyle'] if ('linestyle' in traceArgs) else '-'


			yAx1.plot(x, data, color=color, alpha=alpha, linestyle=linestyle)

		yAx1.set_ylabel(y1_axis_label, fontsize=fontsize)
		yAx1.set(ylim=(y1_lim if y1_lim is not None else [y1_minVal, y1_maxVal]))
		yAx1.set(xlim=(x_lim if x_lim is not None else [0, max(x)]))
		yAx1.grid(False)
		yAx1.tick_params(axis='y', which='major', labelsize=y1_labelsize)

		if(x_labelsize > 0):
			yAx1.set_xlabel('t', fontsize=fontsize)
			yAx1.tick_params(axis='x', which='major', labelsize=x_labelsize)
		else:
			yAx1.set_xticklabels([])


		if(y1_legend):
			yAx1.legend(y1_allLabels, loc='upper right', fontsize='x-small')

	#~~~~~
	return ax


def spike_raster_plot(ax, network):
	spikeTimes 	= network.get_spike_times()
	for n, spike_t in enumerate(spikeTimes):
		ax.vlines(x=spike_t, ymin=n+0.5, ymax=n+1.5)
	ax.set_ylim(0.5, len(spikeTimes) + 0.5)
	ax.set_xlim(0, network.T_max)
	ax.set_yticks([] if network.N > 50 else range(1, network.N+1))
	ax.set_xticks([0, network.T_max])
	ax.grid(False)
	ax.set_xlabel('t')
	ax.set_ylabel('Neuron Spikes')
	ax.invert_yaxis()


def synapse_network_diagram_2d(ax, network, showAxes=False):

	try:
		neuronCoords = network.geometry.surfacePlaneCoords
	except AttributeError:
		print("The "+str(network.geometry.geometry)+" geometry class does not define a 2D surface plane coordinate system for neuron positions.")
		return

	synapseWeights = network.connectionWeights_synExcit - network.connectionWeights_synInhib



	w = network.geometry.w
	h = network.geometry.h

	try:
		torroidal = network.geometry.torroidal
	except AttributeError:
		torroidal = [False, False]

	surfaceBorderColor 		= (0.7, 0.7, 0.7)
	surfaceBackgroundColor 	= (0.99, 0.99, 0.99)
	ax.add_patch(patches.Rectangle((0,0), w, h, facecolor=surfaceBackgroundColor, edgecolor='none', zorder=0))
	ax.plot([0,0], [0,h], color=surfaceBorderColor, linestyle=(':' if torroidal[0] else '-'), zorder=1)
	ax.plot([w,w], [0,h], color=surfaceBorderColor, linestyle=(':' if torroidal[0] else '-'), zorder=1)
	ax.plot([0,w], [0,0], color=surfaceBorderColor, linestyle=(':' if torroidal[1] else '-'), zorder=1)
	ax.plot([0,w], [h,h], color=surfaceBorderColor, linestyle=(':' if torroidal[1] else '-'), zorder=1)

	maxWt = numpy.max(synapseWeights)
	minWt = numpy.min(synapseWeights)
	synWt_cmap = zero_midpoint_cmap(orig_cmap=matplotlib.cm.get_cmap('RdBu'), min_val=minWt, max_val=maxWt)

	for i in range(network.N):
		for j in range(network.N):
			presynCoord 	= neuronCoords[i] #+ [1, 0]
			postsynCoord 	= neuronCoords[j] #+ [1, 0]
			synWt 			= synapseWeights[i,j]

			if(synWt != 0.0):

				edgeIsTorroidal_w 	= torroidal[0] and abs(postsynCoord[0]-presynCoord[0])>(network.geometry.w/2)
				edgeIsTorroidal_h 	= torroidal[1] and abs(postsynCoord[1]-presynCoord[1])>(network.geometry.h/2)

				edgeColor = synWt_cmap( abs(synWt-minWt)/abs(maxWt-minWt)  ) #numpy.random.rand(3,1)

				if(not edgeIsTorroidal_w and not edgeIsTorroidal_h):
					# Plot the edge without torroidal wrapping:
					ax.annotate("", xytext=presynCoord, xy=postsynCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					pass

				elif(edgeIsTorroidal_w and not edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in w dimension:
					torroidalDist_w = w - ( max(presynCoord[0], postsynCoord[0]) - min(presynCoord[0], postsynCoord[0]) )
					delta_h 		= postsynCoord[1] - presynCoord[1]
					if(presynCoord[0] > postsynCoord[0]):
						# Configure edge segments wrapping across right-hand boundary:
						seg1Endpt 	= [network.geometry.w, presynCoord[1]+((network.geometry.w-presynCoord[0])/torroidalDist_w)*delta_h]
						seg2Startpt = [0, presynCoord[1]+((network.geometry.w-presynCoord[0])/torroidalDist_w)*delta_h]
					else:
						# Configure edge segments wrapping across left-hand boundary:
						seg1Endpt 	= [0, postsynCoord[1]-(abs(w-postsynCoord[0])/torroidalDist_w)*delta_h]
						seg2Startpt = [network.geometry.w, postsynCoord[1]-(abs(w-postsynCoord[0])/torroidalDist_w)*delta_h]
					if(seg2Startpt[0] == postsynCoord[0]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=presynCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=presynCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postsynCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)

				elif(not edgeIsTorroidal_w and edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in h dimension:
					torroidalDist_h = h - ( max(presynCoord[1], postsynCoord[1]) - min(presynCoord[1], postsynCoord[1]) )
					delta_w 		= postsynCoord[0] - presynCoord[0]
					if(presynCoord[1] > postsynCoord[1]):
						# Configure edge segments wrapping across top boundary:
						seg1Endpt 	= [presynCoord[0]+((network.geometry.h-presynCoord[1])/torroidalDist_h)*delta_w, network.geometry.h]
						seg2Startpt = [presynCoord[0]+((network.geometry.h-presynCoord[1])/torroidalDist_h)*delta_w, 0]
					else:
						# Configure edge segments wrapping across left-hand boundary:
						seg1Endpt 	= [postsynCoord[0]-(abs(h-postsynCoord[1])/torroidalDist_h)*delta_w, 0]
						seg2Startpt = [postsynCoord[0]-(abs(h-postsynCoord[1])/torroidalDist_h)*delta_w, h]
					if(seg2Startpt[0] == postsynCoord[0]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=presynCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=presynCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postsynCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)

				elif(edgeIsTorroidal_w and edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in w and h dimensions:
					torroidalDist_w = w - ( max(presynCoord[0], postsynCoord[0]) - min(presynCoord[0], postsynCoord[0]) )
					torroidalDist_h = h - ( max(presynCoord[1], postsynCoord[1]) - min(presynCoord[1], postsynCoord[1]) )
					delta_w 		= postsynCoord[0] - presynCoord[0]
					delta_h 		= postsynCoord[1] - presynCoord[1]
					m_edge					= abs(torroidalDist_h/torroidalDist_w)	#absolute value of slope (torroidal) between pre and post synaptic neurons
					m_presynToNearestCorner	= abs( min(h-presynCoord[1], presynCoord[1]-0) / min(w-presynCoord[0], presynCoord[0]-0) ) 	# absolute value of slope between presynaptic neuron and the nearest boundary corner

					if(presynCoord[0]>postsynCoord[0] and presynCoord[1]>postsynCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across TOP boundary and then wrap around the RIGHT boundary:
							# (segment 1 exits on top boundary, segment 2 reenters from left boundary)
							seg1Endpt	= [presynCoord[0]+((h-presynCoord[1])/m_edge), h]
							seg2Startpt	= [0, postsynCoord[1]-((postsynCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across RIGHT boundary and then wrap around the TOP boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from bottom boundary)
							seg1Endpt	= [w, presynCoord[1]+((w-presynCoord[0])*m_edge)]
							seg2Startpt	= [postsynCoord[0]-((postsynCoord[1])/m_edge), 0]
					elif(presynCoord[0]<postsynCoord[0] and presynCoord[1]>postsynCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across TOP boundary and then wrap around the LEFT boundary:
							# (segment 1 exits on top boundary, segment 2 reenters from right boundary)
							seg1Endpt	= [presynCoord[0]-((h-presynCoord[1])/m_edge), h]
							seg2Startpt	= [w, postsynCoord[1]-((w-postsynCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across LEFT boundary and then wrap around the TOP boundary:
							# (segment 1 exits on left boundary, segment 2 reenters from bottom boundary)
							seg1Endpt	= [0, presynCoord[1]+((presynCoord[0])*m_edge)]
							seg2Startpt	= [postsynCoord[0]+((postsynCoord[1])/m_edge), 0]
					elif(presynCoord[0]>postsynCoord[0] and presynCoord[1]<postsynCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across BOTTOM boundary and then wrap around the RIGHT boundary:
							# (segment 1 exits on bottom boundary, segment 2 reenters from left boundary)
							seg1Endpt	= [presynCoord[0]+((h-presynCoord[1])/m_edge), 0]
							seg2Startpt	= [0, postsynCoord[1]+((postsynCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across RIGHT boundary and then wrap around the BOTTOM boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from top boundary)
							seg1Endpt	= [w, presynCoord[1]-((w-presynCoord[0])*m_edge)]
							seg2Startpt	= [postsynCoord[0]-((postsynCoord[1])/m_edge), h]
					elif(presynCoord[0]<postsynCoord[0] and presynCoord[1]<postsynCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across BOTTOM boundary and then wrap around the LEFT boundary:
							# (segment 1 exits on bottom boundary, segment 2 reenters from right boundary)
							seg1Endpt	= [presynCoord[0]-((h-presynCoord[1])/m_edge), 0]
							seg2Startpt	= [w, postsynCoord[1]+((w-postsynCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across LEFT boundary and then wrap around the BOTTOM boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from top boundary)
							seg1Endpt	= [0, presynCoord[1]-((presynCoord[0])*m_edge)]
							seg2Startpt	= [postsynCoord[0]+((postsynCoord[1])/m_edge), h]
					else:
						# Should not reach else condition:
						pass

					if(seg2Startpt[0] == postsynCoord[0] and seg2Startpt[1] == postsynCoord[1]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=presynCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=presynCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postsynCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)

				else:
					# Should not reach else condition:
					pass

	neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
	neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
	neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
	neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
	neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
	neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])

	# ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='black', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='black', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='limegreen', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	# ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='limegreen', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='darkorange', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='darkorange', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)

	ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='midnightblue', edgecolors='none', zorder=3)
	ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='maroon', edgecolors='none', zorder=3)
	ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='limegreen', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='limegreen', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)
	ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='darkorange', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)
	ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='darkorange', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)

	# ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='blue', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='red', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='blue', edgecolors='limegreen', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	# ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='red', edgecolors='limegreen', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='blue', edgecolors='darkorange', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='blue', edgecolors='darkorange', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)

	ax.set_aspect('equal', 'datalim')

	margin_w = 0.02*w
	margin_h = 0.02*h
	ax.set_xlim(0-margin_w, w+margin_w)
	ax.set_ylim(0-margin_h, h+margin_h)

	ax.spines['bottom'].set_color('white')
	ax.spines['top'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.spines['right'].set_color('white')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	if(showAxes):
		ax.tick_params(axis='both', which='major', labelsize=6)
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	ax.set_title("Synaptic Connectivity", {'fontsize':12})

	#~~~~~
	return ax


def synapse_network_diagram_3d(ax, network):

	# ax = pyplot.axes(projection='3d')

	synapseWeights 	= network.connectionWeights_synExcit - network.connectionWeights_synInhib

	neuronCoords 	= network.geometry.cartesianCoords

	maxWt = numpy.max(synapseWeights)
	minWt = numpy.min(synapseWeights)
	synWt_cmap = zero_midpoint_cmap(orig_cmap=matplotlib.cm.get_cmap('RdBu'), min_val=minWt, max_val=maxWt)

	neuronSynEndpts	= []
	neuronSynWts	= []
	for i in range(network.N):
		for j in range(network.N):
			if(synapseWeights[i,j] != 0.0):
				neuronSynEndpts.append( [neuronCoords[i], neuronCoords[j]] )
				neuronSynWts.append(synapseWeights[i,j])

	edgesSyn = Line3DCollection(neuronSynEndpts, cmap=synWt_cmap)
	edgesSyn.set_array(numpy.asarray(neuronSynWts)) # color the segments by our parameter
	ax.add_collection3d(edgesSyn)

	neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
	neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
	neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
	neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
	neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
	neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])

	ax.scatter(xs=neuronCoords[neuronIDs_nonIOExcit,0], ys=neuronCoords[neuronIDs_nonIOExcit,1], zs=neuronCoords[neuronIDs_nonIOExcit,2], marker='o', c='midnightblue', edgecolors='none', zorder=3, depthshade=True)
	ax.scatter(xs=neuronCoords[neuronIDs_nonIOInhib,0], ys=neuronCoords[neuronIDs_nonIOInhib,1], zs=neuronCoords[neuronIDs_nonIOInhib,2], marker='o', c='maroon', edgecolors='none', zorder=3, depthshade=True)
	ax.scatter(xs=neuronCoords[neuronIDs_inputExcit,0], ys=neuronCoords[neuronIDs_inputExcit,1], zs=neuronCoords[neuronIDs_inputExcit,2], marker='^', c='limegreen', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	ax.scatter(xs=neuronCoords[neuronIDs_inputInhib,0], ys=neuronCoords[neuronIDs_inputInhib,1], zs=neuronCoords[neuronIDs_inputInhib,2], marker='^', c='limegreen', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)
	ax.scatter(xs=neuronCoords[neuronIDs_outputExcit,0], ys=neuronCoords[neuronIDs_outputExcit,1], zs=neuronCoords[neuronIDs_outputExcit,2], marker='s', c='darkorange', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)
	ax.scatter(xs=neuronCoords[neuronIDs_outputInhib,0], ys=neuronCoords[neuronIDs_outputInhib,1], zs=neuronCoords[neuronIDs_outputInhib,2], marker='s', c='darkorange', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)

	# Create cubic bounding box to simulate equal aspect ratio
	largest_axis_range = numpy.array([neuronCoords[:,0].max()-neuronCoords[:,0].min(), neuronCoords[:,1].max()-neuronCoords[:,1].min(), neuronCoords[:,2].max()-neuronCoords[:,2].min()]).max()
	Xb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(neuronCoords[:,0].max()+neuronCoords[:,0].min())
	Yb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(neuronCoords[:,1].max()+neuronCoords[:,1].min())
	Zb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(neuronCoords[:,2].max()+neuronCoords[:,2].min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')

	axesGridColor	= (0.95, 0.95, 0.95, 1.0)
	axesPaneColor 	= (0.99, 0.99, 0.99, 1.0)
	ax.w_zaxis.line.set_lw(0.)
	# ax.w_xaxis.set_pane_color(axesPaneColor)
	# ax.w_yaxis.set_pane_color(axesPaneColor)
	# ax.w_zaxis.set_pane_color(axesPaneColor)
	# ax.w_xaxis._axinfo.update({'grid' : {'color': axesGridColor}})
	# ax.w_yaxis._axinfo.update({'grid' : {'color': axesGridColor}})
	# ax.w_zaxis._axinfo.update({'grid' : {'color': axesGridColor}})

	ax.autoscale(tight=True)

	#~~~~~
	return ax


def gapjunction_network_diagram_2d(ax, network, showAxes=False):

	try:
		neuronCoords = network.geometry.surfacePlaneCoords
	except AttributeError:
		print("The "+str(network.geometry.geometry)+" geometry class does not define a 2D surface plane coordinate system for neuron positions.")
		return

	gapjnWeights = network.connectionWeights_gap

	w = network.geometry.w
	h = network.geometry.h

	try:
		torroidal = network.geometry.torroidal
	except AttributeError:
		torroidal = [False, False]

	surfaceBorderColor 		= (0.7, 0.7, 0.7)
	surfaceBackgroundColor 	= (0.99, 0.99, 0.99)
	ax.add_patch(patches.Rectangle((0,0), w, h, facecolor=surfaceBackgroundColor, edgecolor='none', zorder=0))
	ax.plot([0,0], [0,h], color=surfaceBorderColor, linestyle=(':' if torroidal[0] else '-'), zorder=1)
	ax.plot([w,w], [0,h], color=surfaceBorderColor, linestyle=(':' if torroidal[0] else '-'), zorder=1)
	ax.plot([0,w], [0,0], color=surfaceBorderColor, linestyle=(':' if torroidal[1] else '-'), zorder=1)
	ax.plot([0,w], [h,h], color=surfaceBorderColor, linestyle=(':' if torroidal[1] else '-'), zorder=1)

	maxWt = numpy.max(gapjnWeights)
	minWt = numpy.min(gapjnWeights)
	# gapWt_cmap = zero_midpoint_cmap(orig_cmap=matplotlib.cm.get_cmap('RdBu'), min_val=minWt, max_val=maxWt)
	gapWt_cmap = matplotlib.cm.get_cmap('Purples')

	edgesPlotted = []
	for i in range(network.N):
		for j in range(network.N):
			pregapCoord 	= neuronCoords[i] #+ [1, 0]
			postgapCoord 	= neuronCoords[j] #+ [1, 0]
			gapWt 			= gapjnWeights[i,j]

			if(gapWt != 0.0):
				# Gap junctions are always reciprocal (as implemented for now), but we don't need to draw each gap junction edge twice.
				# - If we've already plotted a gap junction edge between neurons j and i, then we don't need to plot the same one between i and j:
				if((j,i) in edgesPlotted):
					# Move on to the next neuron pair:
					continue
				else:
					# Continue with plotting this edge, and note that we're doing so:
					edgesPlotted.append((i,j))

				edgeIsTorroidal_w 	= torroidal[0] and abs(postgapCoord[0]-pregapCoord[0])>(network.geometry.w/2)
				edgeIsTorroidal_h 	= torroidal[1] and abs(postgapCoord[1]-pregapCoord[1])>(network.geometry.h/2)

				edgeColor = gapWt_cmap( abs(gapWt-minWt)/abs(maxWt-minWt)  ) #numpy.random.rand(3,1)

				if(not edgeIsTorroidal_w and not edgeIsTorroidal_h):
					# Plot the edge without torroidal wrapping:
					ax.annotate("", xytext=pregapCoord, xy=postgapCoord, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
					pass

				elif(edgeIsTorroidal_w and not edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in w dimension:
					torroidalDist_w = w - ( max(pregapCoord[0], postgapCoord[0]) - min(pregapCoord[0], postgapCoord[0]) )
					delta_h 		= postgapCoord[1] - pregapCoord[1]
					if(pregapCoord[0] > postgapCoord[0]):
						# Configure edge segments wrapping across right-hand boundary:
						seg1Endpt 	= [network.geometry.w, pregapCoord[1]+((network.geometry.w-pregapCoord[0])/torroidalDist_w)*delta_h]
						seg2Startpt = [0, pregapCoord[1]+((network.geometry.w-pregapCoord[0])/torroidalDist_w)*delta_h]
					else:
						# Configure edge segments wrapping across left-hand boundary:
						seg1Endpt 	= [0, postgapCoord[1]-(abs(w-postgapCoord[0])/torroidalDist_w)*delta_h]
						seg2Startpt = [network.geometry.w, postgapCoord[1]-(abs(w-postgapCoord[0])/torroidalDist_w)*delta_h]
					if(seg2Startpt[0] == postgapCoord[0]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=pregapCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=pregapCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postgapCoord, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)

				elif(not edgeIsTorroidal_w and edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in h dimension:
					torroidalDist_h = h - ( max(pregapCoord[1], postgapCoord[1]) - min(pregapCoord[1], postgapCoord[1]) )
					delta_w 		= postgapCoord[0] - pregapCoord[0]
					if(pregapCoord[1] > postgapCoord[1]):
						# Configure edge segments wrapping across top boundary:
						seg1Endpt 	= [pregapCoord[0]+((network.geometry.h-pregapCoord[1])/torroidalDist_h)*delta_w, network.geometry.h]
						seg2Startpt = [pregapCoord[0]+((network.geometry.h-pregapCoord[1])/torroidalDist_h)*delta_w, 0]
					else:
						# Configure edge segments wrapping across left-hand boundary:
						seg1Endpt 	= [postgapCoord[0]-(abs(h-postgapCoord[1])/torroidalDist_h)*delta_w, 0]
						seg2Startpt = [postgapCoord[0]-(abs(h-postgapCoord[1])/torroidalDist_h)*delta_w, h]
					if(seg2Startpt[0] == postgapCoord[0]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=pregapCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=pregapCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postgapCoord, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)

				elif(edgeIsTorroidal_w and edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in w and h dimensions:
					torroidalDist_w = w - ( max(pregapCoord[0], postgapCoord[0]) - min(pregapCoord[0], postgapCoord[0]) )
					torroidalDist_h = h - ( max(pregapCoord[1], postgapCoord[1]) - min(pregapCoord[1], postgapCoord[1]) )
					delta_w 		= postgapCoord[0] - pregapCoord[0]
					delta_h 		= postgapCoord[1] - pregapCoord[1]
					m_edge					= abs(torroidalDist_h/torroidalDist_w)	#absolute value of slope (torroidal) between pre and post synaptic neurons
					m_presynToNearestCorner	= abs( min(h-pregapCoord[1], pregapCoord[1]-0) / min(w-pregapCoord[0], pregapCoord[0]-0) ) 	# absolute value of slope between presynaptic neuron and the nearest boundary corner

					if(pregapCoord[0]>postgapCoord[0] and pregapCoord[1]>postgapCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across TOP boundary and then wrap around the RIGHT boundary:
							# (segment 1 exits on top boundary, segment 2 reenters from left boundary)
							seg1Endpt	= [pregapCoord[0]+((h-pregapCoord[1])/m_edge), h]
							seg2Startpt	= [0, postgapCoord[1]-((postgapCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across RIGHT boundary and then wrap around the TOP boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from bottom boundary)
							seg1Endpt	= [w, pregapCoord[1]+((w-pregapCoord[0])*m_edge)]
							seg2Startpt	= [postgapCoord[0]-((postgapCoord[1])/m_edge), 0]
					elif(pregapCoord[0]<postgapCoord[0] and pregapCoord[1]>postgapCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across TOP boundary and then wrap around the LEFT boundary:
							# (segment 1 exits on top boundary, segment 2 reenters from right boundary)
							seg1Endpt	= [pregapCoord[0]-((h-pregapCoord[1])/m_edge), h]
							seg2Startpt	= [w, postgapCoord[1]-((w-postgapCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across LEFT boundary and then wrap around the TOP boundary:
							# (segment 1 exits on left boundary, segment 2 reenters from bottom boundary)
							seg1Endpt	= [0, pregapCoord[1]+((pregapCoord[0])*m_edge)]
							seg2Startpt	= [postgapCoord[0]+((postgapCoord[1])/m_edge), 0]
					elif(pregapCoord[0]>postgapCoord[0] and pregapCoord[1]<postgapCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across BOTTOM boundary and then wrap around the RIGHT boundary:
							# (segment 1 exits on bottom boundary, segment 2 reenters from left boundary)
							seg1Endpt	= [pregapCoord[0]+((h-pregapCoord[1])/m_edge), 0]
							seg2Startpt	= [0, postgapCoord[1]+((postgapCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across RIGHT boundary and then wrap around the BOTTOM boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from top boundary)
							seg1Endpt	= [w, pregapCoord[1]-((w-pregapCoord[0])*m_edge)]
							seg2Startpt	= [postgapCoord[0]-((postgapCoord[1])/m_edge), h]
					elif(pregapCoord[0]<postgapCoord[0] and pregapCoord[1]<postgapCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across BOTTOM boundary and then wrap around the LEFT boundary:
							# (segment 1 exits on bottom boundary, segment 2 reenters from right boundary)
							seg1Endpt	= [pregapCoord[0]-((h-pregapCoord[1])/m_edge), 0]
							seg2Startpt	= [w, postgapCoord[1]+((w-postgapCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across LEFT boundary and then wrap around the BOTTOM boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from top boundary)
							seg1Endpt	= [0, pregapCoord[1]-((pregapCoord[0])*m_edge)]
							seg2Startpt	= [postgapCoord[0]+((postgapCoord[1])/m_edge), h]
					else:
						# Should not reach else condition:
						pass

					if(seg2Startpt[0] == postgapCoord[0] and seg2Startpt[1] == postgapCoord[1]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=pregapCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=pregapCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postgapCoord, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)

				else:
					# Should not reach else condition:
					pass

	neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
	neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
	neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
	neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
	neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
	neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])

	# ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='black', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='black', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='limegreen', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	# ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='limegreen', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='darkorange', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='darkorange', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)

	ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='midnightblue', edgecolors='none', zorder=3)
	ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='maroon', edgecolors='none', zorder=3)
	ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='limegreen', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='limegreen', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)
	ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='darkorange', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)
	ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='darkorange', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4)

	# ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='blue', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='red', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='blue', edgecolors='limegreen', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	# ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='red', edgecolors='limegreen', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='blue', edgecolors='darkorange', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='blue', edgecolors='darkorange', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)

	ax.set_aspect('equal', 'datalim')

	margin_w = 0.02*w
	margin_h = 0.02*h
	ax.set_xlim(0-margin_w, w+margin_w)
	ax.set_ylim(0-margin_h, h+margin_h)

	ax.spines['bottom'].set_color('white')
	ax.spines['top'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.spines['right'].set_color('white')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	if(showAxes):
		ax.tick_params(axis='both', which='major', labelsize=6)
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	ax.set_title("Gap Junction Connectivity", {'fontsize':12})

	#~~~~~
	return ax


def gapjunction_network_diagram_3d(ax, network):

	# ax = pyplot.axes(projection='3d')

	gapjnWeights 	= network.connectionWeights_gap

	neuronCoords 	= network.geometry.cartesianCoords

	maxWt = numpy.max(gapjnWeights)
	minWt = numpy.min(gapjnWeights)
	gapWt_cmap = zero_midpoint_cmap(orig_cmap=matplotlib.cm.get_cmap('Purples'), min_val=minWt, max_val=maxWt)

	neuronGapEndpts	= []
	neuronGapWts	= []
	for i in range(network.N):
		for j in range(network.N):
			if(gapjnWeights[i,j] != 0.0):
				neuronGapEndpts.append( [neuronCoords[i], neuronCoords[j]] )
				neuronGapWts.append(gapjnWeights[i,j])

	edgesGap = Line3DCollection(neuronGapEndpts, cmap=gapWt_cmap)
	edgesGap.set_array(numpy.asarray(neuronGapWts)) # color the segments by our parameter
	ax.add_collection3d(edgesGap)

	neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
	neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
	neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
	neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
	neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
	neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])

	ax.scatter(xs=neuronCoords[neuronIDs_nonIOExcit,0], ys=neuronCoords[neuronIDs_nonIOExcit,1], zs=neuronCoords[neuronIDs_nonIOExcit,2], marker='o', c='midnightblue', edgecolors='none', zorder=3, depthshade=True)
	ax.scatter(xs=neuronCoords[neuronIDs_nonIOInhib,0], ys=neuronCoords[neuronIDs_nonIOInhib,1], zs=neuronCoords[neuronIDs_nonIOInhib,2], marker='o', c='maroon', edgecolors='none', zorder=3, depthshade=True)
	ax.scatter(xs=neuronCoords[neuronIDs_inputExcit,0], ys=neuronCoords[neuronIDs_inputExcit,1], zs=neuronCoords[neuronIDs_inputExcit,2], marker='^', c='limegreen', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	ax.scatter(xs=neuronCoords[neuronIDs_inputInhib,0], ys=neuronCoords[neuronIDs_inputInhib,1], zs=neuronCoords[neuronIDs_inputInhib,2], marker='^', c='limegreen', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)
	ax.scatter(xs=neuronCoords[neuronIDs_outputExcit,0], ys=neuronCoords[neuronIDs_outputExcit,1], zs=neuronCoords[neuronIDs_outputExcit,2], marker='s', c='darkorange', edgecolors='midnightblue', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)
	ax.scatter(xs=neuronCoords[neuronIDs_outputInhib,0], ys=neuronCoords[neuronIDs_outputInhib,1], zs=neuronCoords[neuronIDs_outputInhib,2], marker='s', c='darkorange', edgecolors='maroon', linewidths=1.5, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)

	# Create cubic bounding box to simulate equal aspect ratio
	largest_axis_range = numpy.array([neuronCoords[:,0].max()-neuronCoords[:,0].min(), neuronCoords[:,1].max()-neuronCoords[:,1].min(), neuronCoords[:,2].max()-neuronCoords[:,2].min()]).max()
	Xb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(neuronCoords[:,0].max()+neuronCoords[:,0].min())
	Yb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(neuronCoords[:,1].max()+neuronCoords[:,1].min())
	Zb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(neuronCoords[:,2].max()+neuronCoords[:,2].min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')

	axesGridColor	= (0.95, 0.95, 0.95, 1.0)
	axesPaneColor 	= (0.99, 0.99, 0.99, 1.0)
	ax.w_xaxis.set_pane_color(axesPaneColor)
	ax.w_yaxis.set_pane_color(axesPaneColor)
	ax.w_zaxis.set_pane_color(axesPaneColor)
	ax.w_xaxis._axinfo.update({'grid' : {'color': axesGridColor}})
	ax.w_yaxis._axinfo.update({'grid' : {'color': axesGridColor}})
	ax.w_zaxis._axinfo.update({'grid' : {'color': axesGridColor}})

	ax.autoscale(tight=True)

	#~~~~~
	return ax


def rate_network_diagram_2d(ax, network, connectivityMatrix=None, basisT=1000, dark=False, showAxes=False):

	try:
		neuronCoords = network.geometry.surfacePlaneCoords
	except AttributeError:
		print("The "+str(network.geometry.geometry)+" geometry class does not define a 2D surface plane coordinate system for neuron positions.")
		return

	connectivityWeights = connectivityMatrix if connectivityMatrix is not None else (network.connectionWeights_synExcit - network.connectionWeights_synInhib) # synapse weights

	neuronsDataframe 	= network.get_neurons_dataframe()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Get the number of times each neuron spiked in the simulation period:
	neuronsNumSpikes	= numpy.array( neuronsDataframe.groupby(['neuron_id']).sum().reset_index()['spike'].tolist() )

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Calculate the spike rate of each neuron:
	# Rate = numspikes/basisT
	# - basisT is the number of time units to use as the denominator of the rate formula.
	#   - If the network t unit is ms, then a basisT=1000 calculates numspikes per 1000ms, ie numspikes/sec (spike Hz)
	neuronsSpikeRates	= (neuronsNumSpikes/network.T_max) * basisT
	maxRate = numpy.max(neuronsSpikeRates)
	minRate = numpy.min(neuronsSpikeRates)

	# Mask all 0.0 rates as 'bad' so we can force the colormap to set these values to a desired 'set_bad' color (eg transparent alpha):
	neuronsSpikeRates = numpy.ma.masked_where(neuronsSpikeRates==0.0, neuronsSpikeRates)

	# Calculate the index into the colormap range for each rate:
	neuronRateCmapping = numpy.abs(neuronsSpikeRates-minRate)/abs(maxRate-minRate)

	# print neuronsNumSpikes

	# print neuronsSpikeRates

	#~~~~~~~~~~~~~~~~~~~~
	# Render the diagram:
	#~~~~~~~~~~~~~~~~~~~~

	edgeMinColor	= '#000000' if dark else '#FFFFFF'
	edgeMaxColor	= '#111111' if dark else '#EEEEEE'
	outlineColor 	= '#1A1A1A' if dark else '#E0E0E0'
	backgroundColor = '#000000' if dark else '#FFFFFF'
	surfaceBorderColor 		= (0.3, 0.3, 0.3) if dark else (0.7, 0.7, 0.7)
	surfaceBackgroundColor 	= (0.01, 0.01, 0.01) if dark else (0.99, 0.99, 0.99)

	w = network.geometry.w
	h = network.geometry.h

	try:
		torroidal = network.geometry.torroidal
	except AttributeError:
		torroidal = [False, False]

	ax.add_patch(patches.Rectangle((0,0), w, h, facecolor=surfaceBackgroundColor, edgecolor='none', zorder=0))
	ax.plot([0,0], [0,h], color=surfaceBorderColor, linestyle=(':' if torroidal[0] else '-'), zorder=1)
	ax.plot([w,w], [0,h], color=surfaceBorderColor, linestyle=(':' if torroidal[0] else '-'), zorder=1)
	ax.plot([0,w], [0,0], color=surfaceBorderColor, linestyle=(':' if torroidal[1] else '-'), zorder=1)
	ax.plot([0,w], [h,h], color=surfaceBorderColor, linestyle=(':' if torroidal[1] else '-'), zorder=1)

	maxWt = numpy.max(connectivityWeights)
	minWt = numpy.min(connectivityWeights)
	edge_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('SubtleGreys', [edgeMinColor, edgeMaxColor], N=256)

	rate_cmap = matplotlib.cm.get_cmap('afmhot' if dark else 'YlOrRd')
	rate_cmap.set_bad(color=backgroundColor)#(alpha=0.0) # This forces a color for bad (ie masked) values

	for i in range(network.N):
		for j in range(network.N):
			preconnCoord 	= neuronCoords[i] #+ [1, 0]
			postconnCoord 	= neuronCoords[j] #+ [1, 0]
			edgeWt 			= connectivityWeights[i,j]

			if(edgeWt != 0.0):

				edgeIsTorroidal_w 	= torroidal[0] and abs(postconnCoord[0]-preconnCoord[0])>(network.geometry.w/2)
				edgeIsTorroidal_h 	= torroidal[1] and abs(postconnCoord[1]-preconnCoord[1])>(network.geometry.h/2)

				edgeColor = edge_cmap( abs(edgeWt-minWt)/abs(maxWt-minWt)  ) #numpy.random.rand(3,1)

				if(not edgeIsTorroidal_w and not edgeIsTorroidal_h):
					# Plot the edge without torroidal wrapping:
					ax.annotate("", xytext=preconnCoord, xy=postconnCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					pass

				elif(edgeIsTorroidal_w and not edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in w dimension:
					torroidalDist_w = w - ( max(preconnCoord[0], postconnCoord[0]) - min(preconnCoord[0], postconnCoord[0]) )
					delta_h 		= postconnCoord[1] - preconnCoord[1]
					if(preconnCoord[0] > postconnCoord[0]):
						# Configure edge segments wrapping across right-hand boundary:
						seg1Endpt 	= [network.geometry.w, preconnCoord[1]+((network.geometry.w-preconnCoord[0])/torroidalDist_w)*delta_h]
						seg2Startpt = [0, preconnCoord[1]+((network.geometry.w-preconnCoord[0])/torroidalDist_w)*delta_h]
					else:
						# Configure edge segments wrapping across left-hand boundary:
						seg1Endpt 	= [0, postconnCoord[1]-(abs(w-postconnCoord[0])/torroidalDist_w)*delta_h]
						seg2Startpt = [network.geometry.w, postconnCoord[1]-(abs(w-postconnCoord[0])/torroidalDist_w)*delta_h]
					if(seg2Startpt[0] == postconnCoord[0]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=preconnCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=preconnCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postconnCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)

				elif(not edgeIsTorroidal_w and edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in h dimension:
					torroidalDist_h = h - ( max(preconnCoord[1], postconnCoord[1]) - min(preconnCoord[1], postconnCoord[1]) )
					delta_w 		= postconnCoord[0] - preconnCoord[0]
					if(preconnCoord[1] > postconnCoord[1]):
						# Configure edge segments wrapping across top boundary:
						seg1Endpt 	= [preconnCoord[0]+((network.geometry.h-preconnCoord[1])/torroidalDist_h)*delta_w, network.geometry.h]
						seg2Startpt = [preconnCoord[0]+((network.geometry.h-preconnCoord[1])/torroidalDist_h)*delta_w, 0]
					else:
						# Configure edge segments wrapping across left-hand boundary:
						seg1Endpt 	= [postconnCoord[0]-(abs(h-postconnCoord[1])/torroidalDist_h)*delta_w, 0]
						seg2Startpt = [postconnCoord[0]-(abs(h-postconnCoord[1])/torroidalDist_h)*delta_w, h]
					if(seg2Startpt[0] == postconnCoord[0]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=preconnCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=preconnCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postconnCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)

				elif(edgeIsTorroidal_w and edgeIsTorroidal_h):
					# Plot the edge with torroidal wrapping in w and h dimensions:
					torroidalDist_w = w - ( max(preconnCoord[0], postconnCoord[0]) - min(preconnCoord[0], postconnCoord[0]) )
					torroidalDist_h = h - ( max(preconnCoord[1], postconnCoord[1]) - min(preconnCoord[1], postconnCoord[1]) )
					delta_w 		= postconnCoord[0] - preconnCoord[0]
					delta_h 		= postconnCoord[1] - preconnCoord[1]
					m_edge					= abs(torroidalDist_h/torroidalDist_w)	#absolute value of slope (torroidal) between pre and post synaptic neurons
					m_presynToNearestCorner	= abs( min(h-preconnCoord[1], preconnCoord[1]-0) / min(w-preconnCoord[0], preconnCoord[0]-0) ) 	# absolute value of slope between presynaptic neuron and the nearest boundary corner

					if(preconnCoord[0]>postconnCoord[0] and preconnCoord[1]>postconnCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across TOP boundary and then wrap around the RIGHT boundary:
							# (segment 1 exits on top boundary, segment 2 reenters from left boundary)
							seg1Endpt	= [preconnCoord[0]+((h-preconnCoord[1])/m_edge), h]
							seg2Startpt	= [0, postconnCoord[1]-((postconnCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across RIGHT boundary and then wrap around the TOP boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from bottom boundary)
							seg1Endpt	= [w, preconnCoord[1]+((w-preconnCoord[0])*m_edge)]
							seg2Startpt	= [postconnCoord[0]-((postconnCoord[1])/m_edge), 0]
					elif(preconnCoord[0]<postconnCoord[0] and preconnCoord[1]>postconnCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across TOP boundary and then wrap around the LEFT boundary:
							# (segment 1 exits on top boundary, segment 2 reenters from right boundary)
							seg1Endpt	= [preconnCoord[0]-((h-preconnCoord[1])/m_edge), h]
							seg2Startpt	= [w, postconnCoord[1]-((w-postconnCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across LEFT boundary and then wrap around the TOP boundary:
							# (segment 1 exits on left boundary, segment 2 reenters from bottom boundary)
							seg1Endpt	= [0, preconnCoord[1]+((preconnCoord[0])*m_edge)]
							seg2Startpt	= [postconnCoord[0]+((postconnCoord[1])/m_edge), 0]
					elif(preconnCoord[0]>postconnCoord[0] and preconnCoord[1]<postconnCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across BOTTOM boundary and then wrap around the RIGHT boundary:
							# (segment 1 exits on bottom boundary, segment 2 reenters from left boundary)
							seg1Endpt	= [preconnCoord[0]+((h-preconnCoord[1])/m_edge), 0]
							seg2Startpt	= [0, postconnCoord[1]+((postconnCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across RIGHT boundary and then wrap around the BOTTOM boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from top boundary)
							seg1Endpt	= [w, preconnCoord[1]-((w-preconnCoord[0])*m_edge)]
							seg2Startpt	= [postconnCoord[0]-((postconnCoord[1])/m_edge), h]
					elif(preconnCoord[0]<postconnCoord[0] and preconnCoord[1]<postconnCoord[1]):
						if(m_edge > m_presynToNearestCorner):
							# Configure edge segments to wrap across BOTTOM boundary and then wrap around the LEFT boundary:
							# (segment 1 exits on bottom boundary, segment 2 reenters from right boundary)
							seg1Endpt	= [preconnCoord[0]-((h-preconnCoord[1])/m_edge), 0]
							seg2Startpt	= [w, postconnCoord[1]+((w-postconnCoord[0])*m_edge)]
						else:
							# Configure edge segments to wrap across LEFT boundary and then wrap around the BOTTOM boundary:
							# (segment 1 exits on right boundary, segment 2 reenters from top boundary)
							seg1Endpt	= [0, preconnCoord[1]-((preconnCoord[0])*m_edge)]
							seg2Startpt	= [postconnCoord[0]+((postconnCoord[1])/m_edge), h]
					else:
						# Should not reach else condition:
						pass

					if(seg2Startpt[0] == postconnCoord[0] and seg2Startpt[1] == postconnCoord[1]):
						# Segment 2 won't be visible, since the postsynaptic neuron is on the boundary and thus the post-wrap segment has 0 length.
						# - just draw the pre-wrap segment up to the boundary and place the arrowhead on this segment:
						ax.annotate("", xytext=preconnCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)
					else:
						# The post-wrap segment is necessary and visible.
						# - draw both segments, with the arrowhead on the post-wrap segment.
						ax.annotate("", xytext=preconnCoord, xy=seg1Endpt, arrowprops=dict(arrowstyle="-", color=edgeColor), zorder=2)
						ax.annotate("", xytext=seg2Startpt, xy=postconnCoord, arrowprops=dict(arrowstyle="->", color=edgeColor), zorder=2)

				else:
					# Should not reach else condition:
					pass

	neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
	neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
	neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
	neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
	neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
	neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])

	# ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='black', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='black', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='limegreen', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	# ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='limegreen', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='darkorange', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='darkorange', edgecolors='k', linewidths=1, s=1.5*(pyplot.rcParams['lines.markersize']**2), zorder=3)

	ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c=rate_cmap(neuronRateCmapping[neuronIDs_nonIOExcit]), edgecolors=outlineColor, linewidths=1, s=1.1*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c=rate_cmap(neuronRateCmapping[neuronIDs_nonIOInhib]), edgecolors=outlineColor, linewidths=1, s=1.1*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c=rate_cmap(neuronRateCmapping[neuronIDs_inputExcit]), edgecolors=outlineColor, linewidths=1, s=1.6*(pyplot.rcParams['lines.markersize']**2), zorder=4)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c=rate_cmap(neuronRateCmapping[neuronIDs_inputInhib]), edgecolors=outlineColor, linewidths=1, s=1.6*(pyplot.rcParams['lines.markersize']**2), zorder=4)
	ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c=rate_cmap(neuronRateCmapping[neuronIDs_outputExcit]), edgecolors=outlineColor, linewidths=1, s=1.6*(pyplot.rcParams['lines.markersize']**2), zorder=4)
	ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c=rate_cmap(neuronRateCmapping[neuronIDs_outputInhib]), edgecolors=outlineColor, linewidths=1, s=1.6*(pyplot.rcParams['lines.markersize']**2), zorder=4)

	# ax.scatter(x=neuronCoords[neuronIDs_nonIOExcit,0], y=neuronCoords[neuronIDs_nonIOExcit,1], marker='o', c='blue', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_nonIOInhib,0], y=neuronCoords[neuronIDs_nonIOInhib,1], marker='o', c='red', edgecolors='none', zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_inputExcit,0], y=neuronCoords[neuronIDs_inputExcit,1], marker='^', c='blue', edgecolors='limegreen', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	# ax.scatter(x=neuronCoords[neuronIDs_inputInhib,0], y=neuronCoords[neuronIDs_inputInhib,1], marker='^', c='red', edgecolors='limegreen', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputExcit,0], y=neuronCoords[neuronIDs_outputExcit,1], marker='s', c='blue', edgecolors='darkorange', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)
	# ax.scatter(x=neuronCoords[neuronIDs_outputInhib,0], y=neuronCoords[neuronIDs_outputInhib,1], marker='s', c='blue', edgecolors='darkorange', linewidths=2, s=2*(pyplot.rcParams['lines.markersize']**2), zorder=3)

	ax.set_aspect('equal', 'datalim')

	ax.set_axis_bgcolor(backgroundColor)

	margin_w = 0.02*w
	margin_h = 0.02*h
	ax.set_xlim(0-margin_w, w+margin_w)
	ax.set_ylim(0-margin_h, h+margin_h)

	ax.spines['bottom'].set_color('white')
	ax.spines['top'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.spines['right'].set_color('white')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	if(showAxes):
		ax.tick_params(axis='both', which='major', labelsize=6)
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	ax.set_title("Spike Rates", {'fontsize':12})

	#~~~~~
	return ax


def rate_network_diagram_3d(ax, network, connectivityMatrix=None, basisT=1000, dark=False, showAxes=False):

	# ax = pyplot.axes(projection='3d')

	connectivityWeights = connectivityMatrix if connectivityMatrix is not None else (network.connectionWeights_synExcit - network.connectionWeights_synInhib) # synapse weights

	neuronCoords = network.geometry.cartesianCoords

	neuronsDataframe 	= network.get_neurons_dataframe()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Get the number of times each neuron spiked in the simulation period:
	neuronsNumSpikes	= numpy.array( neuronsDataframe.groupby(['neuron_id']).sum().reset_index()['spike'].tolist() )

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Calculate the spike rate of each neuron:
	# Rate = numspikes/basisT
	# - basisT is the number of time units to use as the denominator of the rate formula.
	#   - If the network t unit is ms, then a basisT=1000 calculates numspikes per 1000ms, ie numspikes/sec (spike Hz)
	neuronsSpikeRates	= (neuronsNumSpikes/network.T_max) * basisT
	maxRate = numpy.max(neuronsSpikeRates)
	minRate = numpy.min(neuronsSpikeRates)

	# Mask all 0.0 rates as 'bad' so we can force the colormap to set these values to a desired 'set_bad' color (eg transparent alpha):
	neuronsSpikeRates = numpy.ma.masked_where(neuronsSpikeRates==0.0, neuronsSpikeRates)

	# Calculate the index into the colormap range for each rate:
	neuronRateCmapping = numpy.abs(neuronsSpikeRates-minRate)/abs(maxRate-minRate)

	print neuronsNumSpikes

	print neuronsSpikeRates

	#~~~~~~~~~~~~~~~~~~~~
	# Render the diagram:
	#~~~~~~~~~~~~~~~~~~~~

	edgeMinColor	= '#000000' if dark else '#FFFFFF'
	edgeMaxColor	= '#111111' if dark else '#EEEEEE'
	outlineColor 	= '#222222' if dark else '#DDDDDD'
	backgroundColor = '#000000' if dark else '#FFFFFF'
	axesBorderColor 		= (0.3, 0.3, 0.3) if dark else (0.7, 0.7, 0.7)
	axesBackgroundColor 	= (0.01, 0.01, 0.01) if dark else (0.99, 0.99, 0.99)

	maxWt = numpy.max(connectivityWeights)
	minWt = numpy.min(connectivityWeights)
	edge_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('SubtleGreys', [edgeMinColor, edgeMaxColor], N=256)

	rate_cmap = matplotlib.cm.get_cmap('afmhot' if dark else 'YlOrRd')
	rate_cmap.set_bad(color=backgroundColor)#(alpha=0.0) # This forces a color for bad (ie masked) values

	neuronSynEndpts	= []
	neuronSynWts	= []
	for i in range(network.N):
		for j in range(network.N):
			if(connectivityWeights[i,j] != 0.0):
				neuronSynEndpts.append( [neuronCoords[i], neuronCoords[j]] )
				neuronSynWts.append(connectivityWeights[i,j])

	edges = Line3DCollection(neuronSynEndpts, cmap=edge_cmap)
	edges.set_array(numpy.asarray(neuronSynWts)) # color the segments by our parameter
	ax.add_collection3d(edges)

	neuronIDs_nonIOExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=[''])
	neuronIDs_nonIOInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=[''])
	neuronIDs_inputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['input'])
	neuronIDs_inputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['input'])
	neuronIDs_outputExcit	= network.get_neuron_ids(synapseTypes=['excitatory'], labels=['output'])
	neuronIDs_outputInhib	= network.get_neuron_ids(synapseTypes=['inhibitory'], labels=['output'])

	ax.scatter(xs=neuronCoords[neuronIDs_nonIOExcit,0], ys=neuronCoords[neuronIDs_nonIOExcit,1], zs=neuronCoords[neuronIDs_nonIOExcit,2], marker='o', c=rate_cmap(neuronRateCmapping[neuronIDs_nonIOExcit]), edgecolors=outlineColor, linewidths=1, s=1.05*(pyplot.rcParams['lines.markersize']**2), zorder=3, depthshade=True)
	ax.scatter(xs=neuronCoords[neuronIDs_nonIOInhib,0], ys=neuronCoords[neuronIDs_nonIOInhib,1], zs=neuronCoords[neuronIDs_nonIOInhib,2], marker='o', c=rate_cmap(neuronRateCmapping[neuronIDs_nonIOInhib]), edgecolors=outlineColor, linewidths=1, s=1.05*(pyplot.rcParams['lines.markersize']**2), zorder=3, depthshade=True)
	ax.scatter(xs=neuronCoords[neuronIDs_inputExcit,0], ys=neuronCoords[neuronIDs_inputExcit,1], zs=neuronCoords[neuronIDs_inputExcit,2], marker='^', c=rate_cmap(neuronRateCmapping[neuronIDs_inputExcit]), edgecolors=outlineColor, linewidths=1, s=1.7*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)	# s=x*(pyplot.rcParams['lines.markersize']**2) is setting the size of the marker to x times its default size)
	ax.scatter(xs=neuronCoords[neuronIDs_inputInhib,0], ys=neuronCoords[neuronIDs_inputInhib,1], zs=neuronCoords[neuronIDs_inputInhib,2], marker='^', c=rate_cmap(neuronRateCmapping[neuronIDs_inputInhib]), edgecolors=outlineColor, linewidths=1, s=1.7*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)
	ax.scatter(xs=neuronCoords[neuronIDs_outputExcit,0], ys=neuronCoords[neuronIDs_outputExcit,1], zs=neuronCoords[neuronIDs_outputExcit,2], marker='s', c=rate_cmap(neuronRateCmapping[neuronIDs_outputExcit]), edgecolors=outlineColor, linewidths=1, s=1.55*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)
	ax.scatter(xs=neuronCoords[neuronIDs_outputInhib,0], ys=neuronCoords[neuronIDs_outputInhib,1], zs=neuronCoords[neuronIDs_outputInhib,2], marker='s', c=rate_cmap(neuronRateCmapping[neuronIDs_outputInhib]), edgecolors=outlineColor, linewidths=1, s=1.55*(pyplot.rcParams['lines.markersize']**2), zorder=4, depthshade=False)

	# Create cubic bounding box to simulate equal aspect ratio
	largest_axis_range = numpy.array([neuronCoords[:,0].max()-neuronCoords[:,0].min(), neuronCoords[:,1].max()-neuronCoords[:,1].min(), neuronCoords[:,2].max()-neuronCoords[:,2].min()]).max()
	Xb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(neuronCoords[:,0].max()+neuronCoords[:,0].min())
	Yb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(neuronCoords[:,1].max()+neuronCoords[:,1].min())
	Zb = 0.5*largest_axis_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(neuronCoords[:,2].max()+neuronCoords[:,2].min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')

	ax.w_xaxis.set_pane_color(axesBackgroundColor)
	ax.w_yaxis.set_pane_color(axesBackgroundColor)
	ax.w_zaxis.set_pane_color(axesBackgroundColor)
	ax.w_xaxis._axinfo.update({'grid' : {'color': axesBorderColor}})
	ax.w_yaxis._axinfo.update({'grid' : {'color': axesBorderColor}})
	ax.w_zaxis._axinfo.update({'grid' : {'color': axesBorderColor}})

	ax.autoscale(tight=True)

	#~~~~~
	return ax


def synapse_connectivity_matrix(ax, connectivityMatrix):
	matSyn_cmap = synapse_connectivity_colormap(connectivityMatrix)

	# If the matrix is all 0 values, mask all those 0 values as 'bad' so we can force the colormap to set these values to a desired 'set_bad' color (eg 'white'):
	if(connectivityMatrix.min()==connectivityMatrix.max()==0.0):
		connectivityMatrix = numpy.ma.masked_where(connectivityMatrix==0.0, connectivityMatrix)

	img_matSyn 	= ax.imshow(connectivityMatrix, cmap=matSyn_cmap, interpolation='none')
	cbar_matSyn	= pyplot.colorbar(img_matSyn, ax=ax)
	cbar_matSyn.ax.tick_params(labelsize=(8 if ((connectivityMatrix.min()!=0.0 or connectivityMatrix.max()!=0.0) and connectivityMatrix.max()!=connectivityMatrix.min()) else 0))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	#~~~~~
	return ax


def gapjunction_connectivity_matrix(ax, connectivityMatrix):
	# If the matrix is all 0 values, mask all those 0 values as 'bad' so we can force the colormap to set these values to a desired 'set_bad' color (eg 'white'):
	if(connectivityMatrix.min()==connectivityMatrix.max()==0.0):
		connectivityMatrix = numpy.ma.masked_where(connectivityMatrix==0.0, connectivityMatrix)

	img_matGap 	= ax.imshow(connectivityMatrix, cmap=gapjunction_connectivity_colormap(connectivityMatrix), interpolation='none')
	cbar_matGap	= pyplot.colorbar(img_matGap, ax=ax)
	cbar_matGap.ax.tick_params(labelsize=(8 if ((connectivityMatrix.min()!=0.0 or connectivityMatrix.max()!=0.0) and connectivityMatrix.max()!=connectivityMatrix.min()) else 0))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	#~~~~~
	return ax


def input_connectivity_matrix(ax, connectivityMatrix):
	# If the matrix is all 0 values, mask all those 0 values as 'bad' so we can force the colormap to set these values to a desired 'set_bad' color (eg 'white'):
	if(connectivityMatrix.min()==connectivityMatrix.max()==0.0):
		connectivityMatrix = numpy.ma.masked_where(connectivityMatrix==0.0, connectivityMatrix)

	img_matInp 	= ax.imshow(connectivityMatrix, cmap=input_connectivity_colormap(connectivityMatrix), interpolation='none')
	cbar_matInp	= pyplot.colorbar(img_matInp, ax=ax)
	cbar_matInp.ax.tick_params(labelsize=(8 if ((connectivityMatrix.min()!=0.0 or connectivityMatrix.max()!=0.0) and connectivityMatrix.max()!=connectivityMatrix.min()) else 0))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title("Ext. Input Connectivity", {'fontsize':12})
	#~~~~~
	return ax