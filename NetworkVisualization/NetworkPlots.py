


# def traces_plot(ax, x_data, y1_data=None, y1_labels=None, y1_colors=None, y1_alphas=None, y1_linestyles=None, y2_data=None, y2_labels=None, y2_colors=None, y2_alphas=None, x_axis_label='', y_axis1_label='', fontsize=8, labelsize=6):
def traces_plot(ax, x, y1_traces=None, y2_traces=None, x_axis_label='', y1_axis_label='', y2_axis_label='', y1_legend=False, y2_legend=False, fontsize=8, labelsize=6):

	yAx1 = ax
	yAx2 = yAx1.twinx()

	# Ensure that all trace argument sets are 2D lists, converting 1D lists to 2D lists
	# y1_data 		= y1_data if (not isinstance(y1_data)) else [y1_data]
	# y1_colors 		= y1_colors if (not isinstance(y1_colors)) else [y1_colors]
	# y1_alphas		= y1_alphas if (not isinstance(y1_alphas)) else [y1_alphas]
	# y1_linestyles 	= y1_linestyles if (not isinstance(y1_linestyles)) else [y1_linestyles]
	# y1_labels 		= y1_labels if (not isinstance(y1_labels)) else [y1_labels]
	# y2_data 		= y2_data if (not isinstance(y2_data)) else [y2_data]
	# y2_colors 		= y2_colors if (not isinstance(y2_colors)) else [y2_colors]
	# y2_alphas 		= y2_alphas if (not isinstance(y2_alphas)) else [y2_alphas]
	# y2_linestyles 	= y2_linestyles if (not isinstance(y2_linestyles)) else [y2_linestyles]
	# y2_labels 		= y2_labels if (not isinstance(y2_labels)) else [y2_labels]

	# Establish default color/alpha/linestyle/label

	# Check that tracemargument sets are the same lengths for each y axis
	# if(len(y1_data) != len(y1_colors) or len(y1_data) != len(y1_alphas) or len(y1_data) != len(y1_linestyles) or len(y1_data) != len(y1_labels)):
	# 	print "Traces Plot Error: The number of y-axis 1 colors, alphas, linestyles, or labels do not match the number of y-axis1 data series."
	# 	return
	# if(len(y2_data) != len(y2_colors) or len(y2_data) != len(y2_alphas) or len(y2_data) != len(y2_linestyles) or len(y2_data) != len(y2_labels)):
	# 	print "Traces Plot Error: The number of y-axis 2 colors, alphas, linestyles, or labels do not match the number of y-axis2 data series."
	# 	return

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
		yAx2.set(ylim=[y2_minVal, y2_maxVal])
		yAx2.set(xlim=[0, max(x)])
		yAx2.grid(False)
		yAx2.tick_params(axis='y', which='major', labelsize=6)

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
		yAx1.set(ylim=[y1_minVal, y1_maxVal])
		yAx1.set(xlim=[0, max(x)])
		yAx1.grid(False)
		yAx1.tick_params(axis='y', which='major', labelsize=6)

		if(y1_legend):	
			yAx1.legend(y1_allLabels, loc='upper right', fontsize='x-small')

	#~~~~~
	return ax


