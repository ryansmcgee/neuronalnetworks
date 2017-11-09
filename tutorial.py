#********************
# This tutorial will walk through the construction and simulation of a network using the neuronalnetworks package. 
# We will rapidly build up an elaborate neuronal network and execute a dynamic simulation scenario. 
# While perhaps not biologically meaningful, this simulation will illustrate the range of functionalities offered by this package.
#********************

#~~~~~~~~~~
# Import standard modules:
#~~~~~~~~~~
from __future__ import division    # use float division as in python 3

import numpy as numpy

numpy.random.seed(346911)    # seed the numpy RNG so every run of this tutorial gives the same results

#~~~~~~~~~~
# For this tutorial, let's import all of the package's modules:
#~~~~~~~~~~
from neuronalnetworks import *




############################
# CONSTRUCTING THE NETWORK #
############################

#++++++++++++++++++++++++++
# NEURONAL DYNAMICS MODEL +
#++++++++++++++++++++++++++

#~~~~~~~~~~
# Let's instantiate a network that uses Izhikevich model dynamics:
#~~~~~~~~~~
network = IzhikevichNetwork()



#++++++++++++++++++++++++++++++++++
# ADDING & PARAMETERIZING NEURONS +
#++++++++++++++++++++++++++++++++++

#~~~~~~~~~~
# Let's add a batch of excitatory neurons and a batch of inhibitory neurons to our network:
#~~~~~~~~~~
N_excit = 100    # number of excitatory neurons
N_inhib = 20     # number of inhibitory neurons

network.add_neurons(numNeuronsToAdd=N_excit,
    V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0,
    U_init=0.0, a=0.02, b=0.2, d=8, k=0.7, R_membrane=0.01,
    g_excit_init=0.0, g_inhib_init=0.0, g_gap=1.0, tau_g_excit=2.0, tau_g_inhib=2.0,
    synapse_type='excitatory')

network.add_neurons(numNeuronsToAdd=N_inhib,
    V_init=-65.0, V_r=-60.0, V_t=-40.0, V_peak=30.0, V_reset=-65, V_eqExcit=0.0, V_eqInhib=-70.0,
    U_init=0.0, a=0.02, b=0.2, d=8, k=0.7, R_membrane=0.01,
    g_excit_init=0.0, g_inhib_init=0.0, g_gap=1.0, tau_g_excit=2.0, tau_g_inhib=2.0,
    synapse_type='inhibitory')

#~~~~~~~~~~
# We can get lists of the neuron IDs for the excitatory and inhibitory neurons. 
# This is convenient for referencing and handling excitatory and inhibitory neurons seperately later.
#~~~~~~~~~~
neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]
neuronIDs_inhib = numpy.where(network.neuronSynapseTypes == 'inhibitory')[0]



#+++++++++++++++++++
# NETWORK GEOMETRY +
#+++++++++++++++++++

#~~~~~~~~~~
# Let's have our network reside on the surface of a cylinder:
#~~~~~~~~~~
network.geometry = CylinderSurface(r=1.5, h=10)

#~~~~~~~~~~
# Now, we add neurons to the geometry instance, since the network's geometry 
# was not instantiated until after we added our batches of neurons:
#~~~~~~~~~~
network.geometry.add_neurons(numNeuronsToAdd=N_excit+N_inhib)

#~~~~~~~~~~
# We will position our network's excitatory neurons evenly over the full cylinder surface, 
# and position the inhibitory neurons randomly in a limited band around the midline of the cylinder:
#~~~~~~~~~~
network.geometry.position_neurons(positioning='even', neuronIDs=neuronIDs_excit)
network.geometry.position_neurons(positioning='random', bounds={'h':[4,6]}, neuronIDs=neuronIDs_inhib)

#~~~~~~~~~~
# Let's take a look at our network's geometry and neuron positions. 
# (A number of network plots are provided, we use one here. Excitatory neurons are colored blue, inhibitory red):
#~~~~~~~~~~
axsyn3d = pyplot.subplot(projection='3d')
synapse_network_diagram_3d(axsyn3d, network)
pyplot.show()

#~~~~~~~~~~
# Distances between neurons in our network were already calculated when the neurons were positioned:
#~~~~~~~~~~
#network.geometry.distances



#+++++++++++++++++++++++
# NETWORK CONNECTIVITY +
#+++++++++++++++++++++++

#~~~~~~~~~~
# Let's configure our synaptic connectivities such that probability of neuron adjacency decays with their distance. 
# Excitatory synaptic connections will have weights that decrease linearly in distance, 
# while inhibitory synaptic connections will have weights that decrease exponentially in distance:
#~~~~~~~~~~
W_synE = generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='uniform', 
                                       args={'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.9,
                                             'low':20.0, 'high':40.0,
                                             'distances':network.geometry.distances[neuronIDs_excit]} )

W_synI = generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='distance',
                                       args={'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.6,
                                             'init_weight_dist_fn':'exponential', 'p0_w':40.0, 'sigma_w':2.0,
                                             'distances':network.geometry.distances[neuronIDs_inhib]} )

network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)
network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)

#~~~~~~~~~~
# Let's take a look at our network's synaptic connectivity:
#~~~~~~~~~~
axsyn3d = pyplot.subplot(1, 2, 1, projection='3d')
synapse_network_diagram_3d(axsyn3d, network)
axsyn2d = pyplot.subplot(1, 2, 2)
synapse_network_diagram_2d(axsyn2d, network)
pyplot.show()

#~~~~~~~~~~
# We can also configure our network's gap junction connectivity. 
# For gap junctions, let's use a nearest-neighbors adjacency scheme and a constant connection weight. 
# But let's add an additional element to this connectivity: 
# Let's make the gap junction connectivity increasingly sparse as you move up the length of the cylinder:
#~~~~~~~~~~
W_synG = numpy.zeros(shape=(network.N, network.N))    # starting with empty container for connectivity matrix

for nID in network.get_neuron_ids():
    # Generate and set connectivity vectors one neuron at a time, 
    # using their percentile height on the cylinder as a parameter 
    # for the additional sparsity applied to the generated vector.
    heightPercentile = network.geometry.cartesianCoords[nID][2]/network.geometry.h
    W_synG[nID] = generate_connectivity_vector(N=network.N, neuronID=nID,
                                               adjacencyScheme='nearest_neighbors', initWeightScheme='constant', 
                                               args={'k':4,'c_w':1.0, 'distances':network.geometry.distances[int(nID)]}, 
                                               sparsity=heightPercentile )

network.set_gapjunction_connectivity(connectivity=W_synG)

#~~~~~~~~~~
# Now let's take a look at our network's gap junction connectivity:
#~~~~~~~~~~
axgap3d = pyplot.subplot(1, 2, 1, projection='3d')
gapjunction_network_diagram_3d(axgap3d, network)
axgap2d = pyplot.subplot(1, 2, 2)
gapjunction_network_diagram_2d(axgap2d, network)
pyplot.show()



#+++++++++++++++++
# NETWORK LABELS +
#+++++++++++++++++
#~~~~~~~~~~
# Let's designate the 10 neurons highest-poisitioned on the cylinder as 'output' neurons by labeling them as such. 
# We will specifically reference this set of output neurons later on in the simulation.
#~~~~~~~~~~
neuronIDs_geoTop10 = numpy.argpartition(network.geometry.cartesianCoords[:,2], -10)[-10:]
neuronIDs_outputs  = neuronIDs_geoTop10
network.neuronLabels[neuronIDs_outputs] = 'output'



#+++++++++++++++++
# NETWORK INPUTS +
#+++++++++++++++++

#~~~~~~~~~~
# Let's allocate 2 inputs to our network:
#~~~~~~~~~~
numInputs = 2
network.add_inputs(numInputsToAdd=numInputs)

#~~~~~~~~~~
# Let's select 10 excitatory neurons randomly from the bottom half of our network to be input neurons. 
# We have allocated 2 inputs for our network. We will map input 1 to the first of our 10 randomly selected input neurons, and map input 2 to the remaining 9 selected input neurons.
#~~~~~~~~~~
# Randomly select 10 excitatory neurons from the bottom half of the geometry:
neuronIDs_geoBotHalf = numpy.argpartition(network.geometry.cartesianCoords[:,2], int(network.N/2))[:int(network.N/2)]
neuronIDs_inputs = numpy.random.choice( numpy.intersect1d(neuronIDs_geoBotHalf, neuronIDs_excit), 10, replace=False )
network.neuronLabels[neuronIDs_inputs] = 'input'    # label the selected input neurons with 'input'
# Generate a matrix that maps input 1 to the first selected input neuron, 
# and maps input 2 to the remaining input neurons:
W_inp = numpy.zeros(shape=(numInputs, network.N))
W_inp[0, numpy.sort(neuronIDs_inputs)[0]] 	= 1.0
W_inp[1, numpy.sort(neuronIDs_inputs)[1:]]	= 1.0
# Set the input connectivity to the matrix we generated:
network.set_input_connectivity(connectivity=W_inp)

# We will update our input values continuously throughout the simulation loop later on.

#~~~~~~~~~~
# Let's have input 1 be a constant input current. We instantiate the corresponding input value generating object: 
# We will use this object to generate values when setting input values each time step in the simulation loop later on.
#~~~~~~~~~~
constantInput 	= ConstantInput(constVal=250.0)

#~~~~~~~~~~
#Let's take a moment to visualize the final constructed network, including positions of labeled input and output neurons:
#~~~~~~~~~~
# Input neurons are visualized as green triangles. Output neurons are visualized as orange squares.
axsyn3d = pyplot.subplot(1, 2, 1, projection='3d')
synapse_network_diagram_3d(axsyn3d, network)
axsyn2d = pyplot.subplot(1, 2, 2)
synapse_network_diagram_2d(axsyn2d, network)
pyplot.show()

#~~~~~~~~~~
# Let's also identify the neuronIDs of the neurons that are neither input nor output neurons ("non-IO neurons"):
#~~~~~~~~~~
neuronIDs_nonIO = [nID for nID in network.get_neuron_ids() if nID not in neuronIDs_inputs and nID not in neuronIDs_outputs]




##########################
# SIMULATING THE NETWORK #
##########################

#+++++++++++++++++++++++++
# Our Simulation Scenario: 
# 	The radius of the network's cylinder surface geometry will grow spontaneously at a constant rate. 
# 	- As the geometry changes with this increasing radius, neuron positions and calculated distances will update accordingly. 
# 	- The outgoing connection weights of inhibitory neurons will continuously update, 
#	  and thus inhibitory connections, whose weights decrease exponentially with distance, 
#	  will decrease as the radius increases and neuron distances grow. 
# 	- The connection weights of excitatory neurons will be held static at their intial weights. 
# 	- The network's second input will be a current proportional to the difference between the cylinder's current radius and its baseline radius. 
#	  Thus, this input current will increase linearly along with the radius. 
# 	When the average firing rate of the designated 'output' neurons reaches 30 Hz or greater, 
#	  this triggers the cylinder surface's radius to snap back to its baseline radius. 
# 	- Output neuron spikes will be tracked in order to calculate a moving average of firing rate
#+++++++++++++++++++++++++

#++++++++++++++++++++++++++++
# SIMULATION INITIALIZATION +
#++++++++++++++++++++++++++++

#~~~~~~~~~~
#We begin by initializing a simulation with duration Tmax of 2000ms and time step  deltaT of 0.5ms:
#~~~~~~~~~~
network.initialize_simulation(T_max=2000, deltaT=0.5)

#~~~~~~~~~~
# Variables for updating the radius of the cylinder surface according to the simulation scenario:
#~~~~~~~~~~
# The current cylinder surface radius:
# network.geometry.r
# The baseline radius to which the geometry is reset when output firing rate is sufficiently high:
radius_baseline = network.geometry.r    # initialized to the geometry's initial defined radius
# The constant rate at which the radius grows:
dRdt = 0.005    # 0.005 units/ms = 5 units/sec

#~~~~~~~~~~
# Variables for calculating the average firing rate of output (and also input) neurons:
#~~~~~~~~~~
# Length of the window (ms) for calculating moving average of firing rate:
T_rateSampleWindow = 25    
# Store the number of spikes at each time step within the current window (lists to act as FIFO queues):
outputSpikeCounts = [0 for n in range(int(T_rateSampleWindow/network.deltaT))]
inputSpikeCounts = [0 for n in range(int(T_rateSampleWindow/network.deltaT))]
nonIOSpikeCounts = [0 for n in range(int(T_rateSampleWindow/network.deltaT))]
# The current average spike rates:
inputSpikeRate = 0
outputSpikeRate = 0
nonIOSpikeRate = 0

#~~~~~~~~~~
# Variables for recording the values of interest for the scenario:
#~~~~~~~~~~
# Lists for storing the values of the radius and firing rates at each time step:
log_radius = numpy.zeros(network.numTimeSteps+1)
log_outputRates = numpy.zeros(network.numTimeSteps+1)
log_inputRates = numpy.zeros(network.numTimeSteps+1)
log_nonIORates = numpy.zeros(network.numTimeSteps+1)
# Record the initial values of the radius and firing rates before the simulation loop starts:
log_radius[network.timeStepIndex] = network.geometry.r
log_inputRates[network.timeStepIndex] = inputSpikeRate    # = 0 initially
log_outputRates[network.timeStepIndex] = outputSpikeRate  # = 0 initially
log_nonIORates[network.timeStepIndex] = nonIOSpikeRate    # = 0 initially


#++++++++++++++++++
# SIMULATION LOOP +
#++++++++++++++++++

# while within the allotted simulation time:
while(network.sim_state_valid()):
    # Update the radius of the cylinder surface, which is increasing at a constant rate:
    # (This will automatically trigger updates to neuron positions and calcualted distances)
    # ( ^ this is the time limiting step of this sim loop)
    network.geometry.set_r(network.geometry.r+dRdt)
    
    # Update the inhibitory neuron connectivity based on the new geometry and neuron positions/distances:
    W_synI = generate_connectivity_vectors(neuronIDs=neuronIDs_inhib, N=network.N, adjacencyScheme='given', initWeightScheme='distance',
                                       args={'given_adj':network.connectionWeights_synInhib[neuronIDs_inhib],
                                             'init_weight_dist_fn':'exponential', 'p0_w':40.0, 'sigma_w':2.0,
                                             'distances': network.geometry.distances[neuronIDs_inhib] } )
    network.set_synaptic_connectivity(connectivity=W_synI, synapseType='i', updateNeurons=neuronIDs_inhib)

    # Set the input vals:
    # - Input 1 is a constant current
    # - Input 2 is a current that increases in magnitude in proportion to the radius's enlargement over baseline:
    network.set_input_vals( vals=[constantInput.val(network.t), 100*(network.geometry.r - radius_baseline)] )

    # Advance the state of the network's neurons by integrating their dynamics:
    network.sim_step()

    # Calculate the current average output neuron spike rate within the moving average window:
    outputSpikeCounts.pop(0)
    outputSpikeCounts.append(numpy.sum(network.spikeEvents[neuronIDs_outputs]))
    outputSpikeRate = (sum(outputSpikeCounts)/len(neuronIDs_outputs))*(1000/(T_rateSampleWindow))

    # (Also calculate the current average input neuron spike rate within the moving average window:)
    inputSpikeCounts.pop(0)
    inputSpikeCounts.append(numpy.sum(network.spikeEvents[neuronIDs_inputs]))
    inputSpikeRate = (sum(inputSpikeCounts)/len(neuronIDs_inputs))*(1000/(T_rateSampleWindow))
    
    # (Also calculate the current average non-input/output neuron spike rate within the moving average window:)
    nonIOSpikeCounts.pop(0)
    nonIOSpikeCounts.append(numpy.sum(network.spikeEvents[neuronIDs_nonIO]))
    nonIOSpikeRate = (sum(nonIOSpikeCounts)/len(neuronIDs_nonIO))*(1000/(T_rateSampleWindow))

    # If the average output neuron spike rate is sufficiently high, reset the cylinder radius:
    if(outputSpikeRate >= 30.0):
        network.geometry.r = radius_baseline

    # Record the current values of the radius and firing rates:
    log_radius[network.timeStepIndex] = network.geometry.r
    log_outputRates[network.timeStepIndex] = outputSpikeRate
    log_inputRates[network.timeStepIndex] = inputSpikeRate
    log_nonIORates[network.timeStepIndex] = nonIOSpikeRate



#++++++++++++++++
# VISUALIZATION +
#++++++++++++++++

#~~~~~~~~~~
# Plotting the cylinder radius and firing rates of input, output, and non-IO neurons:
#~~~~~~~~~~
ax_radiusrates = pyplot.subplot()
rAx1 = ax_radiusrates
rAx2 = rAx1.twinx()
rAx2.plot(log_radius, color='black')
rAx1.plot(log_nonIORates, color='lightgray')
rAx1.plot(log_outputRates, color='darkorange')
rAx1.plot(log_inputRates, color='green')
rAx1.set_ylabel('average neuron firing rate (Hz)')
rAx2.set_ylabel('cylinder radius')
rAx1.legend(['interneurons', 'output neurons', 'input neurons'], loc='upper left', fontsize='small')
rAx2.legend(['cylinder radius'], loc='lower right', fontsize='small')
rAx1.set_xlabel('t (ms)')
rAx1.set_xticklabels([0, int(network.T_max/8)*0, int(network.T_max/8)*1, int(network.T_max/8)*2, int(network.T_max/8)*3, int(network.T_max/8)*4, int(network.T_max/8)*5, int(network.T_max/8)*6, int(network.T_max/8)*7, int(network.T_max/8)*8])
pyplot.show()

#~~~~~~~~~~
# Raster plot of spike times for all neurons: 
#~~~~~~~~~~
# (green: input neurons, orange: output neurons, blue: non-IO excitatory neurons, red: non-IO inhibitory neurons)
axraster = pyplot.subplot()
spike_raster_plot(axraster, network, colorSynapseTypes=True, colorIOTypes=True)
pyplot.show()

#~~~~~~~~~~
# Average spike rate of each neuron in the network: 
#~~~~~~~~~~
# (Depicted in heatmap style on network diagram; white: no firing, dark red: high firing rate)
axheatrate = pyplot.subplot(1, 2, 1, projection='3d')
rate_network_diagram_3d(axheatrate, network)
axheatrate = pyplot.subplot(1, 2, 2)
rate_network_diagram_2d(axheatrate, network, truePlaneDimensions=False)
pyplot.show()

#~~~~~~~~~~
# Voltage/Current traces for the input neurons:
#~~~~~~~~~~
neuronsDataFrame 	= network.get_neurons_dataframe()

gs2 = gridspec.GridSpec(len(neuronIDs_inputs), 1)
gs2.update(left=0.0, right=1.0, top=1.0, bottom=0.00, hspace=0.0)

y1_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_inputs)), ['V']].min().min(),
            neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_inputs)), ['V']].max().max()]
y2_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_inputs)), ['I_excit', 'I_inhib', 'I_gap', 'I_input']].min().min(),
                neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_inputs)), ['I_excit', 'I_inhib', 'I_gap', 'I_input']].max().max()*1.05] #*1.05 is to give a 5% margin at high end to make sure highest currents' lines are visible

for i, nID in enumerate(numpy.sort(neuronIDs_inputs)):
    ax_n = pyplot.subplot(gs2[i, :])
    
    x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
    trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
    trace_Iex 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_excit'].values, 'label':'I_excit', 'color':'blue', 'alpha':1.0, 'linestyle':':'}
    trace_Iih 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_inhib'].values, 'label':'I_inhib', 'color':'red', 'alpha':1.0, 'linestyle':':'}
    trace_Igp 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_gap'].values, 'label':'I_gap', 'color':'purple', 'alpha':1.0, 'linestyle':':'}
    trace_Iin 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_input'].values, 'label':'I_input', 'color':'green', 'alpha':1.0, 'linestyle':':'}
    
    traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_Iex, trace_Iih, trace_Igp, trace_Iin], y1_lim=y1_axlim, y2_lim=y2_axlim,
                x_axis_label='t', y1_axis_label='N'+str(nID)+' Voltage', y2_axis_label='N'+str(nID)+' Currents',
                y1_legend=False, y2_legend=False, fontsize=8, x_labelsize=(0 if i<(len(neuronIDs_inputs)-1) else 6) )

    spikeTimes 	= network.get_spike_times()
    ax_n.scatter(x=spikeTimes[nID], y=neuronsDataFrame.loc[((neuronsDataFrame['neuron_id']==nID)&(neuronsDataFrame['t'].isin(spikeTimes[nID]))), 'V'].values, marker='^', c='k', edgecolors='none')

pyplot.show()

#~~~~~~~~~~
# Voltage/Current traces for output neurons:
#~~~~~~~~~~
# Already retrieved DataFrame when plotting input neuron traces:
# neuronsDataFrame 	= network.get_neurons_dataframe()

gs2 = gridspec.GridSpec(len(neuronIDs_outputs), 1)
gs2.update(left=0.0, right=1.0, top=1.0, bottom=0.00, hspace=0.15)

y1_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_outputs)), ['V']].min().min(),
            neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_outputs)), ['V']].max().max()]
y2_axlim = [neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_outputs)), ['I_excit', 'I_inhib', 'I_gap', 'I_input']].min().min(),
                neuronsDataFrame.loc[(neuronsDataFrame['neuron_id'].isin(neuronIDs_outputs)), ['I_excit', 'I_inhib', 'I_gap', 'I_input']].max().max()*1.05] #*1.05 is to give a 5% margin at high end to make sure highest currents' lines are visible

for i, nID in enumerate(numpy.sort(neuronIDs_outputs)):
    ax_n = pyplot.subplot(gs2[i, :])
    
    x_series 	= neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 't'].values
    trace_V 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'V'].values, 'label':'V', 'color':'black', 'alpha':1.0, 'linestyle':'-'}
    trace_Iex 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_excit'].values, 'label':'I_excit', 'color':'blue', 'alpha':1.0, 'linestyle':':'}
    trace_Iih 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_inhib'].values, 'label':'I_inhib', 'color':'red', 'alpha':1.0, 'linestyle':':'}
    trace_Igp 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_gap'].values, 'label':'I_gap', 'color':'purple', 'alpha':1.0, 'linestyle':':'}
    trace_Iin 	= {'data':neuronsDataFrame.loc[(neuronsDataFrame['neuron_id']==nID), 'I_input'].values, 'label':'I_input', 'color':'green', 'alpha':1.0, 'linestyle':':'}
    
    traces_plot(ax_n, x=x_series, y1_traces=[trace_V], y2_traces=[trace_Iex, trace_Iih, trace_Igp, trace_Iin], y1_lim=y1_axlim, y2_lim=y2_axlim,
                x_axis_label='t', y1_axis_label='N'+str(nID)+' Voltage', y2_axis_label='N'+str(nID)+' Currents',
                y1_legend=False, y2_legend=False, fontsize=8, x_labelsize=(0 if i<(len(neuronIDs_inputs)-1) else 6) )

    spikeTimes 	= network.get_spike_times()
    ax_n.scatter(x=spikeTimes[nID], y=neuronsDataFrame.loc[((neuronsDataFrame['neuron_id']==nID)&(neuronsDataFrame['t'].isin(spikeTimes[nID]))), 'V'].values, marker='^', c='k', edgecolors='none')

pyplot.show()

