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

print "[ Constructing network... ]"

#++++++++++++++++++++++++++
# NEURONAL DYNAMICS MODEL +
#++++++++++++++++++++++++++

#~~~~~~~~~~
# Let's instantiate a network that uses Izhikevich model dynamics:
#~~~~~~~~~~
network = LIFNetwork()



#++++++++++++++++++++++++++++++++++
# ADDING & PARAMETERIZING NEURONS +
#++++++++++++++++++++++++++++++++++

#~~~~~~~~~~
# Let's add a batch of excitatory neurons to our network:
#~~~~~~~~~~
N_excit = 100    # number of excitatory neurons

network.add_neurons(numNeuronsToAdd=N_excit,
                    V_init=-68.0, V_thresh=-50.0, V_reset=-70.0, V_eqLeak=-68.0, V_eqExcit=0.0, V_eqInhib=-70.0, R_membrane=1.0,
                    g_leak=0.3, g_excit_init=0.0, g_inhib_init=0.0, g_gap=0.5,
                    tau_g_excit=2.0, tau_g_inhib=2.0, refracPeriod=3.0,
                    synapse_type='excitatory')

#~~~~~~~~~~
# We can get lists of the neuron IDs for the excitatory neurons (trivial in this case because all neurons are excitatory). 
# This is convenient for referencing and handling excitatory and inhibitory neurons seperately later.
#~~~~~~~~~~
neuronIDs_excit = numpy.where(network.neuronSynapseTypes == 'excitatory')[0]



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
network.geometry.add_neurons(numNeuronsToAdd=N_excit)

#~~~~~~~~~~
# We will position our network's excitatory neurons evenly over the full cylinder surface, 
# and position the inhibitory neurons randomly in a limited band around the midline of the cylinder:
#~~~~~~~~~~
network.geometry.position_neurons(positioning='even', neuronIDs=neuronIDs_excit)

#~~~~~~~~~~
# Let's take a look at our network's geometry and neuron positions. 
# (A number of network plots are provided, we use one here. Excitatory neurons are colored blue, inhibitory red):
#~~~~~~~~~~
# axsyn2d = pyplot.subplot()
# synapse_network_diagram_2d(axsyn2d, network)
# pyplot.show()


#~~~~~~~~~~
# Distances between neurons in our network were already calculated when the neurons were positioned:
#~~~~~~~~~~
#network.geometry.distances



#+++++++++++++++++++++++
# NETWORK CONNECTIVITY +
#+++++++++++++++++++++++

#~~~~~~~~~~
# Let's configure our synaptic connectivities such that probability of neuron adjacency decays with their distance. 
# Excitatory synaptic connections will have weights that decrease linearly in distance:
#~~~~~~~~~~

W_synE = generate_connectivity_vectors(neuronIDs=neuronIDs_excit, N=network.N, adjacencyScheme='distance_probability', initWeightScheme='distance', 
                                       args={'adj_prob_dist_fn':'exponential', 'p0_a':1.0, 'sigma_a':0.9,
                                             'init_weight_dist_fn':'linear', 'b_w':1.5, 'm_w':-0.2,
                                             'low':0.0, 'high':1.0,
                                             'distances':network.geometry.distances[neuronIDs_excit]} )

network.set_synaptic_connectivity(connectivity=W_synE, synapseType='e', updateNeurons=neuronIDs_excit)

#~~~~~~~~~~
# Let's take a look at our network's synaptic connectivity:
#~~~~~~~~~~
# axsyn2d = pyplot.subplot()
# synapse_network_diagram_2d(axsyn2d, network)
# pyplot.show()

#~~~~~~~~~~
# We can also configure our network's gap junction connectivity. 
# For gap junctions, let's use a nearest-neighbors adjacency scheme and a uniformly distributed connection weight. 
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
                                               adjacencyScheme='nearest_neighbors', initWeightScheme='uniform', 
                                               args={'k':4,'low':0.75, 'high':1.25, 'distances':network.geometry.distances[int(nID)]}, 
                                               sparsity=heightPercentile )

network.set_gapjunction_connectivity(connectivity=W_synG)

#~~~~~~~~~~
# Now let's take a look at our network's gap junction connectivity:
#~~~~~~~~~~
# axgap2d = pyplot.subplot()
# gapjunction_network_diagram_2d(axgap2d, network)
# pyplot.show()



#+++++++++++++++++
# NETWORK INPUTS +
#+++++++++++++++++

#~~~~~~~~~~
# Let's allocate 1 inputs to our network:
#~~~~~~~~~~
numInputs = 1
network.add_inputs(numInputsToAdd=numInputs)

#~~~~~~~~~~
# Let's select 10 excitatory neurons randomly to be input neurons. 
# We will map our single input to these 10 selected input neurons.
#~~~~~~~~~~
# Randomly select 10 neurons from the set of excitatory neurons:
neuronIDs_inputs = numpy.random.choice(neuronIDs_excit, 10, replace=False)
network.neuronLabels[neuronIDs_inputs] = 'input'    # label the selected input neurons with 'input'
# Generate a matrix that maps the 1 input to the 10 selected input neurons:
W_inp = numpy.zeros(shape=(numInputs, network.N))   # 1xN matrix initialized full of 0's
W_inp[0, numpy.sort(neuronIDs_inputs)] 	= 1.0       # Set W[i,j] = 1 where i = 0 and where j is any index in the neuronIDs_inputs list
# Set the input connectivity to the matrix we generated:
network.set_input_connectivity(connectivity=W_inp)

# We will update our input values continuously throughout the simulation loop later on.

#~~~~~~~~~~
# Let's have the single input be a step input that injects a constant current to the input neurons for a period of time
# to initiate spiking activity, and then shuts off for the remainder of the simulation.
# We instantiate the corresponding input value generating object to generate values when setting input values each time step in the simulation loop later on:
#~~~~~~~~~~
stepInput 	= StepInput(stepVal=10.0, stepOnTime=0, stepOffTime=100)

#~~~~~~~~~~
#Let's take a moment to visualize the final constructed network, including positions of labeled input and output neurons:
#~~~~~~~~~~
# Input neurons are visualized as green triangles. Output neurons are visualized as orange squares.
# axsyn2d = pyplot.subplot()
# synapse_network_diagram_2d(axsyn2d, network)
# pyplot.show()



##########################
# SIMULATING THE NETWORK #
##########################

#++++++++++++++++++++++++++++
# SIMULATION INITIALIZATION +
#++++++++++++++++++++++++++++

#~~~~~~~~~~
#We begin by initializing a simulation with duration Tmax of 2000ms and time step  deltaT of 0.5ms:
#~~~~~~~~~~
network.initialize_simulation(T_max=1000, deltaT=0.05)



#++++++++++++++++++
# SIMULATION LOOP +
#++++++++++++++++++

print "[ RUNNING SIMULATION... ]"

# while within the allotted simulation time:
while(network.sim_state_valid()):

    # Set the input vals:
    network.set_input_vals( vals=[stepInput.val(network.t)] )

    # Advance the state of the network's neurons by integrating their dynamics:
    network.sim_step()

print "[ SIMULATION COMPLETE.  ]"

    

#+++++++++++++++++++++++++++++++
# SAVE SIMULATION DATA TO FILE +
#+++++++++++++++++++++++++++++++

print "[ Writing simualtion data to file... ]"

simData = network.get_neurons_dataframe()

simData.to_csv('tutorial_simplified_simdata.txt', sep='\t')



#++++++++++++++++
# VISUALIZATION +
#++++++++++++++++

# print "[ Generating simulation figures... ]"

# axraster = pyplot.subplot()
# spike_raster_plot(axraster, network, colorSynapseTypes=False, colorIOTypes=True)

# network_overview_figure(network, synapseDiagram2D=True, gapjunctionDiagram2D=True, spikerateDiagram2D=True)

# pyplot.show()

