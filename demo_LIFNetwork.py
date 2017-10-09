# from __future__ import division

# import numpy as numpy
# import pandas as pandas
# from matplotlib import pyplot as pyplot

# import matplotlib.gridspec as gridspec

# import plotly
# from plotly.graph_objs import *

# from LIFNetwork import LIFNetwork
# import NetworkConnectivity
# from MidpointNormalize import MidpointNormalize




network 	= LIFNetwork()

network.add_neurons()

network.geometry = SpheroidSurface(r_xy=10, r_z=10)
network.geometry.position_neurons(coordinates='random')

network.set_synaptic_connectivity()
network.set_gapjunction_connectivity()
network.set_input_connectivity()

network.label_neurons(neuronIDs=[], label='input')
network.label_neurons(neuronIDs=[], label='output')

# network.initialize_simulation()

simulation 	= NetworkSimulation(network, )

simulate_periodic_pulse_input_sim(network, inputPeriod=2, inputValue=1.0)
