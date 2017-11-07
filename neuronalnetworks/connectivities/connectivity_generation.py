import numpy as numpy

# adjacency criteria:
#X	nonspatial_probability			[prob]
#X	nonspatial_degree				[degree]
#X 	nearest_neighbors				[k]
#X 	distance_threshold				[d_thresh]
#X 	distance_probability
#X		linear						[function, distances, m, b]
#X 		exponential					[function, distances, p0, sigma]

# initial weight distribution
#X 	constant						[c_w]
#X 	uniform							[min, max]
#X 	normal							[mean, std]
#X 	by distance
#X		linear						[function, distances, b, m]
#X 		exponential					[function, distances, p0, sigma]
# 		1/|d|?						[function, distances, ?]
# 		1/d^2?						[function, distances, ?]

# sparsity

def exit_on_connectivity_error(error_msg):
	print "NetworkConnectivity Error:" + error_msg
	exit()

def generate_connectivity_vector(N, adjacencyScheme, initWeightScheme, args={}, sparsity=0.0, selfLoops=False, neuronID=None):

	#~~~~~~~~~~~~~~~~~~~~~~
	# Establish adjacency ~
	#~~~~~~~~~~~~~~~~~~~~~~
	adjacencyVector 	= numpy.zeros(N)

	if(adjacencyScheme.lower() == 'nonspatial_probability'):
		try:
			p 				= args['prob']
			adjacencyVector	= (numpy.random.rand(N) < p).astype(int)
		except KeyError:
			exit_on_connectivity_error("When using 'adjacencyScheme = nonspatial_probability', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'prob':<float>}")

	elif(adjacencyScheme.lower() == 'nonspatial_degree'):
		try:
			degree 		= args['degree']
			# Randomly choose degree-amount of adjacent neurons, possibly without allowing this neuron to select itself:
			if(not selfLoops and neuronID is not None):
				indices		= numpy.random.choice(numpy.setdiff1d(numpy.arange(N), numpy.array([neuronID])), degree, replace=False)
			elif(selfLoops):
				indices		= numpy.random.choice(range(N), degree, replace=False)
			else:
				exit_on_connectivity_error("In call to generate_connectivity_vector, selfLoops set to False (default), but no neuronID was given. Without a given neuronID, the index of the self-loop vector element is ambiguous.")
			adjacencyVector[indices] = 1
		except KeyError:
			exit_on_connectivity_error("When using 'adjacencyScheme = nonspatial_degree', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'degree':<int>}")

	elif(adjacencyScheme.lower() == 'nearest_neighbors'):
		try:
			k 			= args['k']
			distances 	= args['distances']
			# print "distances: " + str(distances)
			# Choose k adjacent neurons with the k-lowest distances to this neuron (always excluding this neuron to itself):
			if(neuronID is None):
				exit_on_connectivity_error("In call to generate_connectivity_vector, selfLoops set to False (default), but no neuronID was given. Without a given neuronID, the index of the self-loop vector element is ambiguous.")
			else:
				distances[neuronID]		= numpy.inf	# ensure that the distance from this neuron to itself appears largest
				indices 					= numpy.argpartition(distances, k)[:k]	# get the indices of the k smallest distances (argpartition only sorts until k smallest items have been sorted)
				adjacencyVector[indices]	= 1
		except KeyError:
			exit_on_connectivity_error("When using 'adjacencyScheme = nearest_neighbors', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'k':<int>, 'distances':list|<numpy.ndarray>}")

	elif(adjacencyScheme.lower() == 'distance_threshold'):
		try:
			d_thresh	= args['d_thresh']
			distances 	= args['distances']
			# print "distances: " + str(distances)
			indices 					= numpy.where(distances <= d_thresh)[0]
			adjacencyVector[indices]	= 1
		except KeyError:
			exit_on_connectivity_error("When using 'adjacencyScheme = distance_threshold', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'d_thresh':<int>, 'distances':list|<numpy.ndarray>}")

	elif(adjacencyScheme.lower() == 'distance_probability'):
		try:
			function	= args['adj_prob_dist_fn'].lower()
			distances 	= numpy.asarray(args['distances'])
			# print "-----------------------------------------"
			# print "distances: " + str(distances)
		except KeyError:
			exit_on_connectivity_error("When using 'adjacencyScheme = distance_probability', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'adj_prob_dist_fn':<string>, 'distances':list|<numpy.ndarray>}")
		if(function == 'linear'):
			b 	= args['b_a']
			m 	= args['m_a']
			adjProbs 		= (m*distances + b).clip(min=0.0, max=1.0)
			# print "adjProbs:  " + str(adjProbs)
			adjacencyVector	= (numpy.random.rand(N) < adjProbs).astype(int)
		elif(function == 'exponential'):
			p0 		= args['p0_a']
			sigma 	= args['sigma_a']
			adjProbs 	= p0*numpy.exp(-1*distances/sigma).clip(min=0.0, max=1.0)
			# print "adjProbs:  " + str(adjProbs)
			adjacencyVector	= (numpy.random.rand(N) < adjProbs).astype(int)
			# print "adjVectr:  " + str(adjacencyVector)

	elif(adjacencyScheme.lower() == 'given'):
		try:
			adjacencyVector	= args['given_adj']
		except KeyError:
			exit_on_connectivity_error("When using 'adjacencyScheme' = 'given', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'given_adj':list|<numpy.ndarray> (len=N)}")
		if(len(adjacencyVector) != N):
			exit_on_connectivity_error("When using 'adjacencyScheme' = 'given', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'given_adj':list|<numpy.ndarray> **(len=N)**}")

		adjacencyVector[adjacencyVector != 0.0] = 1

	else:
		exit_on_connectivity_error("'adjacencyScheme' "+str(adjacencyScheme)+" is unrecognized.")

	# print "adjacency: " + str(adjacencyVector)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Eliminate self-loops as needed ~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if(not selfLoops):
		if(neuronID is not None):
			# Zero out the adjacency element referring back to the specified neuron's self:
			adjacencyVector[neuronID]	= 0
		else:
			exit_on_connectivity_error("In call to generate_connectivity_vector, selfLoops set to False (default), but no neuronID was given. Without a given neuronID, the index of the self-loop vector element is ambiguous.")

	# print "adjnoloop: " + str(adjacencyVector)

	#~~~~~~~~~~~~~~~~~~~~~
	# Initialize weights ~
	#~~~~~~~~~~~~~~~~~~~~~
	weightVector	= numpy.zeros(N)

	if(initWeightScheme.lower() == 'constant'):
		try:
			c_w	= args['c_w']
			weightVector 	= numpy.full(N, c_w)
		except KeyError:
			exit_on_connectivity_error("When using 'initWeightScheme = constant', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'c_w':<float>}")

	elif(initWeightScheme.lower() == 'uniform'):
		try:
			low		= args['low']
			high	= args['high']
			weightVector 	= numpy.random.uniform(low=low, high=high, size=N)
		except KeyError:
			exit_on_connectivity_error("When using 'initWeightScheme = uniform', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'low':<float>, 'high':<float>}")

	elif(initWeightScheme.lower() == 'normal'):
		try:
			mean	= args['mean']
			std		= args['std']
			weightVector 	= numpy.random.normal(mean, std, size=N)
		except KeyError:
			exit_on_connectivity_error("When using 'initWeightScheme = normal', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'mean':<float>, 'std':<float>}")

	elif(initWeightScheme.lower() == 'distance'):
		try:
			function	= args['init_weight_dist_fn'].lower()
			distances 	= numpy.asarray(args['distances'])
			# print "distances: " + str(distances)
		except KeyError:
			exit_on_connectivity_error("When using 'initWeightScheme = distance', generate_connectivity_vector expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'init_weight_dist_fn':<string>, 'distances':list|<numpy.ndarray>}")
		if(function == 'linear'):
			b 	= args['b_w']
			m 	= args['m_w']
			weightVector	= (m*distances + b).clip(min=0.0, max=1.0)
		elif(function == 'exponential'):
			p0 		= args['p0_w']
			sigma 	= args['sigma_w']
			weightVector	= p0*numpy.exp(-1*distances/sigma)

	else:
		exit_on_connectivity_error("'initWeightScheme' "+str(initWeightScheme)+" is unrecognized.")


	# print "weights:   " + str(weightVector)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Generate mask for added sparsity ~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	sparsityMask	= numpy.zeros(N)
	indices			= numpy.random.choice(range(N), int(N*(1-sparsity)), replace=False)
	sparsityMask[indices]	= 1

	# print "sparsity:   " + str(sparsityMask)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Compile the connectivity vector: ~
	#	- non-adjacent neurons masked to have 0 weight, sparsity mask
	#	- sparsityMask introduces additional sparsity by clamping some other weights to 0
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	connectivityVector 	= weightVector*adjacencyVector*sparsityMask

	# print "connectivity: " +str(connectivityVector)

	return connectivityVector

def generate_connectivity_matrix(N, adjacencyScheme, initWeightScheme, args={}, sparsity=0.0, selfLoops=False):

	usingDistances	= False
	if(any(adjacencyScheme.lower() == a for a in ['nearest_neighbors', 'distance_probability', 'distance_threshold']) or initWeightScheme.lower() == 'distance'):
		usingDistances	= True
		try:
			allDistances	= numpy.copy( numpy.asarray(args['distances']) ) # deep copy the input distances matrix for future reference
			if(allDistances.shape[0] != N or allDistances.shape[1] != N):
				exit_on_connectivity_error("When using adjacencyScheme == ['nearest_neighbors'|'distance_probability'|'distance_threshold'] or initWeightScheme == 'distance', generateConnectivityMatrix() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'distances':<NxN list|numpy.ndarray>}")
		except KeyError:
			exit_on_connectivity_error("When using adjacencyScheme == ['nearest_neighbors'|'distance_probability'|'distance_threshold'] or initWeightScheme == 'distance', generateConnectivityMatrix() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'distances':<NxN list|numpy.ndarray>}")

	usingGivenAdjacencies = False
	if(adjacencyScheme.lower() == 'given'):
		usingGivenAdjacencies = True
		try:
			allGivenAdjacencies = numpy.copy( numpy.asarray(args['given_adj']) ) # deep copy the input given adjacencies vectors for future reference
			if(allGivenAdjacencies.shape[0] != N or allGivenAdjacencies.shape[1] != N):
				exit_on_connectivity_error("When using adjacencyScheme == 'given', generate_connectivity_vectors() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'given_adj':<NxN list|numpy.ndarray>}")
		except KeyError:
			exit_on_connectivity_error("When using adjacencyScheme == 'given', generate_connectivity_vectors() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'given_adj':<NxN list|numpy.ndarray>}")

	connectivityMatrix 	= numpy.zeros(shape=[N, N])

	for n in range(N):
		if(usingDistances):
			args['distances']	= allDistances[n]	# overwrite the 'distances' key in args to hold only the distances vector for neuron n

		if(usingGivenAdjacencies):
			args['given_adj']	= allGivenAdjacencies[n]	# overwrite the 'given_adj' key in args to hold only the given adjacencies vector for neuron n

		connectivityMatrix[n]	= generate_connectivity_vector(N, adjacencyScheme, initWeightScheme, args, sparsity, selfLoops, n)

	return connectivityMatrix


def generate_connectivity_vectors(neuronIDs, N, adjacencyScheme, initWeightScheme, args={}, sparsity=0.0, selfLoops=False):

	usingDistances	= False
	if(any(adjacencyScheme.lower() == a for a in ['nearest_neighbors', 'distance_probability', 'distance_threshold']) or initWeightScheme.lower() == 'distance'):
		usingDistances	= True
		try:
			allDistances	= numpy.copy( numpy.asarray(args['distances']) ) # deep copy the input distances matrix for future reference
			if(allDistances.shape[0] != len(neuronIDs) or allDistances.shape[1] != N):
				exit_on_connectivity_error("When using adjacencyScheme == ['nearest_neighbors'|'distance_probability'|'distance_threshold'] or initWeightScheme == 'distance', generateConnectivityMatrix() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'distances':<NxN list|numpy.ndarray>}")
		except KeyError:
			exit_on_connectivity_error("When using adjacencyScheme == ['nearest_neighbors'|'distance_probability'|'distance_threshold'] or initWeightScheme == 'distance', generateConnectivityMatrix() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'distances':<NxN list|numpy.ndarray>}")

	usingGivenAdjacencies = False
	if(adjacencyScheme.lower() == 'given'):
		usingGivenAdjacencies = True
		try:
			allGivenAdjacencies = numpy.copy( numpy.asarray(args['given_adj']) ) # deep copy the input given adjacencies vectors for future reference
			if(allGivenAdjacencies.shape[0] != len(neuronIDs) or allGivenAdjacencies.shape[1] != N):
				exit_on_connectivity_error("When using adjacencyScheme == 'given', generate_connectivity_vectors() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'given_adj':<neuronIDsxN list|numpy.ndarray>}")
		except KeyError:
			exit_on_connectivity_error("When using adjacencyScheme == 'given', generate_connectivity_vectors() expects the 'args' method argument to hold a dictionary with the following key-value pairs: {'given_adj':<neuronIDsxN list|numpy.ndarray>}")

	numVectors 	= len(neuronIDs)

	connectivityVectors 	= numpy.zeros(shape=[numVectors, N])

	for i, n in enumerate(neuronIDs):
		if(usingDistances):
			args['distances']	= allDistances[i]	# overwrite the 'distances' key in args to hold only the distances vector for neuron n

		if(usingGivenAdjacencies):
			args['given_adj']	= allGivenAdjacencies[i]	# overwrite the 'given_adj' key in args to hold only the given adjacencies vector for neuron n
			# print "+++++"
			# print args['given_adj']
			# print "-----"

		connectivityVectors[i]	= generate_connectivity_vector(N, adjacencyScheme, initWeightScheme, args, sparsity, selfLoops, n)

	return connectivityVectors


