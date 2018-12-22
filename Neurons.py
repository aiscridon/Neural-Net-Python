import random
import math

#   The declaration of the following constants will become clear later in the code

neuron_weight_size = 10
neuron_weight_scale = 10
neuron_learningrate = 0.2

# We define a neuron
# A neuron has a name, to be able to link it with other neurons
# A neuron has an internal value which will be passed to other neurons, we call this the potential
# A parent neuron has subconnections to other neurons where their potentials will be scaled with weights
# and will be summed to be the parent neuron potential
# We define a dictionary where we will keep these connections, in biology we call these connections dendrites,
# the connections will be defined below
# When we want the neuron to learn something, we want its potential to equal our solution
# Ofcourse there will be an error, this will need to get passed to other functions, see below

# The neuron class
class Neuron():
    def __init__(self, name):
        self.name = name
        self.potential = 0
        self.dendrites = {}
        self.error = 0



# A neuron's output in biology is an axon
# The axon has as members the parent neuron and the weight that will rescale its potential
# before it will be passed to the parent neuron
# The initial weights will be random, neuron_weight_size and neuron_weight_scale give us some control over this

class Axon():
    def __init__(self, neuron):
        self.neuron = neuron
        self.weight = random.randrange(-neuron_weight_size, neuron_weight_size + 1) / neuron_weight_scale



# We will now define functions for the basic manipulations of neurons and axons


# Axon manipulations:

# We can set the axon weight

def Dendrite_set_weight(axon, weight):
    axon.weight = weight

# When a neuron fires, its axon will return the potential multiplied by its weight

def Dendrite_fire(axon):
    return axon.weight * Neuron_calculate(axon.neuron)




# Neuron manipulations:

# We can set the internal potential of the neuron

def Neuron_set_potential(neuron, potential):
    neuron.potential = potential


# We can connect neurons, with or without a preset weight

def Neuron_connect(neuron1, neuron2, weight=None):
    # We create a dendrite-axon connection
    dendrite = Axon(neuron2)
    # If a preset weight was given we set the axon weight to this value
    if weight != None:
        Dendrite_set_weight(dendrite, weight)
    # We add the dendrite-axonconnection to the parent neuron's dendrites
    neuron1.dendrites[neuron2.name] = dendrite


# We often want to have solutions between 0 and 1 (to lessen to chance on infinity recurring in the weights, etc)
# The sigmoid function has nice mathematical properties for this

# We rescale a potential with the sigmoid function f(x) = 1 / (1 + e^(-x))
# we also need its derivative, this will become clear later in the code

def sigmoid(potential, derivative=False):
    # The normal sigmoid
    if not derivative:
        # 1/e^100 equals zero, because e^100 is to large for a computer
        if potential > -100:
            return 1 / (1 + math.exp(-potential))
        else:
            return 0
    # the derivative of the sigmoid (The given potential will already be scaled in this case)
    else:
        return potential * (1 - potential)

# Calculate the neurons potential with its connections

def Neuron_calculate(neuron):
    # if there are connections
    if len(neuron.dendrites) != 0:
        x = 0
        # we sum all the subneurons potentials and weights
        for dendrite in neuron.dendrites:
            x += Dendrite_fire(neuron.dendrites[dendrite])
        # we scale this solution with the sigmoid function
        neuron.potential = sigmoid(x)
    return neuron.potential


# We can add multiple errors to the neuron in case it's a neuron from a hidden layer
# which can have multiple errors given different end results

def Neuron_add_error(neuron, error):
    neuron.error += error


# We can correct a neuron given its error, we need to recalculate the weights of its dendrites
# because the error comes from the wrong weights of the dendrites

# suppose we have 3 layers of neurons l1, l2 and l3
# l1 is the input layer and every neuron is connected to all the neurons in l2
# which in its turn is connected to the output layer l3
# the layers are actually vectors and the connections are actually matrices
# so Sigmoid(l1*A1) = l2 and Sigmoid(l2*A2) = l3
# so Sigmoid(Sigmoid(l1*A1)*A2) = l3

# Now some way simplified calculus

# f(x)_S = y a Surface of functions each for a different matrix S
# f(l1)_A1 = l2
# we SUPPOSE that our correct function f_A is on this surface of S's
# f(x)_A = f(x)_S0 + d(f(x)_S)/dS * dS + O(dS²)...  linear approximation of f_A
# dS = f_A - f_S0  our error in S
# so that dS = d(f_S)/ dS * (f_A - f_S0) is our error in S    (backpropagation)
# and S is matrix A1 or S is matrix A2
# f_A is our desired solution and f_S0 is our calculation
# suppose the correct solution is f_A = Y
# our starting point on the surface f_S0 = l3
# so that
# 1)  dA2 = (Y - l3) * d(Sigmoid(Sigmoid(l1*A1)*A2))/d(A2)
# and
# 2)  dA1 = (Y - l3) * d(Sigmoid(Sigmoid(l1*A1)*A2))/d(A1)
# 3)  the derivative of the Sigmoid is equal to Sigmoid * (1 - Sigmoid)
# to get to the following code is an excersize for the reader given 1,2 and 3

def Neuron_correct(neuron):
    gradient = neuron.error * sigmoid(neuron.potential, derivative=True)
    for dendrite in neuron.dendrites:
        delta_weight = neuron_learningrate * neuron.dendrites[dendrite].neuron.potential * gradient
        neuron.dendrites[dendrite].weight += delta_weight
        Neuron_add_error(neuron.dendrites[dendrite].neuron, gradient * neuron.dendrites[dendrite].weight)
        Neuron_correct(neuron.dendrites[dendrite].neuron)
    # reset the error
    neuron.error = 0
    # calculate the new potential
    Neuron_calculate(neuron)


# A test example of how to use this code

def Test_Neuron():
    print(' ')
    print(' ')
    print('Test Neuron')
    print(' ')
    print(' ')
    # create 3 neurons
    n1 = Neuron(1)
    n2 = Neuron(2)
    n3 = Neuron(3)
    # the first 2 are the inputs so we give them an initial potential
    n1.potential = 0.5
    n2.potential = 0.5
    # we connect the third one to the first two and give it some weigths(not needed explicitly)
    Neuron_connect(n3, n1, 2)
    Neuron_connect(n3, n2, 3)

    # we imagine a solution y
    y = 0.1234567890123456789
    # we will reapproximate multiple times
    for i in range(5001):
        # we get the current n3 potential
        s0 = Neuron_calculate(n3)
        # calculate the error
        e = y - s0
        # add the error to the n3 error
        Neuron_add_error(n3, e)
        # and calculate the correction
        Neuron_correct(n3)
        # give solutions in between
        if i % 500 == 0:
            print(i, n3.potential)
    print(' ')
    print('final solution')
    print(Neuron_calculate(n3))
    print(' ')
    print(' ')
    print('End Test Neuron')
    print(' ')
    print(' ')

# play around with the learningrate to get more accurate results

Test_Neuron()