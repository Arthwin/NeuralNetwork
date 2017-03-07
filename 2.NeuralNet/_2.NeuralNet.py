from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #seed the random number generator so it generates the same numbers everytime
        random.seed(1)
        #we model a single neuron with 3 input connections and 1 output connection
        #we assign random wights to a 3*1 matrix with values in the range oof -1 to 1
        #and mean 0
        self.synaptic_weights = 2 * random.random((3,1))-1

    #the sigmoid function describes an s shaped curve we pass the weighted sum of the inputs
    #through this function to normalize them between 0 and 1
    def __sigmoid(self,x):
        return 1/(1+exp(-x))

    #gradient of sigmund 
    def __sigmoid_derivative(self,x):
        return x * (1-x)

    def train(self, trainning_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            #calculate error
            error = training_set_outputs - output
            #multiply the error by the input and again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T, error + self.__sigmoid_derivative(output))
            #adjust the weights
            self.synaptic_weights += adjustment

    def think(self,inputs):
        #pass inputs through ourneural net
        return self.__sigmoid(dot(inputs,self.synaptic_weights))

    


if __name__ == '__main__':
    #initialise a single neuron neural net
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights: ")

    #training set. we have 4 eamples, each consisting of 3 input values and 1 output
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #train net using the set, iterate 10k times and adjust each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic wights after training: ")
    print(neural_network.synaptic_weights)

    #test the neural network with a new case
    print("Consider new situation [1,0,0]")
    print(neural_network.think([1,0,0]))