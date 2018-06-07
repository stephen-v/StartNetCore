import numpy
import scipy.special


# linear network classify definition
class linearNetwork:
    # initialise the linear network
    def __init__(self, inputnodes, outputnodes, learningrate):
        # set number of nodes in each input, output layer
        self.inodes = inputnodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.onodes, self.inodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the linear network
    def train(self, inputs_list, targets_list):
        # linear output
        linear_output = numpy.dot(self.wih, inputs_list)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(linear_output)

        # output layer error is the (target - actual)
        output_errors = targets_list - final_outputs

        # update the weights for the links between the input and final layers
        self.wih += self.lr * numpy.dot((output_errors * linear_output * (1.0 - linear_output)),
                                        linear_output)

        pass

    # query the linear network
    def query(self, inputs_list):
        # calculate signals into linear output layer
        linear_output = numpy.dot(self.wih, inputs_list)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(linear_output)

        return final_outputs
