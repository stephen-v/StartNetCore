from chapter1.iris import linear_classify
import numpy as np

# number of input, hidden and output nodes, learning rate
input_nodes = 4
output_nodes = 2
learning_rate = 0.01
# create instance of linear network
n = linear_classify.linearNetwork(input_nodes, output_nodes, learning_rate)

# load the iris training data and labels into a list(the type of the file is xlsx)
testing_data_file = open("iris_test.csv", 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()


# go through all the records in the test dataset
n.wih = np.load('wih.npy')
scorecard = []
for record in testing_data_list:
    record = record.replace('\n', '').split(',')
    record = [float(data) for data in record]
    input_data = record[0:4]
    # true label is the last one in record
    true_label = int(record[4:][0])
    # query the network
    outputs = n.query(input_data)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if label == true_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
