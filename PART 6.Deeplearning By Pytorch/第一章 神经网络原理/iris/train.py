import numpy as np
from chapter1.iris import linear_classify

# number of input, output nodes,and learning rate
input_nodes = 4
output_nodes = 2
learning_rate = 0.01
# epochs is the number of times the training data set is used for training
epochs = 1000
# create instance of linear network
n = linear_classify.linearNetwork(input_nodes, output_nodes, learning_rate)

# load the iris training data into a list
training_data_file = open("iris_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the linear network
for e in range(epochs):
    # go through all records in the training data set
    print('start training, epoch:', e)
    for record in training_data_list:
        # split the record by the ',' commas
        record = record.replace('\n', '').split(',')
        record = [float(data) for data in record]
        input_data = record[0:4]
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets_list = [0.01, 0.99]
        """
        targets = int(record[4:][0])
        # create the target output values (all 0.01, except the desired label which is 0.99)
        if targets == 1:
            targets_list = [0.01, 0.99]
        else:
            targets_list = [0.99, 0.01]
        """
        n.train(input_data, targets_list)
        pass
    np.save('wih.npy', n.wih)
    pass
