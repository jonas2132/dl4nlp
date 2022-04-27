import numpy as np
import os


vector_dim = 100
# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------
def dataset_reader(dataset):
    file = open(dataset, mode='r')
    file_length = sum(1 for line in file)
    data = np.empty((file_length, vector_dim + 1))
    labels = np.empty((file_length,1))
    file.seek(0)

    for i, line in enumerate(file):
        text, label, vector = line.split(sep='\t')
        if label == 'label=POS':
            labels[i] = 1
        else:
            labels[i] = 0
        vec = vector.split()
        vec.append(1)

        for x, empty_vec in enumerate(data[i]):
            data[i][x] = vec[x]

    return data, labels
# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------
def split_batches(data, labels, batch_size):
    # permutate the arrays simultaneously and split them in batches of equal sizes
    permutation = np.random.permutation(len(labels))
    X = data[permutation]
    Y = labels[permutation]
    X = np.array_split(X, len(labels)/batch_size)
    Y = np.array_split(Y, len(labels)/batch_size)
    return X, Y

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def train_perceptron(initial_w, train_x, train_y, epochs, batch_size, learning_rate):

    def epoch(train_x, train_y, w_input, batch_size, learning_rate):
        train_x_batches, train_y_batches = split_batches(train_x, train_y, batch_size)

        for x_batch, y_batch in zip(train_x_batches, train_y_batches):
            sum = 0
            for x, y in zip(x_batch, y_batch):
                sigm = sigmoid(np.dot(w_input, x))
                sum += (sigm-y) * sigm * (1 - sigm) * x
            w_input = w_input - (learning_rate/len(y_batch)) * sum
        return w_input

    for i in range(epochs):
        trained_w = epoch(train_x, train_y, initial_w, batch_size, learning_rate)

        print(trained_w)
    return

def evaluate_perceptron():

    return
# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------
def run_perceptron():
    data, labels = dataset_reader('DATA/rt-polarity.dev.vecs')
    w = np.random.normal(0, 1, (vector_dim + 1))
    print('Run perceptron with initial parameters: ')
    train_perceptron(w, data, labels, epochs=50, batch_size=10, learning_rate=0.01)




if __name__ == "__main__":
    run_perceptron()
