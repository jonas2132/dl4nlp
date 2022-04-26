import numpy as np
import os


vector_dim = 100
# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------
def dataset_reader(dataset):
    file = open(dataset, mode='r')
    file_length = sum(1 for line in file)
    data = np.empty((file_length, vector_dim))
    labels = np.empty((file_length,1))
    file.seek(0)

    for i, line in enumerate(file):
        text, label, vector = line.split(sep='\t')
        if label == 'label=POS':
            labels[i] = 1
        else:
            labels[i] = 0
        vec = vector.split()

        for x, empty_vec in enumerate(data[i]):
            data[i][x] = vec[x]

    return data, labels
# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    dataset_reader('DATA/rt-polarity.dev.vecs')