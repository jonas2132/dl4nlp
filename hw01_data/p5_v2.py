import numpy as np

np.random.seed(seed=42)


# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------


def dataset_reader(file_path, dim):
    with open(file_path) as f:
        # Count number of records
        num_records = 0
        for i in enumerate(f):
            num_records += 1
        f.seek(0)
        # Create empty arrays according to requirements
        data_array = np.zeros((num_records, dim))
        label_array = np.zeros((num_records, 1))

        # Transform raw file data to perceptron input arrays
        for i, line in enumerate(f):
            _, label, embed = line.split("\t")
            embed_split = embed.split()
            embed_split = [np.float(x) for x in embed_split]
            data_array[i, :] = embed_split
            bias_vector = np.ones((num_records, 1))
            label_array[i, 0] = 1 if label == "label=POS" else 0

    # Add 1s-column for bias variable and return arrays
    data_array = np.c_[bias_vector, data_array]
    return data_array, label_array


# Load training, test and dev data
train_x, train_y = dataset_reader("DATA/rt-polarity.train.vecs", 100)
test_x, test_y = dataset_reader("DATA/rt-polarity.test.vecs", 100)
dev_x, dev_y = dataset_reader("DATA/rt-polarity.dev.vecs", 100)


# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------


def create_batches(x_data, y_labels, batch_size):
    # Shuffle perceptron inputs
    combined_array = np.c_[x_data, y_labels]
    np.random.shuffle(combined_array)
    x_data = combined_array[:, :-1]
    y_labels = combined_array[:, -1]

    # Split and return perceptron inputs in batches
    num_batches = len(y_labels) // batch_size
    x_data_batches_grouped = np.array_split(x_data, num_batches)
    y_labels_batches_grouped = np.array_split(y_labels, num_batches)
    return x_data_batches_grouped, y_labels_batches_grouped


def func_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def func_mini_batch_grad_desc(perc_out, x, y):
    return (perc_out - y) * (1 - perc_out) * perc_out * x


def func_square_loss(x, y):
    return sum([(x_prediction - y) ** 2 for x_prediction, y in zip(x, y)]) / len(y)


def train_perc(weights, x_data, y_labels, x_dev, y_dev, epochs=50, activation_func=func_sigmoid,
               optimizer_func=func_mini_batch_grad_desc, loss_func=func_mini_batch_grad_desc,
               learning_rate=0.01, batch_size=10, print_epoch_benchm=False):
    def epoch(weights_init, x_data=x_data, y_labels=y_labels, activation_func=activation_func,
              optimizer_func=optimizer_func, learning_rate=learning_rate, batch_size=batch_size):
        weights = weights_init

        # Create batches according to batch size
        x_data_batches_grouped, y_labels_batches_grouped = create_batches(x_data, y_labels, batch_size)

        # Calculate optimized weights across all batches in dataset
        for x_data_batch, y_label_batch in zip(x_data_batches_grouped, y_labels_batches_grouped):
            batch_optimization_val = 0
            # Calculate optimization value across all samples in batch
            for x_sample, y in zip(x_data_batch, y_label_batch):
                # Calculate perceptron output for sample
                sample_perc_out = activation_func(np.dot(x_sample, weights))
                # Calculate optimization value of sample and add to batch optimization value
                sample_optimization = optimizer_func(sample_perc_out, x_sample, y)
                batch_optimization_val += sample_optimization

            # Average batch optimization value across samples for updating weights
            batch_optimization_val /= len(y_label_batch)
            weights -= batch_optimization_val * learning_rate

        return weights

    # Update weights for each epoch and pass them on to next epoch and benchmark epoch progress
    for i in range(epochs):
        weights = epoch(weights)
        accuracy_train, loss_train = benchmark_perc(weights, x_data, y_labels, activation_func, loss_func)
        accuracy_dev, loss_dev = benchmark_perc(weights, x_dev, y_dev, activation_func, loss_func)
        if print_epoch_benchm:
            print(f"Epoch No. {i}\t|||\tTraining Loss: {loss_train}\tTraining Accuracy: {accuracy_train}"
                  f"\tDev Loss: {loss_dev}\tDev Accuracy: {accuracy_dev}")

    return weights


def benchmark_perc(weights, x, y, activation_func=func_sigmoid, loss_func=func_square_loss):
    # Calculate predictions with given weights
    predictions = [activation_func(np.dot(x_sample, weights)) for x_sample in x]
    predictions_rnd = np.round(predictions)

    # Calculate accuracy as percentage of correct predictions
    accuracy = sum([predictions_sample == y_sample for predictions_sample, y_sample
                    in zip(predictions_rnd, y)]) / len(y)

    # Calculate loss with output and predicted output
    loss = loss_func(predictions, y)

    return accuracy, loss


# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------


def training_assistant(x_data, y_labels, x_dev, y_dev, x_test, y_test, dim, epochs=50, activation_func=func_sigmoid,
                       optimizer_func=func_mini_batch_grad_desc, loss_func=func_square_loss,
                       learning_rate=0.01, batch_size=10, print_epoch_benchm=False):
    # Initialize weights randomly
    weights_init = np.random.normal(0, 1, dim + 1)

    # Train perceptron
    weights = train_perc(weights_init, x_data, y_labels, x_dev, y_dev, epochs, activation_func,
                         optimizer_func, loss_func, learning_rate, batch_size, print_epoch_benchm)

    # Calculate accuracy and loss and print results
    accuracy_dev, loss_dev = benchmark_perc(weights, x_dev, y_dev, activation_func, loss_func)
    accuracy_test, loss_test = benchmark_perc(weights, x_test, y_test, activation_func, loss_func)
    print("--------------------------------------------------------------")
    print(f"Process finished successfully after {epochs} Epochs!")
    print(f"Dev Loss: {loss_dev}\tDev Accuracy: {accuracy_dev}")
    print(f"Test Loss: {loss_test}\tTest Accuracy: {accuracy_test}")
    print("--------------------------------------------------------------")


# --------------------------------------------------------------
# Trial 1: parameters from homework sheet
# print("Trial 1: parameters from homework sheet")
training_assistant(train_x, train_y, dev_x, dev_y, test_x, test_y, 100, epochs=50, learning_rate=0.01, batch_size=10)
# Output:
# Process finished successfully after 50 Epochs!
# Dev Loss: [0.49046587]	Dev Accuracy: [0.50969356]
# Test Loss: [0.50338258]	Test Accuracy: [0.49593496]

# --------------------------------------------------------------
# Trial 2: more epochs, faster learning rate, larger batches
# print("Trial 2: more epochs, faster learning rate, larger batches")
training_assistant(train_x, train_y, dev_x, dev_y, test_x, test_y, 100, epochs=400, learning_rate=0.05, batch_size=50)
# Output:
# Process finished successfully after 400 Epochs!
# Dev Loss: [0.28076832]	Dev Accuracy: [0.71544715]
# Test Loss: [0.29292858]	Test Accuracy: [0.70419012]
