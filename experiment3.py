import argparse
from itertools import accumulate
import network
import data
import image
import numpy as np
import matplotlib.pyplot as plt

# This is the experiment of softmax regression
def main(hyperparameters):
    # Get the trainset and testset
    trainset = data.load_data(train = True)
    testset = data.load_data(train = False)

    # Normalize and append bias to the data, and one hot encode target
    X_train, y_train = trainset
    X_train, (_, _) = hyperparameters.normalization(X_train)
    X_train = data.append_bias(X_train)
    y_train = data.onehot_encode(y_train)
    trainset = X_train, y_train

    X_test, y_test = testset
    X_test, (_, _) = hyperparameters.normalization(X_test)
    X_test = data.append_bias(X_test)
    y_test = data.onehot_encode(y_test)
    testset = X_test, y_test

    # Shuffle the trainset
    trainset = data.shuffle(trainset)

    # Initialize the model
    model = network.Network(hyperparameters, activation = network.softmax, loss = network.multiclass_cross_entropy, out_dim = 10)

    if hyperparameters.validation == True: # K-fold cross-validation mode
        folds = data.generate_k_fold_set(trainset, k = hyperparameters.k_folds)
        train_losses_vs_epoch = [] # A list of k lists that each list contains the information of train loss of fold k
        val_losses_vs_epoch = [] # A list of k lists that each list contains the information of val loss of fold k
        accumulated_val_accuracy = 0.0

        for fold in folds:
            print("Starting a new fold...")
            train_subset, val_subset = fold

            # Initialize the model
            model = network.Network(hyperparameters, activation = network.softmax, loss = network.multiclass_cross_entropy, out_dim = 10)

            # Training with SGD
            train_loss_vs_epoch = []
            val_loss_vs_epoch = []
            consecutive_diverge_epochs = 0

            print("Training...")
            for epoch in range(hyperparameters.epochs):
                train_subset = data.shuffle(train_subset)
                batches = data.generate_minibatches(train_subset, batch_size = hyperparameters.batch_size)
                for batch in batches:
                    model.train(batch)

                train_loss, _ = model.test(train_subset)
                val_loss, val_accuracy = model.test(val_subset)
                train_loss_vs_epoch.append(train_loss / len(train_subset))
                val_loss_vs_epoch.append(val_loss / len(train_subset))

                # Early stop if the loss on validation set haven't decreased for 2 consecutive epochs
                if epoch == 0:
                    continue

                if val_loss_vs_epoch[-1] >= val_loss_vs_epoch[-2]:
                    consecutive_diverge_epochs += 1
                else:
                    consecutive_diverge_epochs = 0
                
                if consecutive_diverge_epochs >= 2:
                    break
            print("Finish training.")

            train_losses_vs_epoch.append(train_loss_vs_epoch)
            val_losses_vs_epoch.append(val_loss_vs_epoch)
            accumulated_val_accuracy += val_accuracy

        # Print average accuracy on validation sets
        print(f"Average accuracy on validation sets with {hyperparameters.k_folds} folds: {accumulated_val_accuracy / hyperparameters.k_folds}")

        # Plot average loss against epoch for train set and validation set
        for i in range(hyperparameters.k_folds):
            plt.figure(i)
            plt.title(f'Average Loss against Training Epoch, Fold {i}')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.plot(train_losses_vs_epoch[i], label = 'train')
            plt.plot(val_losses_vs_epoch[i], label = 'val')
            plt.legend()
        plt.show()
    else: # Normal mode
        # Initialize the model
        model = network.Network(hyperparameters, activation = network.softmax, loss = network.multiclass_cross_entropy, out_dim = 10)

        # Training with SGD
        print("Training...")
        for epoch in range(hyperparameters.epochs):
            trainset = data.shuffle(trainset)
            batches = data.generate_minibatches(trainset, batch_size = hyperparameters.batch_size)
            for batch in batches:
                model.train(batch)
        print("Finish training.")

        _, accuracy = model.test(testset)
        print(f"Accuracy on test set: {accuracy}")

        # Visualize the weights for each class
        weights = model.weights[1:, :].T # Remove bias
        weights -= np.min(weights, axis = 1).reshape(-1, 1)
        weights = weights / np.max(weights, axis = 1).reshape(-1, 1) * 255
        weights = weights.reshape(10, 28, 28)
        weights = np.uint8(weights)
        for i, weight in enumerate(weights):
            image.export_image(weight, name = f'weights_{i}.png')

parser = argparse.ArgumentParser(description = 'CSE151B PA1')
parser.add_argument('--batch-size', type = int, default = 32,
    help = 'input batch size for training (default: 64)')
parser.add_argument('--epochs', type = int, default = 100,
    help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', type = float, default = 0.001,
    help = 'learning rate (default: 0.001)')
parser.add_argument('--z-score', dest = 'normalization', action='store_const',
    default = data.min_max_normalize, const = data.z_score_normalize,
    help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--k-folds', type = int, default = 10,
    help = 'number of folds for cross-validation')
parser.add_argument('--validation', type = bool, default = False,
    help = 'enable cross-validation mode')

hyperparameters = parser.parse_args()
main(hyperparameters)
