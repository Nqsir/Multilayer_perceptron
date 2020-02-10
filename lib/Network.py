import numpy as np

from lib.Activations import softmax
from lib.Losses import cross_entropy


class Network:
    """
    A class that contains all our model
    """
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.loss_list = []
        self.val_loss_list = []
        self.acc_list = []
        self.val_acc_list = []
        self.mean_list = []
        self.val_mean_list = []

    # Add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # Set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predict output
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # Run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # Return accuracy
    def accuracy(self, df_x, df_y):
        # rearrange data
        out = np.asarray(self.predict(df_x)).reshape(len(df_x), 2)
        df_y = list(df_y)
        df_y = np.asarray(df_y).reshape(len(df_y), 2)
        tab_y_pred = []

        # Prediction accuracy
        valid = 0
        index = 0
        for prediction in out:
            prediction = softmax(prediction)
            tab_y_pred.append(np.amax(prediction))
            verification = df_y[index]
            if prediction[0] > prediction[1]:
                prediction[0] = 1
                prediction[1] = 0
            else:
                prediction[0] = 0
                prediction[1] = 1

            if np.array_equal(prediction, verification):
                valid += 1
            index += 1
        accuracy = valid * 100 / index

        return accuracy

    # Train network
    def fit(self, x_train, y_train, x_test, y_test, epoch, learning_rate, error_rate):
        samples = len(x_train)
        samples_val = len(x_test)

        # Training loop
        run = 1
        output_val = 0
        nb_iter = 1

        while run:
            loss = 0
            val_loss = 0
            loss_cross = 0
            for j in range(samples):
                # Forward propagation
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                loss += self.loss(y_train[j], output)
                loss_cross += cross_entropy(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            for j in range(samples_val):
                output_val = x_test[j]
                for layer in self.layers:
                    output_val = layer.forward_propagation(output_val)
                val_loss += self.loss(y_test[j], output_val)

            # Calculates metrics (loss, val_loss, accuracy...)
            self.loss_list.append(loss / samples)
            self.val_loss_list.append(val_loss / samples_val)
            self.acc_list.append(self.accuracy(x_train, y_train))
            self.val_acc_list.append(self.accuracy(x_test, y_test))

            print(f'epoch {nb_iter:{len(str(epoch))}}/{epoch}'
                  f'    loss = {self.loss_list[-1]:.4f}'
                  f'    loss_cross = {(loss_cross / samples):.4f}'
                  f'    val_loss = {self.val_loss_list[-1]:.4f}'
                  f'    accuracy = {self.acc_list[-1]:.4f}'
                  f'    val_accuracy = {self.val_acc_list[-1]:.4f}')

            # stop condition
            if self.loss_list[-1] <= error_rate or nb_iter >= epoch:
                run = 0

            nb_iter += 1
