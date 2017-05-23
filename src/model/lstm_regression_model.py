import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import pickle

from IPython import display
from matplotlib import pyplot as plt

import cntk as C
import cntk.axis
from cntk.layers import Dense, Dropout, Recurrence

class LSTMRegressionModel:
    BATCH_SIZE_DEFAULT = 1
    EPOCHS_DEFAULT = 1500

    def create_model(self, x, N):
        """Create the model for time series prediction"""
        with C.layers.default_options(initial_state = 0.1):
            m = C.layers.Recurrence(C.layers.LSTM(N))(x)
            m = C.ops.sequence.last(m)
            m = C.layers.Dropout(0.2, seed=1)(m)
            m = cntk.layers.Dense(3, activation=C.ops.softmax)(m)
            return m

    def __init__(self, train_data, val_data=None, test_data=None, window_size=3, batch_size=-1):
        self.train_x, self.train_y = train_data
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y) / 100.0
        print(self.train_x)
        print(self.train_y)

        if val_data is not None:
            self.val_x, self.val_y = val_data
            self.val_x = np.array(self.val_x)
            self.val_y = np.array(self.val_y) / 100.0

        if test_data is not None:
            self.test_x, self.test_y = test_data
            self.test_x = np.array(self.test_x)
            self.test_y = np.array(self.test_y) / 100.0

        self._model = None
        self.window_size = window_size

        if batch_size == -1:
            self.batch_size = LSTMRegressionModel.BATCH_SIZE_DEFAULT
        else:
            self.batch_size = batch_size

        train_x_temp = self.train_x
        train_y_temp = self.train_y
        assert(len(train_x_temp) == len(train_y_temp))

        # Select the right target device when this notebook is being tested:
        """
        if 'TEST_DEVICE' in os.environ:
            if os.environ['TEST_DEVICE'] == 'cpu':
                C.device.try_set_default_device(C.device.cpu())
            else:
                C.device.try_set_default_device(C.device.gpu(0))
        """

        self.x_seq = C.sequence.input(shape=self.train_x.shape[1])
        N = 5
        self._model = self.create_model(self.x_seq, N)

    def _next_batch(self):
        i = 0
        while i < len(self.train_x):
            new_batch_x = np.array(self.train_x[i : i + self.batch_size])
            new_batch_y = np.array(self.train_y[i : i + self.batch_size])

            new_batch_x_len = len(new_batch_x)
            remaining_examples = self.batch_size - new_batch_x_len
            if remaining_examples:
                j = 0
                while j < remaining_examples:
                    new_batch_x = np.append(new_batch_x, [ self.train_x[j % len(self.train_x)] ], axis=0)
                    new_batch_y = np.append(new_batch_y, [ self.train_y[j % len(self.train_y)] ], axis=0)
                    j += 1
            yield (new_batch_x, new_batch_y)
            i += self.batch_size

    def train(self, save_intermediates=False, save_prefix=''):
        x = self.x_seq
        labels = C.input(shape=(self.batch_size, 3), name="y")
        #labels = C.input_variable(3)

        # the learning rate
        learning_rate = 0.001
        lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)

        # loss function
        loss = C.squared_error(self._model, labels)

        # use squared error to determine error for now
        error = C.squared_error(self._model, labels)

        # use adam optimizer
        momentum_time_constant = C.momentum_as_time_constant_schedule(
                LSTMRegressionModel.BATCH_SIZE_DEFAULT / -math.log(0.9)) 
        learner = C.fsadagrad(self._model.parameters,
                              lr = lr_schedule,
                              momentum = momentum_time_constant,
                              unit_gain = True)
        trainer = C.Trainer(self._model, (loss, error), [learner])
        
        # train
        loss_summary = []
        start = time.time()
        f, a = plt.subplots(3, 2, figsize=(12, 12))
        x_axis = list(range(len(self.train_x)))

        epoch_list = list(range(0, LSTMRegressionModel.EPOCHS_DEFAULT))
        train_loss_epochs = [0.0] * len(epoch_list)
        val_loss_epochs = [0.0] * len(epoch_list)
        test_loss_epochs = [0.0] * len(epoch_list)
        
        for epoch in epoch_list:
            for x1, y1 in self._next_batch():
                trainer.train_minibatch({x: x1, labels: y1})

            val_loss = 0.0
            for idx, example in enumerate(self.val_x):
                y = np.array([self.val_y[idx]])
                example = np.array([example])
                loss_amt = loss.eval({x: example, labels: y})[0]
                val_loss += loss_amt
            #print("val loss is {}".format(val_loss))
            val_loss_epochs[epoch] = val_loss

            test_loss = 0.0
            for idx, example in enumerate(self.test_x):
                y = np.array([self.test_y[idx]])
                example = np.array([example])
                loss_amt = loss.eval({x: example, labels: y})[0]
                test_loss += loss_amt
            #print("test loss is {}".format(test_loss))
            test_loss_epochs[epoch] = test_loss
            training_loss = trainer.previous_minibatch_loss_average
            train_loss_epochs[epoch] = training_loss
            if epoch % (100) == 0:
                if save_intermediates:
                    self.save(save_prefix + 'WIP-training-epoch-{}.dat'.format(epoch))
                loss_summary.append(training_loss)
                evaluation = []
                for example in self.train_x:
                    prediction = self._model.eval({x: example})[0].tolist()
                    evaluation.append(prediction)

                evaluation = np.array(evaluation)
                a[0][0].cla()
                a[1][0].cla()
                a[2][0].cla()
                a[0][1].cla()
                a[1][1].cla()
                a[2][1].cla()
                a[0][0].plot(x_axis, evaluation[:, 0], label='approval')
                a[0][0].plot(x_axis, self.train_y[:, 0], label='approval actual')
                a[0][0].set_title("Approval rating prediction over training set")
                a[0][0].legend()
                a[1][0].plot(x_axis, evaluation[:, 1], label='disapproval')
                a[1][0].plot(x_axis, self.train_y[:, 1], label='disapproval actual')
                a[1][0].set_title("Dispproval rating prediction over training set")
                a[1][0].legend()
                a[2][0].plot(x_axis, evaluation[:, 2], label='neutral')
                a[2][0].plot(x_axis, self.train_y[:, 2], label='neutral actual')
                a[2][0].set_title("Neutral rating prediction over training set")
                a[2][0].legend()

                a[0][1].plot(epoch_list, train_loss_epochs)
                a[0][1].set_title("Training loss vs. epochs")
                a[1][1].plot(epoch_list, val_loss_epochs)
                a[1][1].set_title("Validation loss vs. epochs")
                a[2][1].plot(epoch_list, test_loss_epochs)
                a[2][1].set_title("Test loss vs. epochs")

                for axes in a:
                    axes[0].set_xlabel('training example number/idx')
                    axes[0].set_ylabel('rating (% represented as fraction)')
                    axes[1].set_xlabel('epoch number')
                    axes[1].set_ylabel('MSE loss')

                display.clear_output(wait=True)
                display.display(plt.gcf())

        print("training took {0:.1f} sec".format(time.time() - start))

    def predict(self, x_in):
        return self._model.eval({x: x_in})

    def evaluate(self, x_in, y_out):
        raise NotImplementedError

    def save(self, filename):
        self._model.save(filename)

    def load(filename):
        return joblib.load(filename)

