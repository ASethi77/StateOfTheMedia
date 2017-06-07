import keras
import tensorflow as tf
import numpy as np
import pandas
import pickle

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

class LSTMRegressionModel:
    BATCH_SIZE_DEFAULT  = 5 # how many samples to use per forward/backward feed (more takes more memory, but has better optima)
    EPOCHS_DEFAULT = 1500 # of times to optimize over the entire training set
    HISTORY_DEFAULT = 30 # how many days of features should we include in each feature vector
                         # NOTE: THIS NEEDS TO MATCH Config.DAY_RANGE, but cannot enforce this due to 
                         #       python circular dependencies
    SEED_DEFAULT = 7 # seed value to reproduce random elements

    def __init__(self, train_data, window_size = 30):
        print("Initializing data...")
        init_x, init_y = train_data
        if len(init_x) == 0:
            raise ValueError("No Training Samples Passed")
        if len(init_y) == 0:
            raise ValueError("No Training Labels Passed")
        self.num_features = len(init_x[0])
        self.train_x = np.array(init_x)
        self.train_y = np.array(init_y) / 100.0 # scale to range [0.0, 1.0]

        # re-shape the numpy arrays
        # (# batch, # samples, # features)
        self.train_x = self.train_x.reshape((len(init_x), self.history, self.num_features))
        self.train_y = self.train_y.reshape((len(init_y), 1, 3))
        
        if val_data is not None:
            self.val_x, self.val_y = val_data
            self.val_x = np.array(self.val_x)
            self.val_y = np.array(self.val_y) / 100.0
        
        if test_data is not None:
            self.test_x, self.test_y = test_data
            self.test_x = np.array(self.test_x)
            self.test_y = np.array(self.test_y) / 100.0
        print("Done.")

        self.history = window_size
        self._model = self.create_model(len(self.train_x[0]))
        self._trained = False 

    # set 'seeded' to true if we want consistently reproducible dropout
    # input_shape should be [samples, time steps, features]
    def create_model(self, num_features, num_blocks=30, seeded=False):
        used_seed = None
        if seeded:
            used_seed = self.SEED_DEFAULT
        model = Sequential()
        print(self.num_features)
        model.add(Masking(mask_value=(-1.0), batch_input_shape=(1, self.history, self.num_features)))
        model.add(LSTM(num_blocks, return_sequences=True))
        model.add(Dropout(0.2, seed=used_seed))
        model.add(LSTM(num_blocks, return_sequences=True))
        model.add(Dropout(0.2, seed=used_seed))
        model.add(LSTM(num_blocks, return_sequences=True))
        model.add(Dense(units=3, activation=K.softmax)) # final 3 node dense layer for approval, disapproval, neutral
        return model
    
    def train(self):
        if self._model is not None:
            self._model.compile(loss='mean_squared_error', optimizer='adam')
            for i in range(100):
                self._model.fit(self.train_x, self.train_y, epochs=1, batch_size=1, verbose=2, shuffle=False)
                self._model.reset_states()
            self._trained = True

    def predict(self, x):
        if not self._trained or self._model is None:
            raise ValueError("Model not trained")
        else:
            X = np.array(x)
            X = X.reshape((len(x), self.history, self.num_features))
            predictions = self._model.predict(X, batch_size=1)
            self._model.reset_states()
            return predictions 
    
    def save(self, filename):
        pickle.dump(self._model, open(filename, "wb"))

    def load(self, filename):
        self._model = pickle.load(open(filename, "rb"))

    # saves the graph of the model to a file
    # set dimensions to true if you want the shapes of the outputs on the graph
    def plot_model(self, filename, dimensions=False):
        if self._model is not None:
            plot_model(self._model, to_file=filename, show_shapes=dimensions)
