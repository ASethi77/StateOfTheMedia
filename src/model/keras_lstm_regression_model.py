import keras
import tensorflow as tf
import numpy as np
import pandas
import pickle

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

class LSTMRegressionModel:
    BATCH_SIZE = 10 # how many samples to use per forward/backward feed (more takes more memory, but has better optima)
    EPOCHS = 5000 # of times to optimize over the entire training set
    HISTORY_DEFAULT = 30 # how many days of features should we include in each feature vector
                         # NOTE: THIS NEEDS TO MATCH Config.DAY_RANGE, but cannot enforce this due to 
                         #       python circular dependencies
    SEED_DEFAULT = 7 # seed value to reproduce random elements

    def __init__(self, train_data, test_data, window_size = 30):
        print("Initializing data...")
        init_x, init_y = train_data
        test_x, test_y = test_data
        if len(init_x) == 0:
            raise ValueError("No Training Samples Passed")
        if len(init_y) == 0:
            raise ValueError("No Training Labels Passed")
        self.train_x = np.array(init_x)
        self.train_y = np.array(init_y) / 100.0 # scale to range [0.0, 1.0]
        
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y) / 100.0
        self.num_features = self.train_x.shape[2]
        # re-shape the numpy arrays
        # (# batch, # samples, # features)
        print(self.train_x.shape)
        self.train_x = self.train_x.reshape((self.train_x.shape[0], window_size, self.num_features))
        self.train_y = self.train_y.reshape((self.train_y.shape[0], 3))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], window_size, self.num_features))
        self.test_y = self.test_y.reshape((self.test_y.shape[0], 3))
       
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
        model.add(Masking(mask_value=(-1.0), batch_input_shape=(LSTMRegressionModel.BATCH_SIZE, self.history, self.num_features)))
        model.add(LSTM(num_blocks, return_sequences=True))
        model.add(Dropout(0.2, seed=used_seed))
        model.add(LSTM(num_blocks, return_sequences=True))
        model.add(Dropout(0.2, seed=used_seed))
        model.add(LSTM(num_blocks, return_sequences=False))
        model.add(Dense(units=3, activation=K.softmax)) # final 3 node dense layer for approval, disapproval, neutral
        plot_model(model, to_file="lstm.png", show_shapes=True)
        return model
    
    def train(self):
        if self._model is not None:
            self._model.compile(loss='mean_squared_error', optimizer='adam')
            for i in range(LSTMRegressionModel.EPOCHS):
                print("STARTING EPOCH #" + str(i))
                batch_num = 0
                while LSTMRegressionModel.BATCH_SIZE * batch_num < self.train_x.shape[0]:
                    #print("Training batch #" + str(batch_num))
                    start_indx = LSTMRegressionModel.BATCH_SIZE * batch_num
                    end_indx = start_indx + LSTMRegressionModel.BATCH_SIZE
                    remainder = 0
                    if end_indx > self.train_x.shape[0]:
                        remainder = end_indx - self.train_x.shape[0]
                        end_indx = self.train_x.shape[0]
                    batch_x = self.train_x[start_indx: end_indx]
                    batch_y = self.train_y[start_indx: end_indx]
                    if remainder != 0:
                        batch_x = np.append(batch_x, self.train_x[:remainder], axis=0)
                        batch_y = np.append(batch_y, self.train_y[:remainder], axis=0)
                    self._model.train_on_batch(batch_x, batch_y)
                    batch_num += 1
                print("DONE TRAINING EPOCH #" + str(i) + "...")
                print("EVALUATING ON EPOCH #" + str(i) + "...")
                batch_num = 0
                total_loss = 0.0
                while LSTMRegressionModel.BATCH_SIZE * batch_num < self.train_x.shape[0]:
                    #print("Training batch #" + str(batch_num))
                    start_indx = LSTMRegressionModel.BATCH_SIZE * batch_num
                    end_indx = start_indx + LSTMRegressionModel.BATCH_SIZE
                    remainder = 0
                    if end_indx > self.train_x.shape[0]:
                        remainder = end_indx - self.train_x.shape[0]
                        end_indx = self.train_x.shape[0]
                    batch_x = self.train_x[start_indx: end_indx]
                    batch_y = self.train_y[start_indx: end_indx]
                    if remainder != 0:
                        batch_x = np.append(batch_x, self.train_x[:remainder], axis=0)
                        batch_y = np.append(batch_y, self.train_y[:remainder], axis=0)
                    total_loss += self._model.test_on_batch(batch_x, batch_y)
                    batch_num += 1
                print("TRAINING LOSS FOR EPOCH #" + str(i) + " = " + str(total_loss/batch_num))
                print("EVALUATING ON EPOCH #" + str(i) + "...")
                batch_num = 0
                total_loss = 0.0
                while LSTMRegressionModel.BATCH_SIZE * batch_num < self.test_x.shape[0]:
                    #print("Training batch #" + str(batch_num))
                    start_indx = LSTMRegressionModel.BATCH_SIZE * batch_num
                    end_indx = start_indx + LSTMRegressionModel.BATCH_SIZE
                    remainder = 0
                    if end_indx > self.test_x.shape[0]:
                        remainder = end_indx - self.test_x.shape[0]
                        end_indx = self.test_x.shape[0]
                    batch_x = self.test_x[start_indx: end_indx]
                    batch_y = self.test_y[start_indx: end_indx]
                    if remainder != 0:
                        batch_x = np.append(batch_x, self.test_x[:remainder], axis=0)
                        batch_y = np.append(batch_y, self.test_y[:remainder], axis=0)
                    total_loss += self._model.test_on_batch(batch_x, batch_y)
                    batch_num += 1
                print("TEST LOSS FOR EPOCH #" + str(i) + " = " + str(total_loss/batch_num)) 
                if i % 5 == 0:
                    self._model.save("lstm_models/LSTM_EPOCH_" + str(i)) 
            #self._model.fit(self.train_x, self.train_y, epochs=10, batch_size=1, verbose=2, shuffle=False)
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
    def plot_model(self, filename, dimensions=True):
        if self._model is not None:
            plot_model(self._model, to_file=filename, show_shapes=dimensions)
