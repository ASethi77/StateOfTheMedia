from .regression_model import RegressionModel
from sklearn.neural_network import MLPRegressor

class MLPRegressionModel(RegressionModel):
    def __init__(self, train_data, hidden_layer_sizes=(100,), activation='tanh', alpha=0.0001, learning_rate='adaptive'):
        RegressionModel.__init__(self, train_data)
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, learning_rate=learning_rate)

    def train(self, x=None, y=None):
        x = x if x is not None else self.train_x
        y = y if y is not None else self.train_y
        
        self.model.fit(x, y)
    
    def predict(self, x_in):
        return self.model.predict(x_in)

    def evaluate(self, x_in, y_out):
        return self.model.score(x_in, y_out)