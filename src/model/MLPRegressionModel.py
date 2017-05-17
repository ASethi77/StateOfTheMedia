from .regression_model import RegressionModel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class MLPRegressionModel(RegressionModel):
    def __init__(self, train_data):
        RegressionModel.__init__(self, train_data)
        self.model = MLPRegressor()

    def train(self, x=None, y=None):
        x = x if x is not None else self.train_x
        y = y if y is not None else self.train_y
        
        self.model.fit(x, y)
    
    def predict(self, x_in):
        return self.model.predict(x_in)

    def evaluate(self, x_in, y_out):
        predicted = self.predict(x_in)
        return mean_squared_error(y_out, predicted)
