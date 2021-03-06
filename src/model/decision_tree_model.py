from .regression_model import RegressionModel
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor

class DecisionTreeRegressionModel(RegressionModel):
	def __init__(self, train_data, max_depth=2):
		RegressionModel.__init__(self, train_data)
		self.model = DecisionTreeRegressor(max_depth=max_depth)

	def train(self, x=None, y=None):
		x = x if x is not None else self.train_x
		y = y if y is not None else self.train_y

		self.model.fit(x, y)

	def predict(self, x_in):
		return self.model.predict(x_in)

	def evaluate(self, x_in, y_out):
		return self.model.score(x_in, y_out)

	def save(self, filename):
		joblib.dump(self.model, filename)

	def load(self, filename):
		self.model = joblib.load(filename)

