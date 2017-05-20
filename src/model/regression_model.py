from sklearn.externals import joblib

class RegressionModel:
	def __init__(self, train_data):
		self.train_x, self.train_y = train_data
		self._model = None

	def train(self):
		raise NotImplementedError

	def predict(self, x_in):
		raise NotImplementedError

	def evaluate(self, x_in, y_out):
		raise NotImplementedError

	def save(self, filename):
		joblib.dump(self.model, filename)

	def load(filename):
		return joblib.load(filename)

