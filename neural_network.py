import numpy as np
import pickle as pkl


def sig(x):
	return np.tanh(x)

def sigd(x):
	return 1 - sig(x)**2

def diag(x):
	flat_x = x.flatten()
	dg = np.eye(x.size)
	for i in range(x.size):
		dg[i, i] = flat_x[i]
	return dg


class NeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size, alpha, beta):
		self.alpha, self.beta = alpha, beta

		self.w1 = np.random.rand(hidden_size+1, input_size+1)
		self.w2 = np.random.rand(output_size, hidden_size+1)

		self.last_delta_w1 = np.zeros_like(self.w1)
		self.last_delta_w2 = np.zeros_like(self.w2)

	def predict(self, x):
		return sig(self.w2 @ sig(self.w1 @ np.append(x, [1])))

	def train(self, x, y_real):
		x = np.append(x, [1])
		hidden_out = sig(self.w1 @ x)
		out = sig(self.w2 @ hidden_out)

		e = y_real - out
		sg = e * sigd(self.w2 @ hidden_out)

		delta_w2 = np.reshape(sg * hidden_out, (1, -1))
		delta_w1 = np.reshape(diag(self.w2 * sg) @ sigd(self.w1 @ x), (-1, 1)) @ np.reshape(x, (1, -1))

		self.w1 += self.alpha * delta_w1 + self.beta * self.last_delta_w1
		self.w2 += self.alpha * delta_w2 + self.beta * self.last_delta_w2

		self.last_delta_w1 = delta_w1
		self.last_delta_w2 = delta_w2

		return e

	def save(self, filename):
		obj_data = {
			"alpha": self.alpha,
			"beta": self.beta,
			"layer1_weights": self.w1,
			"layer2_weights": self.w2,
			"last_delta_w1_update": self.last_delta_w1,
			"last_delta_w2_update": self.last_delta_w2
		}
		pkl.dump(obj_data, open(filename, "wb"))

	@staticmethod
	def load(filename):
		obj_data = pkl.load(open(filename, "rb"))

		a = obj_data["alpha"]
		b = obj_data["beta"]

		nn = NeuralNetwork(1, 1, 1, a, b)

		nn.w1 = obj_data["layer1_weights"]
		nn.w2 = obj_data["layer2_weights"]
		nn.last_delta_w1 = obj_data["last_delta_w1_update"]
		nn.last_delta_w2 = obj_data["last_delta_w2_update"]

		return nn
