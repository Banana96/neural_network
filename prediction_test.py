import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from neural_network import NeuralNetwork

if __name__ == "__main__":
	# Load dataset
	inputs, _ = pkl.load(open("./dataset.pkl", "rb"))
	x1_min, x1_max = inputs[:, 0].min(), inputs[:, 0].max()
	x2_min, x2_max = inputs[:, 1].min(), inputs[:, 1].max()

	# Load trained nn model
	nn = NeuralNetwork.load("model.pkl")

	figure = plt.figure()
	axes = Axes3D(figure)

	# Draw original model
	x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
	z = np.sin(2 * x1 + x2)
	axes.plot_wireframe(x1, x2, z, rcount=12, ccount=12)

	# Generate random input data
	pts_count = 200
	rnd_x1 = x1_min + np.random.rand(pts_count) * (x1_max - x1_min)
	rnd_x2 = x2_min + np.random.rand(pts_count) * (x2_max - x2_min)

	# Predict network output, calculate real model value, calculate mean error
	rnd_z = np.array([nn.predict(np.array([rnd_x1[i], rnd_x2[i]])) for i in range(pts_count)])
	real_z = np.array([np.sin(2 * rnd_x1[i] + rnd_x2[i]) for i in range(pts_count)])
	mean_error = (real_z - rnd_z).mean()
	print("Mean prediction error: {}".format(mean_error))

	# Draw predicted values
	axes.scatter(rnd_x1, rnd_x2, rnd_z, c="r")

	plt.show()
