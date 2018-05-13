import numpy as np
import pickle as pkl

def generate_dataset(density, filename):
	x1, x2 = np.meshgrid(np.linspace(-3, 3, density), np.linspace(-1, 3, density))
	z = np.sin(2 * x1 + x2)
	print("Generated {}x{}  data grid (total train size: {})".format(*x1.shape, x1.size))

	inputs = np.array([[np.reshape(x1, (1, -1))], [np.reshape(x2, (1, -1))]]).squeeze().transpose()
	labels = np.reshape(z, (-1, 1))
	print("Dataset reformatted")

	pkl.dump([inputs, labels], open(filename, "wb"))

	print("Dataset saved to {}".format(filename))


if __name__ == "__main__":
	generate_dataset(density=100, filename="./dataset.pkl")
