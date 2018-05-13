import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from time import time
from datetime import datetime, timedelta
from neural_network import NeuralNetwork

if __name__ == "__main__":
	max_epochs = 200
	inputs, labels = pkl.load(open("./dataset.pkl", "rb"))

	network = NeuralNetwork(
		input_size=2,
		hidden_size=5,
		output_size=1,
		alpha=0.001,
		beta=0.0003
	)

	error_history = []
	err_mean = 1
	current_epoch = 0

	time_begin = time()

	while current_epoch < max_epochs:
		errors = np.zeros((inputs.shape[0], 1))

		for data_row in range(inputs.shape[0]):
			errors[data_row] = network.train(inputs[data_row, :], labels[data_row, :])

		err_mean = np.abs(errors).mean()
		error_history.append(err_mean)
		current_epoch += 1

		if current_epoch % 5 == 0:
			time_str = str(datetime.now().strftime("[%H:%M:%S]"))
			print("{} Epoch: {}, error: {}".format(time_str, current_epoch, err_mean))

	time_end = time()

	duration_str = str(timedelta(seconds=int(time_end - time_begin)))

	print("Final epoch: {}, error: {}, total time: {}".format(current_epoch, err_mean, duration_str))

	figure = plt.figure()
	plt.title("Alpha: {}, Beta: {}, Final avg. error: {}".format(network.alpha, network.beta, round(error_history[-1], 4)))
	plt.plot(error_history)

	timestamp = str(datetime.now().strftime("%Y%d%m_%H%M%S"))
	figure.savefig("error_hist.png")
	network.save("model.pkl")
