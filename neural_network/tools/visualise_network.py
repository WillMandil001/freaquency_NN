import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def show_network_topology(inputs, standards, outputs):
	input_x = []
	input_y = []
	input_z = []
	for neuron in inputs:
		input_x.append(neuron.pose[0])
		input_y.append(neuron.pose[1])
		input_z.append(neuron.pose[2])

	standard_x = []
	standard_y = []
	standard_z = []
	for neuron in standards:
		standard_x.append(neuron.pose[0])
		standard_y.append(neuron.pose[1])
		standard_z.append(neuron.pose[2])

	output_x = []
	output_y = []
	output_z = []
	for neuron in outputs:
		output_x.append(neuron.pose[0])
		output_y.append(neuron.pose[1])
		output_z.append(neuron.pose[2])


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(input_x, input_y, input_z, c="r")
	ax.scatter(standard_x, standard_y, standard_z, c="g")
	ax.scatter(output_x, output_y, output_z, c="b")

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()