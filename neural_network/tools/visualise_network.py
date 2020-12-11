import numpy as np
from matplotlib import lines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs
	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		FancyArrowPatch.draw(self, renderer)


def show_network_topology(inputs, standards, outputs, image_time=0.5):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
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

	output_x_fired = []
	output_x_not_fired = []
	output_y_fired = []
	output_y_not_fired = []
	output_z_fired = []
	output_z_not_fired = []
	for neuron in outputs:
		if neuron.fired == True:
			output_x_fired.append(neuron.pose[0])
			output_y_fired.append(neuron.pose[1])
			output_z_fired.append(neuron.pose[2])
		else:
			output_x_not_fired.append(neuron.pose[0])
			output_y_not_fired.append(neuron.pose[1])
			output_z_not_fired.append(neuron.pose[2])

	# print(inputs[0].output_ids)
	# print(standards[0].output_ids)
	# print(outputs[0].output_ids)

	for input_neuron in inputs:
		for id_ in input_neuron.output_ids:
			for standard_neuron in standards:
				if id_ == standard_neuron.id:
					c = "k"
					if id_ in input_neuron.fired_to_ids:
						c = "r"
					a = Arrow3D([input_neuron.pose[0], standard_neuron.pose[0]],
								[input_neuron.pose[1], standard_neuron.pose[1]],
								[input_neuron.pose[2], standard_neuron.pose[2]],
								mutation_scale=5, lw=0.5, arrowstyle="-|>", color=c)
					ax.add_artist(a)

	for standard_neuron in standards:
		for id_ in standard_neuron.output_ids:
			for other_standard_neuron in standards:
				if id_[1] == other_standard_neuron.id:
					c = "k"
					if id_[1] in standard_neuron.fired_to_ids:
						c = "r"
					a = Arrow3D([standard_neuron.pose[0], other_standard_neuron.pose[0]],
								[standard_neuron.pose[1], other_standard_neuron.pose[1]],
								[standard_neuron.pose[2], other_standard_neuron.pose[2]],
								mutation_scale=5, lw=0.5, arrowstyle="-|>", color=c)
					ax.add_artist(a)
			for output_neuron in outputs:
				if id_[1] == output_neuron.id:
					c = "k"
					if id_[1] in standard_neuron.fired_to_ids:
						c = "r"
					a = Arrow3D([standard_neuron.pose[0], output_neuron.pose[0]],
								[standard_neuron.pose[1], output_neuron.pose[1]],
								[standard_neuron.pose[2], output_neuron.pose[2]],
								mutation_scale=5, lw=0.5, arrowstyle="-|>", color=c)
					ax.add_artist(a)

	ax.scatter(input_x, input_y, input_z, c="r", s=50)
	ax.scatter(standard_x, standard_y, standard_z, c="g", s=50)
	ax.scatter(output_x_fired, output_y_fired, output_z_fired, c="r",s=50)
	ax.scatter(output_x_not_fired, output_y_not_fired, output_z_not_fired, c="b", s=50)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show(block=False)
	plt.pause(image_time)
	plt.close()

	# plt.show()