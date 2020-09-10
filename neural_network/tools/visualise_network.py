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


def show_network_topology(inputs, standards, outputs):
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

	output_x = []
	output_y = []
	output_z = []
	for neuron in outputs:
		output_x.append(neuron.pose[0])
		output_y.append(neuron.pose[1])
		output_z.append(neuron.pose[2])

	for input_neuron in inputs:
		for id_ in input_neuron.input_ids:
			c = "k"
			if input_neuron.fired == True:
				c = "r"
			for standard_neuron in standards:
				if id_ == standard_neuron.id:
					a = Arrow3D([input_neuron.pose[0], standard_neuron.pose[0]],
								[input_neuron.pose[1], standard_neuron.pose[1]],
								[input_neuron.pose[2], standard_neuron.pose[2]],
								mutation_scale=5, lw=0.5, arrowstyle="-|>", color=c)
					ax.add_artist(a)

	for standard_neuron in standards:
		for id_ in standard_neuron.input_ids:
			if id_ != []:
				c = "k"
				if id_[0] == "output":
					if standard_neuron.fired == True:
						c = "r"
					for output_neuron in outputs:
						if id_[1] == output_neuron.id:
							a = Arrow3D([standard_neuron.pose[0], output_neuron.pose[0]],
										[standard_neuron.pose[1], output_neuron.pose[1]],
										[standard_neuron.pose[2], output_neuron.pose[2]],
										mutation_scale=5, lw=0.5, arrowstyle="-|>", color=c)
							ax.add_artist(a)

				if id_[0] == "standard":
					if standard_neuron.fired == True:
						c = "r"						
					for other_standard_neuron in standards:
						if id_[1] == other_standard_neuron.id:
							a = Arrow3D([standard_neuron.pose[0], other_standard_neuron.pose[0]],
										[standard_neuron.pose[1], other_standard_neuron.pose[1]],
										[standard_neuron.pose[2], other_standard_neuron.pose[2]],
										mutation_scale=5, lw=0.5, arrowstyle="-|>", color=c)
							ax.add_artist(a)

	ax.scatter(input_x, input_y, input_z, c="r")
	ax.scatter(standard_x, standard_y, standard_z, c="g")
	ax.scatter(output_x, output_y, output_z, c="b")

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show(block=False)
	plt.pause(0.5)
	plt.close()

	# plt.show()