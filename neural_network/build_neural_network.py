from tools.visualise_network import show_network_topology
from single_neuron import neuron, input_neuron, output_neuron

class neural_network():
	def __init__(self):
		self.generate_hyper_parameters()
		self.create_simple_network()
		show_network_topology(self.input_neurons, self.standard_neurons, self.output_neurons)

	def create_simple_network(self):
		for i in range(0, 3):
			self.input_neurons.append(input_neuron([0, 0, i]))
		for i in range(0, 3):
			self.output_neurons.append(neuron([0, 6, i]))
		for i in range(0, 3):
			self.standard_neurons.append(output_neuron([0, 3, i]))

	def step(self):
		pass

	def generate_hyper_parameters(self):
		self.neuron_range = 10
		self.input_neurons = []
		self.standard_neurons = []
		self.output_neurons = []

nn = neural_network()