import math
import random
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from build_neural_network import neural_network
from gates.passthrough_test import passthrough_test
from tools.visualise_neuron import show_neuron_spiking_plot
from tools.visualise_network import show_network_topology


class spiking_neural_network_RL_trainer():
	def __init__(self, neural_network, left, right):
		self.left = left
		self.right = right
		self.neural_network = neural_network
		# model_fitness = []
		graph, = plt.plot([], [], 'o')

		for i in range(0, 1000):
			print("epoch:", i, " Fitness = ", sum(self.fitnesses))
			self.train()

		epoch = [i for i in range(0, len(model_fitness))]
		plt.plot(epoch, model_fitness, 'o')
		plt.plot(np.unique(epoch), np.poly1d(np.polyfit(epoch, model_fitness, 1))(np.unique(epoch)))
		plt.show()

	def log_history(self, yesno):
		if yesno:
			for neuron in self.neural_network.input_neurons:
				neuron.log_history = False
			for neuron in self.neural_network.standard_neurons:
				neuron.log_history = False
			for neuron in self.neural_network.output_neurons:
				neuron.log_history = False
		if not yesno:
			for neuron in self.neural_network.input_neurons:
				neuron.log_history = False
				neuron.current_state_histroy = []
				neuron.fired_history = []
				neuron.recieve_fired_history = []
			for neuron in self.neural_network.standard_neurons:
				neuron.log_history = False
				neuron.current_state_histroy = []
				neuron.fired_history = []
				neuron.recieve_fired_history = []
			for neuron in self.neural_network.output_neurons:
				neuron.log_history = False
				neuron.current_state_histroy = []
				neuron.fired_history = []
				neuron.recieve_fired_history = []

	def train(self):
		self.log_history(True)
		self.passthrough_test = passthrough_test()  # init simulation
		self.event_horrizon(horrizon=100)  # run the simulation for 100 timesteps and return fitness of model at each time_step
		self.train_neural_network(self.fitnesses)
		self.log_history(False)

	def train_neural_network(self, fitnesses):
		if fitnesses != None:
			pipelines = self.passthrough_test.find_correct_pipelines(fitnesses, self.nn_structure)

		for neuron_to_train in pipelines:
			self.hebbian_learn(neuron_to_train)

	def hebbian_learn(self, neuron_to_train):
		'''
		For each succesful neural network output, adjust the parameters of each neuron according to the hebbian principle. 
		Inputs: neuron = the neuron whos connection needs strengthening.
				connection_neuron = the neuron that "neuron" has connection to. 
		For each time step we locat the string of succesful neuron paths.
		for each string:
				adjust neuron parameters:
					1. connection strength.
		'''
		neuron_type = neuron_to_train[0]
		neuron_id = neuron_to_train[1]
		connection_neuron = neuron_to_train[2]
		t = neuron_to_train[3]
		strengthen = neuron_to_train[4]

		if strengthen:
			self.correct_pipelines_length[-1] += 1
			if neuron_type == 0:
				self.neural_network.input_neurons[neuron_id].strengthen_conenction(connection_neuron)
				self.neural_network.input_neurons[neuron_id].trained = True
			elif neuron_type == 1:
				self.neural_network.standard_neurons[neuron_id].strengthen_conenction(connection_neuron)
				self.neural_network.standard_neurons[neuron_id].trained = True
			elif neuron_type == 2:
				self.neural_network.output_neurons[neuron_id].strengthen_conenction(connection_neuron)
				self.neural_network.output_neurons[neuron_id].trained = True
		else:
			if neuron_type == 0:
				self.neural_network.input_neurons[neuron_id].weaken_connection(connection_neuron)
				self.neural_network.input_neurons[neuron_id].trained = True
			elif neuron_type == 1:
				self.neural_network.standard_neurons[neuron_id].weaken_connection(connection_neuron)
				self.neural_network.standard_neurons[neuron_id].trained = True
			elif neuron_type == 2:
				self.neural_network.output_neurons[neuron_id].weaken_connection(connection_neuron)
				self.neural_network.output_neurons[neuron_id].trained = True
def event_horrizon(self, horrizon):
		self.fitnesses = []
		self.nn_structure = []
		self.output_list = []
		for i in range(0, horrizon):
			fitness = self.run_loop()
			self.fitnesses.append(fitness)

	def run_loop(self):
		events = []
		inputs = self.passthrough_test.calculate_inputs()
		outputs, network = self.neural_network.step(inputs, return_nn_states=True)
		fitness = self.passthrough_test.fitness(inputs, outputs)
		self.nn_structure.append(network)
		return fitness


input_size = 1
nn = neural_network(input_size, 1, 1, log_history=False)  # init neural network
trainer = spiking_neural_network_RL_trainer(nn, 0, input_size)  # init training system
