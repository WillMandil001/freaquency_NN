import math
import random
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from build_neural_network import neural_network
from passthrough_test.py import passthrough_test
from tools.visualise_neuron import show_neuron_spiking_plot
from tools.visualise_network import show_network_topology


class spiking_neural_network_RL_trainer():
	def __init__(self, neural_network, left, right):
		self.left = left
		self.right = right
		self.neural_network = neural_network
		model_fitness = []
		graph, = plt.plot([], [], 'o')

		# for i in tqdm(range(0, 50)):
		for i in range(0, 1000):
			self.correct_pipelines_length = []
			model_fitness.append(self.test_current_network_fitness(num_of_tests=10))
			if self.correct_pipelines_length:
				print("epoch:", i, "fitness: ", model_fitness[-1], "number pipelines: ", len(self.correct_pipelines_length), " mean pipeline length, ", sum(self.correct_pipelines_length) / len(self.correct_pipelines_length))
			else:
				print("epoch:", i, "fitness: ", model_fitness[-1])
			self.train()

		model_fitness.append(self.test_current_network_fitness(num_of_tests=10))

		print(model_fitness)

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
		start_state = random.randint(self.left, self.right)  # random start state.
		goal_state = random.randint(self.left, self.right)  # random goal state.
		while start_state == goal_state:
			goal_state = random.randint(self.left, self.right)  # change goal state if its the same as the start state.

		self.log_history(True)
		self.passthrough_test = passthrough_test()  # init simulation

		events = self.event_horrizon(horrizon=20)
		# [print(event, i) for i, event in enumerate(events)]

		# self.calc_successful_pipelines(events, nn_structure)
		self.train_neural_network(events)

		self.log_history(False)

	def train_neural_network(self, events):
		correct_events = self.find_correct_steps(events)
		if correct_events != None:
			self.find_correct_pipelines(correct_events, events)


	def hebbian_learn(self, neuron, connection_neuron, t, strengthen):
		'''
		For each succesful neural network output, adjust the parameters of each neuron according to the hebbian principle. 
		Inputs: neuron = the neuron whos connection needs strengthening.
				connection_neuron = the neuron that "neuron" has connection to. 
		For each time step we locat the string of succesful neuron paths.
		for each string:
				adjust neuron parameters:
					1. connection strength.
		'''
		if strengthen == True:
			self.correct_pipelines_length[-1] += 1
			neuron.strengthen_conenction(connection_neuron)
			neuron.trained = True
		# print("strengthened input connection", neuron.id, t)
		else:
			neuron.weaken_connection(connection_neuron)
			neuron.trained = True
		# print("weakend input connection", neuron.id, t)

	def event_horrizon(self, horrizon):
		events = []
		output = []
		self.nn_structure = []
		self.output_list = []
		for i in range(0, horrizon):
			event, nn_output = self.run_loop()
			events.append(event)
			output.append(nn_output)
		return events

	def run_loop(self):
		events = []
		inputs = self.passthrough_test.calculate_inputs()
		outputs, network = self.neural_network.step(inputs, return_nn_states=True)
		self.output_list.append(outputs)
		self.passthrough_test.move(outputs)
		self.nn_structure.append(network)
		return self.passthrough_test.visualise_current_state(), outputs


input_size = 1

nn = neural_network(2*input_size, 1, 1, log_history=False)  # init neural network
trainer = spiking_neural_network_RL_trainer(nn, 0, input_size)  # init training system
