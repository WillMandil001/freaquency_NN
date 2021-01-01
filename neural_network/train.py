import math
import random

import numpy as np
import matplotlib.pyplot as plt

#from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from build_neural_network import neural_network
from tools.visualise_neuron import show_neuron_spiking_plot
from tools.visualise_network import show_network_topology
from single_d_position_test_case import simulation


class spiking_neural_network_RL_trainer():
	def __init__(self, neural_network, left, right):
		self.left = left
		self.right = right
		self.test_horrizon = 20
		self.neural_network = neural_network
		graph, = plt.plot([], [], 'o')

		model_fitness = []
		time = []
		line1 = []

		# for i in tqdm(range(0, 50)):
		for i in range(0, 1000):
			self.correct_pipelines_length = []
			model_fitness.append(self.test_current_network_fitness(num_of_tests=4))

			time.append(i)
			line1 = self.live_plotter(time, model_fitness, line1)

			if self.correct_pipelines_length:
				print("epoch:", i, "fitness: ", model_fitness[-1], "number pipelines: ", len(self.correct_pipelines_length), " mean pipeline length, ", sum(self.correct_pipelines_length) / len(self.correct_pipelines_length))
			else:
				print("epoch:", i, "fitness: ", model_fitness[-1])
			correct_steps = self.train()
			print("correct steps: ", len(correct_steps))

		model_fitness.append(self.test_current_network_fitness(num_of_tests=4, print_=True))

		print(model_fitness)

		epoch = [i for i in range(0, len(model_fitness))]
		plt.plot(epoch, model_fitness, 'o')
		plt.plot(np.unique(epoch), np.poly1d(np.polyfit(epoch, model_fitness, 1))(np.unique(epoch)))
		plt.show()

	def test_current_network_fitness(self, num_of_tests, print_=False):
		fitness_tests = []
		for j in range(0, num_of_tests):
			start_state = random.randint(self.left, self.right)  # random start state.
			goal_state = random.randint(self.left, self.right)  # random goal state.
			while start_state == goal_state:
				goal_state = random.randint(self.left,
											self.right)  # change goal state if its the same as the start state.
			self.simulation = simulation(goal_state=goal_state, start_state=start_state, left=self.left,
										 right=self.right)  # init simulation

			events = self.event_horrizon(horrizon=self.test_horrizon)

			correct_events = []
			previous_dist = 0
			for i in range(1, len(events)):
				# find distance from o to x at each time step.
				if "8" in events[i] and "o" in events[i - 1]:
					correct_events.append(i - 1)
				elif "8" in events[i] and "8" in events[i - 1]:
					correct_events.append(i - 1)
				elif "o" in events[i] and "o" in events[i - 1]:
					state = events[i].index("o")
					goal = events[i].index("x")
					distance__ = np.sqrt(np.sum((state - goal) ** 2))
					previous_dist = np.sqrt(np.sum((events[i - 1].index("o") - events[i - 1].index("x")) ** 2))
					if previous_dist > distance__:
						correct_events.append(i - 1)
			fitness_tests.append(len(correct_events))
		average_fitness = sum(fitness_tests) / len(fitness_tests)

		if print_:
			for i, event in enumerate(events):
				print(event, i) 

		return average_fitness

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
		self.simulation = simulation(goal_state=goal_state, start_state=start_state, left=self.left, right=self.right)  # init simulation

		events = self.event_horrizon(horrizon=50)
		# [print(event, i) for i, event in enumerate(events)]

		# self.calc_successful_pipelines(events, nn_structure)
		correct_events = self.train_neural_network(events)

		self.log_history(False)

		return correct_events

	def train_neural_network(self, events):
		correct_events = self.find_correct_steps(events)

		if correct_events != None:
			# print("found good events to train")
			# print(correct_events)
			self.find_correct_pipelines(correct_events, events)

		return correct_events

	# self.hebbian_learn(correct_events, events, nn_structure)

	def find_correct_pipelines(self, correct_events, events):
		'''
		Finds the correct sequence of neurons firing that caused the correct ouput.
		Hyper parameters are:
			1. Time difference between n-1 firing and n firing.
		pipeline is defined as a list of [prev_neuron_id, next_neuron_id] -- basically the connection that needs strengthening
		'''
		pipeline_st = []
		pipeline_st_holder = []
		pipeline_in = []
		pipeline_in_holder = []
		for time_step in correct_events:
			self.correct_pipelines_length.append(0)
			# print("======================, ", time_step)
			for t in range(time_step + 1, -1, -1):
				if t == time_step + 1:
					### output neuron stage:
					for out_neuron in self.nn_structure[t][2]:
						if out_neuron.fired == True:
							for st_neuron in self.nn_structure[t - 1][1]:
								if st_neuron.fired == True and ["output",out_neuron.id] in st_neuron.output_ids and self.neural_network.standard_neurons[st_neuron.id].trained == False:
									self.hebbian_learn(1, st_neuron.id, ["output", out_neuron.id], t, strengthen=True)
									pipeline_st_holder.append(st_neuron)
				else:
					if pipeline_st != []:
						for neuron in pipeline_st:
							# FOR STRENGTHENING -- hebbian learn on this neuron and add to pipeline:
							for st_neuron in self.nn_structure[t - 1][1]:
								if st_neuron.fired == True and ["standard",neuron.id] in st_neuron.output_ids and self.neural_network.standard_neurons[st_neuron.id].trained == False:
									self.hebbian_learn(1, st_neuron.id, ["standard", neuron.id], t, strengthen=True)
									pipeline_st_holder.append(st_neuron)
							for in_neuron in self.nn_structure[t - 1][0]:
								if in_neuron.fired == True and neuron.id in in_neuron.output_ids and self.neural_network.standard_neurons[st_neuron.id].trained == False:
									self.hebbian_learn(0, in_neuron.id, neuron.id, t, strengthen=True)
							# FOR WEAKENING -- hebbian learn on this neuron and add to pipeline:
							for st_neuron in self.nn_structure[t + 1][1]:
								if st_neuron.fired == True and ["standard", neuron.id] in st_neuron.output_ids and self.neural_network.standard_neurons[st_neuron.id].trained == False:
									self.hebbian_learn(1, st_neuron.id, ["standard", neuron.id], t, strengthen=False)
									pipeline_st_holder.append(st_neuron)
							for in_neuron in self.nn_structure[t + 1][0]:
								if in_neuron.fired == True and neuron.id in in_neuron.output_ids and self.neural_network.standard_neurons[st_neuron.id].trained == False:
									self.hebbian_learn(0, in_neuron.id, neuron.id, t, strengthen=False)
				pipeline_st = pipeline_st_holder

	def hebbian_learn(self, neuron_type, neuron_id, connection_neuron, t, strengthen):
		'''
		For each succesful neural network output, adjust the parameters of each neuron according to the hebbian principle. 
		Inputs: neuron = the neuron whos connection needs strengthening.
				connection_neuron = the neuron that "neuron" has connection to. 
		For each time step we locat the string of succesful neuron paths.
		for each string:
				adjust neuron parameters:
					1. connection strength.
		'''
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
			pass
			if neuron_type == 0:
			# 	self.neural_network.input_neurons[neuron_id].weaken_connection(connection_neuron)
				self.neural_network.input_neurons[neuron_id].trained = True
			elif neuron_type == 1:
			# 	self.neural_network.standard_neurons[neuron_id].weaken_connection(connection_neuron)
				self.neural_network.standard_neurons[neuron_id].trained = True
			elif neuron_type == 2:
			# 	self.neural_network.output_neurons[neuron_id].weaken_connection(connection_neuron)
				self.neural_network.output_neurons[neuron_id].trained = True

	def find_correct_steps(self, events):
		'''
		This function loops through the events from a sample and finds cases where the output neurons fired correctly.
		It returns these timesteps. 
		'''
		fitness = [0 for i in range(0, len(events))]

		correct_events = []
		previous_dist = 0
		for i in range(1, len(events)):
			# find distance from o to x at each time step.
			if "8" in events[i] and "o" in events[i - 1]:
				correct_events.append(i - 1)
			elif "8" in events[i] and "8" in events[i - 1]:
				correct_events.append(i - 1)
			elif "o" in events[i] and "o" in events[i - 1]:
				state = events[i].index("o")
				goal = events[i].index("x")
				distance__ = np.sqrt(np.sum((state - goal) ** 2))
				previous_dist = np.sqrt(np.sum((events[i - 1].index("o") - events[i - 1].index("x")) ** 2))
				if previous_dist > distance__:
					correct_events.append(i - 1)

		return correct_events

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
		inputs = self.simulation.calculate_inputs()
		outputs, network = self.neural_network.step(inputs, return_nn_states=True)
		self.output_list.append(outputs)
		self.simulation.move(outputs)
		self.nn_structure.append(network)
		return self.simulation.visualise_current_state(), outputs

	def live_plotter(self, x_vec, y1_data, line1, identifier='', pause_time=0.1):
		if line1 == []:
			plt.ion()  # this is the call to matplotlib that allows dynamic plotting
			fig = plt.figure(figsize=(13,6))
			ax = fig.add_subplot(111)
			line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)  # create a variable for the line so we can later update it
			plt.xlabel('Model fitness')  # update plot label/title
			plt.ylabel('Training time_step')
			plt.title('Model training')
			plt.show()

		plt.xlim([np.min(x_vec)-np.std(x_vec),np.max(x_vec)+np.std(x_vec)])
		plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])

		line1.set_data(x_vec,y1_data)
		plt.pause(pause_time)
		return line1

input_size = 10

nn = neural_network(input_size*2, 10, 2, log_history=False)  # init neural network
trainer = spiking_neural_network_RL_trainer(nn, 0, input_size)  # init training system
