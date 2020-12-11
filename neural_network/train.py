import math
import random
import numpy as np

from build_neural_network import neural_network
from tools.visualise_neuron import show_neuron_spiking_plot
from tools.visualise_network import show_network_topology
from single_d_position_test_case import simulation

class spiking_neural_network_RL_trainer():
	def __init__(self, neural_network, left, right):
		self.left = left
		self.right = right
		self.neural_network = neural_network
		self.train()

	def train(self):
		start_state = random.randint(self.left, self.right)  # random start state.
		goal_state = random.randint(self.left, self.right)  # random goal state.
		while start_state == goal_state:
			goal_state = random.randint(self.left, self.right)  # change goal state if its the same as the start state.

		self.simulation = simulation(goal_state=goal_state, start_state=start_state, left=self.left, right=self.right)  # init simulation

		events, nn_structure = self.event_horrizon(horrizon=100)
		[print(event) for event in events]
		# show_neuron_spiking_plot(nn_structure[-1][2][1].current_state_histroy, nn_structure[-1][2][1].fired_history, nn_structure[-1][2][1].recieve_fired_history)
		
		# self.calc_successful_pipelines(events, nn_structure)
		self.train_neural_network(events, nn_structure)

	def train_neural_network(self, events, nn_structure):		
		correct_events = self.find_correct_steps(events)

		print(len(nn_structure))

		# for neurons in nn_structure:
		# 	print(neurons[2][0].fired)
		# 	print(neurons[2][1].fired)

		# for i, event in enumerate(events):
		# 	print("---------------------------------------")
		# 	# print(len(nn_structure[i][2]))
		# 	print(nn_structure[i][2][0].fired)
		# 	print(nn_structure[i][2][1].fired)
			# print(event)

		# for event in correct_events:
		# 	print(events[event-1])
		# 	print(nn_structure[event][2][0].fired)
		# 	print(nn_structure[event][2][1].fired)
		# 	print(events[event])

		if correct_events != None:
			print("found good events to train")
			print(correct_events)
			# pipelines = self.find_correct_pipelines(correct_events, events, nn_structure)

		# self.hebbian_learn(correct_events, events, nn_structure)

	def find_correct_pipelines(self, correct_events, events, nn_structure):
		'''
		Finds the correct sequence of neurons firing that caused the correct ouput.
		Hyper parameters are:
			1. Time difference between n-1 firing and n firing. 
		'''
		# show_neuron_spiking_plot(nn_structure[correct_events[0]][1][1].current_state_histroy, nn_structure[correct_events[0]][1][1].fired_history, nn_structure[correct_events[0]][1][1].recieve_fired_history)
		# show_network_topology(nn_structure[correct_events[0]][0], nn_structure[correct_events[0]][1], nn_structure[correct_events[0]][2], image_time=1000)

		print(len(correct_events))
		for time_step in correct_events:
			for neuron in nn_structure[time_step-1][2]:
				print(neuron.fired)
				# if neuron.fired == True:
				# 	print(neuron.fired)

		return False

	def hebbian_learn(self, correct_events, events, nn_structure):
		'''
		For each succesful neural network output, adjust the parameters of each neuron according to the hebbian principle. 
		Inputs: correct events = list of the timesteps in the last sample that had correct outputs.
				nn_structure = the neural network at each time step.
		For each time step we locat the string of succesful neuron paths.
		for each string:
				adjust neuron parameters:
					1. connection strength.
					
		'''
		print(nn_structure[-1][1][1].output_transmition_values)
		print(nn_structure[-1][1][1].output_ids)


		for event in correct_events:
			pass

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
			if "8" in events[i] and "o" in events[i-1]:
				correct_events.append(i-1)

			elif "o" in events[i] and "o" in events[i-1]:
				state = events[i].index("o")
				goal = events[i].index("x")
				distance__ = np.sqrt(np.sum((state - goal) ** 2))
				previous_dist = np.sqrt(np.sum((events[i-1].index("o") - events[i-1].index("x")) ** 2))
				if previous_dist > distance__:
					correct_events.append(i-1)

		return correct_events

	def calc_fitness(self, events):
		fitness = [0 for i in range(0, len(events))]
		# for i in range(1, len(events)):
		# 	if "8" in events[i]:
		# 		current_state = events[i].index("8")  # calc dist to goal
		# 		goal_state = events[i].index("8")
		# 	else:
		# 		current_state = events[i].index("o")  # calc dist to goal
		# 		goal_state = events[i].index("x")
			
		# 	if current_state < goal_state:
		# 		current_dist = math.sqrt((current_state - goal_state)**2)
		# 	else:
		# 		current_dist = math.sqrt((goal_state - current_state)**2)

		# 	if "8" in events[i-1]:
		# 		previous_current_state = events[i-1].index("8")  # calc dist to goal
		# 		previous_goal_state = events[i-1].index("8")
		# 	self.right = right
		# 	else:
		# 		previous_current_state = events[i-1].index("o")  # calc previous dist to goal
		# 		previous_goal_state = events[i-1].index("x")
		# 	if previous_current_state < previous_goal_state:
		# 		previous_dist = math.sqrt((previous_current_state - previous_goal_state)**2)
		# 	else:
		# 		previous_dist = math.sqrt((previous_goal_state - previous_current_state)**2)

		# 	if current_dist < previous_dist:
		# 		fitness[i] = 1
		return fitness

	def event_horrizon(self, horrizon):
		events = []
		output = []
		nn_structure = []
		for i in range(0, horrizon):
			event, nn_output, nn = self.run_loop()
			events.append(event)
			output.append(nn_output)
			nn_structure.append(nn)
		return events, nn_structure

	def run_loop(self):
		events = []
		inputs = self.simulation.calculate_inputs()
		outputs, network = self.neural_network.step(inputs, return_nn_states=True)
		self.simulation.move(outputs)
		return self.simulation.visualise_current_state(), outputs, network

nn = neural_network(25,50,2, log_history=True)  # init neural network
trainer = spiking_neural_network_RL_trainer(nn, 0, 25)  # init training system

