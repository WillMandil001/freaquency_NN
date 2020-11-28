import math
import random
import numpy as np

from build_neural_network import neural_network
from tools.visualise_neuron import show_neuron_spiking_plot
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
		# show_neuron_spiking_plot(nn_structure[-1][1][1].current_state_histroy, nn_structure[-1][1][1].fired_history, nn_structure[-1][1][1].recieve_fired_history)
		
		[print(event) for event in events]
		self.calc_successful_pipelines(events, nn_structure)

	def calc_successful_pipelines(self, events, nn_structure):
		fitness = self.calc_fitness(events)
		# last_success = len(fitness) - fitness[::-1].index(1) - 1  # only last time it happened

		# pipeline = self.calc_pipeline(events, last_success)
		# print("THIS IS THE PIPELINE")

	def calc_pipeline(self):
		

		return pipeline

	def calc_fitness(self, events):
		'''
		1. find the last time the output was correct.
		2. for the correct fired output neuron loop through the neurons connected to it.
			3. if one of those neurons fired at previous timestep. increase strength of connection.
				4. loop through that neurons input connections at time t-2 so on and so on.
		'''

		fitness = [0 for i in range(0, len(events))]
		
		good_event = []
		previous_dist = 0
		for i in range(len(events)-2, 1, -1):
			# find distance from o to x at each time step.
			if "8" in events[i] and "o" in events[i+1]:
				print("here")
				good_event.append(i)

			if "o" in events[i]:
				state = events[i].index("o")
				goal = events[i].index("x")
				distance__ = np.sqrt(np.sum((state - goal) ** 2))
				previous_dist = np.sqrt(np.sum((events[i+1].index("o") - events[i+1].index("x")) ** 2))
				if previous_dist > distance__:
					good_event.append(i)

		for i in good_event:
			#### do reinforcment learning
			self.hebbian_learn(event[i])
			print("Found a good sample for training || trigger hebbian")
			print(events[i+1])
			print(events[i])


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

	def hebbian_learn(self):
		pass


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

