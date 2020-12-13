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

		events = self.event_horrizon(horrizon=100)
		[print(event, i) for i, event in enumerate(events)]

		# self.calc_successful_pipelines(events, nn_structure)
		self.train_neural_network(events)

	def train_neural_network(self, events):
		correct_events = self.find_correct_steps(events)

		if correct_events != None:
			print("found good events to train")
			print(correct_events)
			self.find_correct_pipelines(correct_events, events)

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
			print("======================, ", time_step)
			for t in range(time_step+1, -1, -1):
				if t == time_step+1:
					### output neuron stage:
					for out_neuron in self.nn_structure[t][2]:
						if out_neuron.fired == True:
							for st_neuron in self.nn_structure[t-1][1]:
								if st_neuron.fired == True and ["output", out_neuron.id] in st_neuron.output_ids:
									print("11111")
									self.hebbian_learn(st_neuron, ["output", out_neuron.id], strengthen=True)
									pipeline_st_holder.append(st_neuron.id)
				else:
					if pipeline_st != []:
						for nn_id in pipeline_st:
							for neuron in self.nn_structure[t][1]:  # st neuron
								if neuron.id == nn_id:
									### FOR STRENGTHENING
									### hebbian learn on this neuron and add to pipeline:
									for st_neuron in self.nn_structure[t-1][1]:
										if st_neuron.fired == True and ["standard", neuron.id] in st_neuron.output_ids:
											print("121212121")
											self.hebbian_learn(st_neuron, ["standard", neuron.id], strengthen=True)
											pipeline_st_holder.append(st_neuron.id)
									for in_neuron in self.nn_structure[t-1][0]:
										if in_neuron.fired == True and neuron.id in in_neuron.output_ids:
											print("22222222")
											self.hebbian_learn(in_neuron, ["standard", neuron.id], strengthen=True)
									### FOR WEAKENING
									### hebbian learn on this neuron and add to pipeline:
									for st_neuron in self.nn_structure[t+1][1]:
										if st_neuron.fired == True and ["standard", neuron.id] in st_neuron.output_ids:
											print("33333333")
											self.hebbian_learn(st_neuron, ["standard", neuron.id], strengthen=False)
											pipeline_st_holder.append(st_neuron.id)
									for in_neuron in self.nn_structure[t+1][0]:
										if in_neuron.fired == True and neuron.id in in_neuron.output_ids:
											print("44444444")
											self.hebbian_learn(in_neuron, ["standard", neuron.id], strengthen=False)

				pipeline_st = pipeline_st_holder
	#
	# for neuron in self.nn_structure[t][1]:
	# 	for id__ in neuron.output_ids:
	# 		if id__[1] == neuron.id:
	# 			print(neuron.id, neuron.output_ids)

	def hebbian_learn(self, neuron, connection_neuron, strengthen):
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
			neuron.strengthen_conenction(connection_neuron)
			# print("strengthened input connection")
		else:
			neuron.weaken_connection(connection_neuron)
			print("weakend input connection")


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

nn = neural_network(25,50,2, log_history=True)  # init neural network
trainer = spiking_neural_network_RL_trainer(nn, 0, 25)  # init training system
