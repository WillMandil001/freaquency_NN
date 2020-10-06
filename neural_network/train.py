import numpy as np
from build_neural_network import neural_network
from single_d_position_test_case import simulation

class spiking_neural_network_RL_trainer():
	def __init__(self, neural_network, simulation_environment):
		self.neural_network = neural_network
		self.simulation = simulation_environment
		self.train()

	def train(self):
		self.event_horrizon(horrizon=100)

	def event_horrizon(self, horrizon):
		events = []
		output = []
		nn_structure = []
		for i in range(0, horrizon):
			event, nn_output, nn = self.run_loop()
			events.append(event)
			output.append(nn_output)
			nn_structure.append(nn)

		print(events[1])
		print(output[1])
		print(nn_structure[1][1][1].fired)

	def run_loop(self):
		events = []
		inputs = self.simulation.calculate_inputs()
		outputs, network = self.neural_network.step(inputs, return_nn_states=True)
		self.simulation.move(outputs)
		return self.simulation.visualise_current_state(), outputs, network

goal_state=5
start_state=22

nn = neural_network(2,10,2)  # init neural network
simulation_environment = simulation(goal_state=goal_state, start_state=start_state, left=0, right=25)  # init simulation
trainer = spiking_neural_network_RL_trainer(nn, simulation_environment)  # init training system

