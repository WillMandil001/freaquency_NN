import math
import copy
import random

from tools.visualise_network import show_network_topology
from neurons.probabilistic_neuron import neuron, input_neuron, output_neuron

class neural_network():
	def __init__(self, in_ ,st_ ,out_, log_history=False):
		self.generate_hyper_parameters()
		self.generate_network(in_ ,st_ ,out_, log_history)
		# self.create_simple_network()
		self.generate_initial_connections()
		# show_network_topology(self.input_neurons, self.standard_neurons, self.output_neurons)

	def generate_hyper_parameters(self):
		self.neuron_range = 3.5
		self.input_neurons = []
		self.standard_neurons = []
		self.output_neurons = []
		self.input_ids = []
		self.standard_ids = []
		self.output_ids = []

	def generate_initial_connections(self):
		for i in range(0, len(self.input_neurons)):
			self.generate_input_connection(i)
		for i in range(0, len(self.standard_neurons)):
			self.generate_standard_connection(i)

	def generate_input_connection(self, input_neuron_index):
		list_of_neurons_in_range = []
		pose = self.input_neurons[input_neuron_index].pose
		distance_list_st = []
		connection_made = False
		for st_n in self.standard_neurons:
			distance = math.sqrt((st_n.pose[0] - pose[0])**2 + (st_n.pose[1] - pose[1])**2 + (st_n.pose[2] - pose[2])**2)
			distance_list_st.append(["s", distance])
			if distance < self.neuron_range:
				if random.uniform(0,1) > 0.85:
					self.input_neurons[input_neuron_index].output_ids.append(st_n.id)
					connection_made = True

		if connection_made == False:  # if no connection made
			min_st = min(range(len(distance_list_st)), key=distance_list_st.__getitem__)
			self.input_neurons[input_neuron_index].output_ids.append(self.standard_neurons[min_st].id)	

		for id___ in self.input_neurons[input_neuron_index].output_ids:
			self.input_neurons[input_neuron_index].output_transmition_values.append(self.input_neurons[input_neuron_index].starting_transmition_value)

	def generate_standard_connection(self, standard_neuron_index):
		list_of_neurons_in_range = []
		pose = self.standard_neurons[standard_neuron_index].pose
		distance_list_st = []
		distance_list_out = []
		connection_made = False
		# for other standard neurons:
		for st_n in self.standard_neurons:
			distance = math.sqrt((st_n.pose[0] - pose[0])**2 + (st_n.pose[1] - pose[1])**2 + (st_n.pose[2] - pose[2])**2)
			distance_list_st.append(["s", distance])
			if distance < self.neuron_range:
				if random.uniform(0,1) > 0.85:
					self.standard_neurons[standard_neuron_index].output_ids.append(["standard", st_n.id])
					connection_made = True
		for out_n in self.output_neurons:
			distance = math.sqrt((out_n.pose[0] - pose[0])**2 + (out_n.pose[1] - pose[1])**2 + (out_n.pose[2] - pose[2])**2)
			distance_list_out.append(["o", distance])
			if distance < self.neuron_range:
				if random.uniform(0,1) > 0.1:
					self.standard_neurons[standard_neuron_index].output_ids.append(["output", out_n.id])
					connection_made = True

		if connection_made == False:
			min_out = min(range(len(distance_list_out)), key=distance_list_out.__getitem__)
			min_st = min(range(len(distance_list_st)), key=distance_list_st.__getitem__)
			if  min_out >= min_st:
				self.standard_neurons[standard_neuron_index].output_ids.append(["standard", self.standard_neurons[min_st].id])	
			else:
				self.standard_neurons[standard_neuron_index].output_ids.append(["output", self.output_neurons[min_out].id])	

		### generate individual conection strengths:
		for id___ in self.standard_neurons[standard_neuron_index].output_ids:
			self.standard_neurons[standard_neuron_index].output_transmition_values.append(self.standard_neurons[standard_neuron_index].starting_transmition_value)		


	def generate_output_connection(self, input_neuron_index):
		list_of_neurons_in_range = []
		pose = self.input_neurons[input_neuron_index].pose
		distance_list = []
		connection_made = False
		for st_n in self.standard_neurons:
			distance = math.sqrt((st_n.pose[0] - pose[0])**2 + (st_n.pose[1] - pose[1])**2 + (st_n.pose[2] - pose[2])**2)
			distance_list.append(distance)
			if distance < self.neuron_range:
				if random.uniform(0,1) > 0.7:
					st_n.output_ids.append(self.input_neurons[input_neuron_index].id)
					connection_made = True
		if connection_made == False:
			self.input_neurons[input_neuron_index].output_ids.append(self.standard_neurons[min(range(len(distance_list)), key=distance_list.__getitem__)].id)

	def generate_network(self, no_inputs, no_standards, no_outputs, log_history):
		self.build_input_shape(no_inputs)
		length_of_pipes = self.build_standard_shape(no_standards, log_history)
		self.build_output_shape(no_outputs, length_of_pipes, log_history)

	def build_output_shape(self, number_of_points, length_of_pipe, log_history):
		radius_of_output_pipe = int(number_of_points / 1.5)
		min_start = 1
		list_x = []
		list_y = []
		for i in range(0, number_of_points):
			id_ = i
			self.output_ids.append(id_)
			while 1:
				r_squared, theta = [random.randint(0,radius_of_output_pipe), 2*math.pi*random.random()]
				x = math.sqrt(r_squared)*math.cos(theta)
				y = math.sqrt(r_squared)*math.sin(theta)
				if x not in list_x or y not in list_y:
					list_x.append(x)
					list_y.append(y)
					self.output_neurons.append(output_neuron([x, (length_of_pipe + min_start), y], id_, log_history))
					break

	def build_standard_shape(self, number_of_points, log_history):
		min_start = 1
		radius_of_standard_pipe = int(number_of_points / 4)
		if radius_of_standard_pipe == 0:
			radius_of_standard_pipe = 1
		length_of_pipe = int(number_of_points / 3)
		if length_of_pipe == 0:
			length_of_pipe = 1
		list_x = []
		list_y = []
		list_z = []
		for i in range(0, number_of_points):
			id_ = i
			self.standard_ids.append(id_)
			while 1:
				y = random.randint(min_start,length_of_pipe)
				r_squared, theta = [random.randint(0,radius_of_standard_pipe), 2*math.pi*random.random()]
				x = math.sqrt(r_squared)*math.cos(theta)
				z = math.sqrt(r_squared)*math.sin(theta)
				if x not in list_x or y not in list_y or z not in list_z:
					list_x.append(x)
					list_y.append(y)
					list_z.append(z)
					self.standard_neurons.append(neuron([x, y, z], id_, log_history))
					break
		return length_of_pipe + min_start

	def build_input_shape(self, number_of_points):
		radius_of_input_pipe = int(number_of_points / 1.5)
		list_x = []
		list_z = []
		for i in range(0, number_of_points):
			id_ = i
			self.input_ids.append(id_)
			while 1:
				r_squared, theta = [random.randint(0,radius_of_input_pipe), 2*math.pi*random.random()]
				x = math.sqrt(r_squared)*math.cos(theta)
				z = math.sqrt(r_squared)*math.sin(theta)
				if x not in list_x or z not in list_z:
					list_x.append(x)
					list_z.append(z)
					self.input_neurons.append(input_neuron([x, 0, z], id_))
					break

	def step(self, input_frequencies, return_nn_states=False):
		# 1. Check if each neuron needs to fire based on it's resting frequency
		for index, input_neuron in enumerate(self.input_neurons):
			input_neuron.update(input_frequencies[index], self.standard_neurons)

		for standard_neuron in self.standard_neurons:
			standard_neuron.update(self.standard_neurons, self.output_neurons)

		fired = [False for i in self.output_neurons]
		for index, output_neuron in enumerate(self.output_neurons):
			fired[index] = output_neuron.update()

		# show_network_topology(self.input_neurons, self.standard_neurons, self.output_neurons)
		if return_nn_states == True:
			return fired, [copy.deepcopy(self.input_neurons), copy.deepcopy(self.standard_neurons), copy.deepcopy(self.output_neurons)]
		else:
			return fired

# nn = neural_network()
# for i in range(0, 100):
# 	nn.step([0.5, 0.5, 0.5, 0.5, 0.5])