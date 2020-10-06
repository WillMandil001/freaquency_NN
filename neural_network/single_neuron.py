import random

####################################################################
class neuron():
	def __init__(self, pose, id_):
		# self.input_weights = []
		# self.input_ids = []
		self.output_ids = []
		self.resting_frequency = 0.2  # 0.2
		self.emition_threshold = 10
		self.decay_rate = 0.025
		self.transmition_value = 0.1
		self.pose = pose
		self.id = id_
		self.current_state = 1.0
		self.fired_to_ids = []
		self.fired = False

	def update(self, standard_neurons, output_neurons):
		self.fired = False
		if random.uniform(0,self.current_state) < self.resting_frequency:
			self.fire(standard_neurons, output_neurons)
			self.fired = True
			self.value = 0
			self.current_state = 1

		self.current_state += self.decay_rate
		if self.current_state > 1.0:
			self.current_state = 1.0

	def fire(self, standard_neurons, output_neurons):
		for output_id in self.output_ids:
			for st_neuron in standard_neurons:
				if output_id[1] == st_neuron.id and output_id[0] == "standard":
					st_neuron.recieve_fire()
					self.fired_to_ids.append(st_neuron.id)
			for out_neuron in output_neurons:
				if output_id[1] == out_neuron.id and output_id[0] == "output":
					out_neuron.recieve_fire()
					self.fired_to_ids.append(out_neuron.id)

	def recieve_fire(self):
		self.current_state -= self.transmition_value
		# print("standard neuron current_state == ", self.current_state)

####################################################################
class input_neuron():
	def __init__(self, pose, id_):
		self.input_weights = []
		self.output_ids = []
		self.emition_threshold = 10
		self.pose = pose
		self.id = id_
		self.fired_to_ids = []
		self.fired = False

	def update(self, frequency, standard_neurons):
		self.fired = False
		self.fired_to_ids = []
		if random.uniform(0,1) < frequency:
			self.fire(standard_neurons)
			self.fired = True

	def fire(self, standard_neurons):
		for output_id in self.output_ids:
			for st_neuron in standard_neurons:
				if output_id == st_neuron.id:
					st_neuron.recieve_fire()
					self.fired_to_ids.append(st_neuron.id)

####################################################################
class output_neuron():
	def __init__(self, pose, id_):
		self.input_weights = []
		self.output_ids = []
		self.resting_frequency = 0.5
		self.pose = pose
		self.id = id_
		self.decay_rate = 0.025
		self.current_state = 1.0
		self.fired = False

	def update(self):
		self.fired = False
		if random.uniform(0,self.current_state) < self.resting_frequency:
			self.fired = True
			self.value = 0
			self.current_state = 1

		self.current_state += self.decay_rate
		if self.current_state > 1.0:
			self.current_state = 1.0

		return self.fired

	def recieve_fire(self):
		self.current_state -= 0.1
