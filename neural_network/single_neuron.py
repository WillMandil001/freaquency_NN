import random

####################################################################
class neuron():
	def __init__(self, pose, id_):
		self.input_weights = []
		self.input_ids = []
		self.resting_frequency = 0.2
		self.emition_threshold = 10
		self.decay_rate = 0.025
		self.pose = pose
		self.id = id_
		self.fired = False
		self.current_state = 1.0

	def update(self, standard_neurons, output_neurons):
		self.fired = False
		if random.uniform(0,self.current_state) < self.resting_frequency:
			self.fire(standard_neurons, output_neurons)
			self.value = 0
			self.current_state = 1

		self.current_state += self.decay_rate
		if self.current_state > 1.0:
			self.current_state = 1.0

	def recieve_fire(self):
		self.current_state -= 0.1
		# print("standard neuron current_state == ", self.current_state)

	def fire(self, standard_neurons, output_neurons):
		self.fired = True
		for neuron in standard_neurons:
			for input_ids__ in neuron.input_ids:		 
				if input_ids__[1] == self.id:
					neuron.recieve_fire()

		for neuron in output_neurons:
			for input_ids__ in neuron.input_ids:		 
				if input_ids__[1] == self.id:
					neuron.recieve_fire()


####################################################################
class input_neuron():
	def __init__(self, pose, id_):
		self.input_weights = []
		self.input_ids = []
		self.emition_threshold = 10
		self.pose = pose
		self.id = id_
		self.fired = False

	def update(self, frequency, standard_neurons):
		self.fired = False
		if random.uniform(0,1) < frequency:
			self.fire(standard_neurons)

	def fire(self, standard_neurons):
		self.fired = True
		for neuron in standard_neurons:
			for input_ids__ in neuron.input_ids:		 
				# print(neuron.input_ids[0])
				if input_ids__[1] == self.id:
					neuron.recieve_fire()

####################################################################
class output_neuron():
	def __init__(self, pose, id_):
		self.input_weights = []
		self.input_ids = []
		self.resting_frequency = 0.5
		self.pose = pose
		self.id = id_
		self.decay_rate = 0.025
		self.current_state = 1.0

	def update(self):
		self.fired = False
		if random.uniform(0,self.current_state) < self.resting_frequency:
			self.fire()
			self.value = 0
			self.current_state = 1

		self.current_state += self.decay_rate
		if self.current_state > 1.0:
			self.current_state = 1.0

		return self.fired

	def recieve_fire(self):
		self.current_state -= 0.1
		# print("standard neuron current_state == ", self.current_state)

	def fire(self):
		self.fired = True
		# print("output neuron " + str(self.id) + " FIRED")
