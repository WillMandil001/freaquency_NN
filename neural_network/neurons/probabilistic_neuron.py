import random


####################################################################
class neuron():
	def __init__(self, pose, id_, log_history=False):
		# self.input_weights = []
		# self.input_ids = []
		self.trained = False
		self.output_ids = []
		self.recieved_fire_from = []
		self.output_transmition_values = []  # conneciton strengths
		self.resting_frequency = 0.1  # 0.2
		self.emition_threshold = 10
		self.decay_rate = 0.1 # 0.025
		self.starting_transmition_value = 0.1
		self.pose = pose
		self.id = id_
		self.current_state = 1.0
		self.fired_to_ids = []
		self.fired = False
		self.recieved_fire = 0
		self.recieve_fired_history = []
		self.log_history = log_history
		if self.log_history:
			self.current_state_histroy = []
			self.fired_history = []
			self.log_neurons_history()

	def strengthen_conenction(self, output_id):
		self.output_transmition_values[self.output_ids.index(output_id)] += 0.1

	def weaken_connection(self, output_id):
		self.output_transmition_values[self.output_ids.index(output_id)] -= 0.1

	def update(self, standard_neurons, output_neurons):
		self.fired = False
		self.fired_to_ids = []
		self.recieved_fire_from = []

		if random.uniform(0,self.current_state) < self.resting_frequency:
			self.fire(standard_neurons, output_neurons)
			self.fired = True
			self.value = 0
			self.current_state = 1

		self.current_state += self.decay_rate
		if self.current_state > 1.0:
			self.current_state = 1.0

		if self.log_history: 
			self.log_neurons_history()
		self.recieved_fire = 0
		self.trained = False

	def fire(self, standard_neurons, output_neurons):
		for index, output_id in enumerate(self.output_ids):
			for st_neuron in standard_neurons:
				if output_id[1] == st_neuron.id and output_id[0] == "standard":
					st_neuron.recieve_fire(self.output_transmition_values[index], self.id)
					self.fired_to_ids.append(st_neuron.id)
			for out_neuron in output_neurons:
				if output_id[1] == out_neuron.id and output_id[0] == "output":
					out_neuron.recieve_fire(self.output_transmition_values[index], self.id)
					self.fired_to_ids.append(out_neuron.id)

	def recieve_fire(self, transmition_value, input_id):
		self.recieved_fire += 1
		self.current_state -= transmition_value
		self.recieved_fire_from.append(input_id)

	def log_neurons_history(self):
		self.current_state_histroy.append(self.current_state)
		self.fired_history.append(self.fired)
		self.recieve_fired_history.append(self.recieved_fire)


####################################################################
class input_neuron():
	def __init__(self, pose, id_):
		self.trained = False
		self.input_weights = []
		self.output_ids = []
		self.starting_transmition_value = 0.1
		self.output_transmition_values = []
		self.emition_threshold = 10
		self.pose = pose
		self.id = id_
		self.fired_to_ids = []
		self.fired = False

	def strengthen_conenction(self, output_id):
		self.output_transmition_values[self.output_ids.index(output_id)] += 0.1

	def weaken_connection(self, output_id):
		self.output_transmition_values[self.output_ids.index(output_id)] -= 0.1

	def update(self, frequency, standard_neurons):
		self.fired = False
		self.fired_to_ids = []
		if random.uniform(0,1) < frequency:
			self.fire(standard_neurons)
			self.fired = True

		self.trained = False

	def fire(self, standard_neurons):
		for index, output_id in enumerate(self.output_ids):
			for st_neuron in standard_neurons:
				if output_id == st_neuron.id:
					st_neuron.recieve_fire(self.output_transmition_values[index], self.id)
					self.fired_to_ids.append(st_neuron.id)


####################################################################
class output_neuron():
	def __init__(self, pose, id_, log_history=False):
		self.trained = False
		self.output_ids = []
		self.input_weights = []
		self.recieved_fire_from = []
		self.resting_frequency = 0.001
		self.pose = pose
		self.id = id_
		self.decay_rate = 0.025
		self.current_state = 1.0
		self.fired = False
		self.transmition_value = 0.334
		self.recieved_fire = 0
		self.recieve_fired_history = []
		self.log_history = log_history
		if self.log_history:
			self.current_state_histroy = []
			self.fired_history = []
			self.log_neurons_history()

	def update(self):
		self.fired = False
		self.recieved_fire_from = []
		if random.uniform(0, self.current_state) < self.resting_frequency:
			self.fired = True
			self.value = 0
			self.current_state = 1

		self.current_state += self.decay_rate
		if self.current_state > 1.0:
			self.current_state = 1.0

		if self.log_history: 
			self.log_neurons_history()
		self.recieved_fire = 0
		self.trained = False

		return self.fired

	def recieve_fire(self, transmition_value, input_id):
		self.current_state -= transmition_value
		self.recieved_fire += 1
		self.recieved_fire_from.append(input_id)

	def log_neurons_history(self):
		self.current_state_histroy.append(self.current_state)
		self.fired_history.append(self.fired)
		self.recieve_fired_history.append(self.recieved_fire)
