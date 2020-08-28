class neuron():
	def __init__(self, pose):
		self.input_weights = []
		self.resting_frequency = 5
		self.emition_threshold = 10
		self.decay_rate = 1
		self.pose = pose

class input_neuron():
	def __init__(self, pose):
		self.input_weights = []
		self.resting_frequency = 5
		self.emition_threshold = 10
		self.decay_rate = 1
		self.pose = pose

class output_neuron():
	def __init__(self, pose):
		self.input_weights = []
		self.resting_frequency = 1
		# self.emition_threshold = 10
		# self.decay_rate = 1
		self.pose = pose