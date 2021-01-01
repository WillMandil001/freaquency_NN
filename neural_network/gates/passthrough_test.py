from build_neural_network import neural_network

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import random
import math

class passthrough_test():
	def __init__(self):
		self.consistency_time_steps = 20
		self.same = 0
		self.value = [bool(random.getrandbits(1))]

	def find_correct_pipelines(self, fitnesses, nn_structure):
		'''
		Finds the correct sequence of neurons firing that caused the correct ouput.
		Hyper parameters are:
			1. Time difference between n-1 firing and n firing.
		pipeline is defined as a list of [prev_neuron_id, next_neuron_id] -- basically the connection that needs strengthening
		'''
		pipelines = []
		pipeline_st = []
		pipeline_st_holder = []
		pipeline_in = []
		pipeline_in_holder = []

		for fitness, time_step in enumerate(fitnesses):
			if fitness:
				for t in range(time_step + 1, -1, -1):
					if t == time_step + 1:
						### output neuron stage:
						for out_neuron in nn_structure[t][2]:
							if out_neuron.fired == True:
								for st_neuron in nn_structure[t - 1][1]:
									if st_neuron.fired == True and ["output", out_neuron.id] in st_neuron.output_ids and st_neuron.trained == False:
										pipeline_st_holder.append(st_neuron)				
										pipelines.append([1, st_neuron.id, ["output", out_neuron.id], t, True])
					else:
						if pipeline_st != []:
							for neuron in pipeline_st:
								for st_neuron in nn_structure[t - 1][1]:  # FOR STRENGTHENING
									if st_neuron.fired == True and ["standard",neuron.id] in st_neuron.output_ids and st_neuron.trained == False:
										pipeline_st_holder.append(st_neuron)
										pipelines.append([1, st_neuron.id, ["standard", neuron.id], t, True])
								for in_neuron in nn_structure[t - 1][0]:
									if in_neuron.fired == True and neuron.id in in_neuron.output_ids and st_neuron.trained == False:
										pipelines.append([0, in_neuron.id, neuron.id, t, True])
								for st_neuron in nn_structure[t + 1][1]:  # FOR WEAKENING
									if st_neuron.fired == True and ["standard",neuron.id] in st_neuron.output_ids and st_neuron.trained == False:
										pipeline_st_holder.append(st_neuron)
										pipelines.append([1, st_neuron.id, ["standard", neuron.id], t, False])
								for in_neuron in nn_structure[t + 1][0]:
									if in_neuron.fired == True and neuron.id in in_neuron.output_ids and st_neuron.trained == False:
										pipelines.append([0, in_neuron.id, neuron.id, t, False])
						else:
							break
					pipeline_st = pipeline_st_holder

		return pipelines

	def fitness(self, input_, output):
		if input_[0] and output[0] and not output[1]:
			return True
		elif not input_[0] and output[1] and not output[0]:
			return True
		return False

	def calculate_inputs(self):
		if self.same < self.consistency_time_steps:
			self.same += 1
			return self.value
		else:
			self.same = 0
			self.value = [bool(random.getrandbits(1))]
			return self.value
