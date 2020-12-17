from build_neural_network import neural_network

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import random
import math

class passthrough_test():
	def __init__(self):
		pass

	def fitness(self, input_, output):
		if output and input_:
			return True
		elif output == False and input_ == False:
			return True
		else
			return False

	def calculate_inputs(self):
		return bool(random.getrandbits(1))
