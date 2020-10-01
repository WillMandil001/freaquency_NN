from build_neural_network import neural_network

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import random
import math

class simulation():
	def __init__(self, goal_state, start_state, left, right):
		self.left = left
		self.right = right
		self.goal_state = goal_state
		self.start_state = start_state
		self.current_state = start_state

	def visualise_current_state(self):
		grid = []
		[grid.append("_") for i in range(self.left, self.right)]
		grid[self.current_state] = "o"
		grid[self.goal_state] = "x"
		print(grid)

	def move(self, outputs):
		if outputs == [False, True]:
			self.current_state += 1
			if self.current_state > self.right - 1:
				self.current_state = self.right - 1
		if outputs == [True, False]:
			self.current_state -= 1
			if self.current_state < self.left:
				self.current_state = self.left

	def calculate_inputs(self):
		x = np.linspace(0, 1, self.right-self.left) 
		return [x[self.current_state], x[self.goal_state]]

####################################################################
goal_state=5
start_state=22

nn = neural_network(2,10,2)

simulation = simulation(goal_state=goal_state, start_state=start_state, left=0, right=25)
simulation.visualise_current_state()

####################################################################
for i in range(0, 200):
	inputs = simulation.calculate_inputs()
	outputs = nn.step(inputs)
	simulation.move(outputs)
	simulation.visualise_current_state()

