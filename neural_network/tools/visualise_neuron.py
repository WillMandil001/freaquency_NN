import matplotlib.pyplot as plt
import numpy as np

def show_neuron_spiking_plot(neuron_state_list, fired, recieved_fire):
	fig=plt.figure()
	ax = plt.subplot(111)
	plt.ylim(-1.5, 1.5)

	plt.plot(neuron_state_list)
	plt.grid(True)

	y_ticks = np.arange(-1.5, 1.5, 0.1)
	plt.yticks(y_ticks)

	x_ticks = np.arange(0, len(recieved_fire), 1)
	plt.xticks(x_ticks)

	fired_list = [i for i, value in enumerate(fired) if value]
	for fire in fired_list:
		plt.axvline(x=fire, c="r")
	for i, txt in enumerate(recieved_fire):
		ax.annotate(txt, (i, 1.1))

	ax.set_xlabel('time step')
	ax.set_ylabel('neuron current state')
	ax.set_title('plot of neuron state over time')
	plt.show()