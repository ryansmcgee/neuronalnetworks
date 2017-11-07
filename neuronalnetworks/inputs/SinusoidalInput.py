from NetworkInput import NetworkInput
import numpy as numpy

class SinusoidalInput(NetworkInput):

	def __init__(self, period=2*numpy.pi, amplitude=1.0):

		NetworkInput.__init__(self)

		self.period 	= period
		self.amplitude 	= amplitude


	def val(self, t):
		return self.amplitude * numpy.sin((2*numpy.pi/self.period)*t)


	def set_amplitude(self, amplitude):
		self.amplitude = amplitude


	def set_period(self, period):
		self.period = period