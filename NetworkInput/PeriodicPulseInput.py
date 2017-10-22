from NetworkInput import NetworkInput

class PeriodicPulseInput(NetworkInput):

	def __init__(self, pulseVal, pulsePeriod):

		NetworkInput.__init__(self)

		self.pulseVal = pulseVal

		self.pulsePeriod = pulsePeriod


	def val(self, t):
		return self.pulseVal if (t%self.pulsePeriod <= 0.0000001) else 0.0


	def set_pulse_val(self, pulseVal):
		self.pulseVal = pulseVal


	def set_pulse_period(self, pulsePeriod):
		self.pulsePeriod = pulsePeriod