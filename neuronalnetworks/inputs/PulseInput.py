from NetworkInput import NetworkInput

class PulseInput(NetworkInput):

	def __init__(self, pulseVal, pulsePeriod, pulseDuration=1, baseVal=0.0):

		NetworkInput.__init__(self)

		self.pulseVal 		= pulseVal
		self.pulsePeriod 	= pulsePeriod
		self.pulseDuration 	= pulseDuration
		self.baseVal	 	= baseVal

		self.pulseDurationCounter	= 0


	def val(self, t):
		if(t%self.pulsePeriod <= 0.0000001):
			self.pulseDurationCounter = self.pulseDuration

		if(self.pulseDurationCounter > 0):
			self.pulseDurationCounter -= 1
			return self.pulseVal
		else:
			self.pulseDurationCounter = 0
			return self.baseVal


	def set_pulse_val(self, pulseVal):
		self.pulseVal = pulseVal


	def set_pulse_period(self, pulsePeriod):
		self.pulsePeriod = pulsePeriod


	def set_pulse_duration(self, pulseDuration):
		self.pulseDuration = pulseDuration


	def set_base_val(self, baseVal):
		self.baseVal = baseVal