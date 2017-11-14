from NetworkInput import NetworkInput

class StepInput(NetworkInput):

	def __init__(self, stepVal, stepOnTime, stepOffTime=None, baseVal=0.0):

		NetworkInput.__init__(self)

		self.stepVal 		= stepVal
		self.stepOnTime 	= stepOnTime
		self.stepOffTime 	= stepOffTime if stepOffTime is not None else float("inf")
		self.baseVal	 	= baseVal


	def val(self, t):
		if(t >= self.stepOnTime and t < self.stepOffTime):
			return self.stepVal
		else:
			return self.baseVal


	def set_step_val(self, stepVal):
		self.stepVal = stepVal


	def set_step_on_time(self, stepOnTime):
		self.stepOnTime = stepOnTime


	def set_step_off_time(self, stepOffTime):
		self.stepOffTime = stepOffTime


	def set_base_val(self, baseVal):
		self.baseVal = baseVal