from NetworkInput import NetworkInput

class LinearInput(NetworkInput):

	def __init__(self, initVal, slope):

		NetworkInput.__init__(self)

		self.initVal 	= initVal
		self.slope 		= slope

		self.currentVal = self.initVal


	def val(self, t):
		return self.currentVal + t*self.slope


	def set_current_val(self, val):
		self.currentVal = val


	def set_slope(self, slope):
		self.slope = slope