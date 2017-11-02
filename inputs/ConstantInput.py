from NetworkInput import NetworkInput

class ConstantInput(NetworkInput):

	def __init__(self, constVal):

		NetworkInput.__init__(self)

		self.constVal = constVal


	def val(self, t):
		return self.constVal


	def set_const_val(self, constVal):
		self.constVal = constVal