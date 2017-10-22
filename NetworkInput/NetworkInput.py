from abc import ABCMeta, abstractmethod

class NetworkInput(object):

	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def val(self, t):
		""" """
		pass
