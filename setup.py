from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='neuronalnetworks',
   version='0.0',
   description='Facilitates construction and simulation of dynamical neuronal networks.',
   license="MIT",
   long_description=long_description,
   author='Ryan S McGee',
   author_email='ryansmcgee@gmail.com',
   url="https://github.com/ryansmcgee/neuronalnetworks",
   packages=['neuronalnetworks'],  #same as name
   install_requires=['numpy', 'matplotlib'], #external packages as dependencies
   scripts=[
            'demos/tutorial.py'
           ]
)